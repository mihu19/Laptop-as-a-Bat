"""End-to-end DSP tests from microphone PCM, including calibration and timing."""

import unittest

import numpy as np

from audio_engine import CARRIERS, HOP, RATE
from dsp_processor import DSPProcessor


class Scene:
    """Independent echo generator with unequal path gains and microphone noise."""

    def __init__(self, seed=91, channels=2, drift=0.0):
        self.rng = np.random.default_rng(seed)
        self.dsp = DSPProcessor(channels, calibration=0.2, quiet=0.6)
        self.phase = self.rng.uniform(-np.pi, np.pi, (2, channels))
        self.gain = np.array([[0.045, 0.032], [0.026, 0.040]])[:, :channels]
        self.drift = drift
        self.run(0.2, active=False)
        self.run(0.6)

    def run(self, duration, *, active=True, shift=0, ends=(0, 0), amplitude=0.014,
            bpm=None, modulation="phase", acceleration=0, silence=False):
        count = round(duration*RATE/HOP)
        for k in range(count):
            sample = np.arange(HOP) + self.dsp.expected
            t = sample/RATE
            u = (k + np.arange(HOP)/HOP) / count
            x = ends[0] + (ends[1]-ends[0])*u
            envelope = np.minimum(1, np.minimum(u, 1-u)*12)
            data = self.rng.normal(0, 0.000035, (HOP, self.dsp.channels))
            if active and not silence:
                for i, carrier in enumerate(CARRIERS):
                    theta = 2*np.pi*(carrier+self.drift*(carrier/CARRIERS[0]))*t[:, None]
                    theta = theta + self.phase[i]
                    data += self.gain[i]*np.cos(theta) + 0.007*np.cos(theta + 1.1)
                    if bpm is not None:
                        chest = np.sin(2*np.pi*bpm/60*t[:, None])
                        if modulation == "phase":
                            data += 0.007*(np.cos(theta+1.1+0.48*chest)-np.cos(theta+1.1))
                        else:
                            data += 0.003*chest*np.cos(theta)
                    if shift:
                        side = (1 + (1 if i else -1)*x)
                        phi = 2*np.pi*(shift*t + 0.5*acceleration*(u*duration)**2)
                        data += (amplitude*envelope*side)[:, None]*np.cos(theta+phi[:, None])
            self.dsp.process(data.astype(np.float32), active)

    def zones(self):
        for zone, pos in (("left", -0.85), ("center", 0), ("right", 0.85)):
            assert self.dsp.mark_zone(zone)
            self.run(0.12)
            self.run(0.85, shift=147, ends=(pos, pos))
            self.run(1.06)
        assert self.dsp.cue_maps


class SensingTests(unittest.TestCase):
    def test_all_gestures_random_order_and_debounce(self):
        scene = Scene()
        scene.zones()
        rng = np.random.default_rng(154)
        labels = ["Push", "Pull", "Swipe Left", "Swipe Right", "Double Tap"]
        hits = 0
        for _ in range(5):
            for name in rng.permutation(labels):
                before = scene.dsp.expected/RATE
                offset = rng.uniform(115, 235)
                gain = rng.uniform(0.009, 0.017)
                if name == "Double Tap":
                    for j in range(2):
                        scene.run(0.09, shift=offset, amplitude=gain)
                        scene.run(0.09, shift=-offset, amplitude=gain)
                        if j == 0:
                            scene.run(0.18)
                elif name.startswith("Swipe"):
                    sign = 1 if name.endswith("Left") else -1
                    scene.run(rng.uniform(0.35, 0.55), shift=-offset,
                              ends=(0.95*sign, -0.95*sign), amplitude=gain, acceleration=28)
                else:
                    scene.run(rng.uniform(0.17, 0.35), shift=offset if name=="Push" else -offset,
                              amplitude=gain, acceleration=34)
                scene.run(0.7)
                found = [g for g in scene.dsp.gestures.log if g.time >= before]
                self.assertEqual([g.name for g in found], [name])
                self.assertGreater(found[0].confidence, 0.5)
                hits += 1
        print(f"Synthetic gestures: {hits}/25, one event per trial.")

    def test_breathing_rates_from_raw_echoes(self):
        for bpm, mode in ((6, "phase"), (12, "phase"), (15, "phase"),
                          (15, "amplitude"), (24, "amplitude"), (36, "phase")):
            with self.subTest(bpm=bpm, modulation=mode):
                scene = Scene(seed=10+bpm, channels=1, drift=0.09)
                scene.run(72, bpm=bpm, modulation=mode)
                snap = scene.dsp.snapshot()
                self.assertEqual(snap["breath_state"], "tracking")
                self.assertLess(abs(snap["bpm"]-bpm), 1.0)
                self.assertGreater(snap["quality"], 0.7)
                self.assertEqual(len(snap["waveform"]), 1200)
                self.assertFalse(snap["events"])
                print(f"{mode} echo {bpm:2d} BPM -> {snap['bpm']:.3f}")
                scene.run(0.15, shift=145)
                self.assertIsNone(scene.dsp.breath.bpm)
                scene.run(0.2, active=False)
                self.assertEqual(scene.dsp.breath.state, "muted")

    def test_radar_axes_and_calibration_gate(self):
        scene = Scene(channels=1)
        scene.run(0.2, shift=160, ends=(-0.9, -0.9))
        np.testing.assert_allclose(scene.dsp.shifts[0], 160, atol=9)
        self.assertGreaterEqual(scene.dsp.bandwidths[0, 1], 150)
        self.assertIsNone(scene.dsp.target)
        self.assertGreater(scene.dsp.radar.max(), 0)
        self.assertTrue(np.allclose(scene.dsp.radar[:, 0], scene.dsp.radar[:, -1]))
        scene.run(0.6)
        scene.zones()
        for pos, shift in ((-0.85, 145), (0.85, -155)):
            scene.run(0.2, shift=shift, ends=(pos, pos))
            x, row = scene.dsp.target
            self.assertGreater(x * pos, 0.2)
            self.assertGreater((40-row)*shift, 0)
            scene.run(0.7)

    def test_loss_and_discontinuous_input(self):
        scene = Scene()
        scene.run(1.0, silence=True)
        self.assertEqual(scene.dsp.breath.state, "weak signal")
        scene.run(0.1)
        scene.dsp.process(np.zeros((HOP, 2)), True, start=scene.dsp.expected+HOP)
        self.assertEqual(scene.dsp.state, "quiet carrier calibration")
        self.assertIsNone(scene.dsp.breath.bpm)

    def test_ambient_requires_contiguous_muted_audio(self):
        d = DSPProcessor()
        zeros = np.zeros((HOP, 2), np.float32)
        for _ in range(299):
            d.process(zeros, False)
        self.assertFalse(d.ready)
        d.process(zeros, True)
        for _ in range(299):
            d.process(zeros, False)
        self.assertFalse(d.ready)
        d.process(zeros, False)
        self.assertTrue(d.ready)

    def test_no_breathing_or_gestures_from_static_scene(self):
        scene = Scene()
        scene.run(45)
        scene.run(0.09, shift=160)
        scene.run(0.09, shift=-160)
        scene.run(0.8)
        self.assertIsNone(scene.dsp.breath.bpm)
        self.assertFalse(scene.dsp.gestures.log)

    def test_unrelated_ambient_pilot_cannot_qualify_a_carrier(self):
        dsp = DSPProcessor(channels=1, calibration=0.3, quiet=0.4)
        for hop in range(100):
            t = (np.arange(HOP) + hop*HOP) / RATE
            data = 0.02*np.cos(2*np.pi*CARRIERS[0]*t)
            if hop >= 30:
                data += 0.035*np.cos(2*np.pi*CARRIERS[1]*t)
            dsp.process(data[:, None], hop >= 30)
        self.assertEqual(dsp.usable.ravel().tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
