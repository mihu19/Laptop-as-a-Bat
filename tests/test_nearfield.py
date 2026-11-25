"""Range-gate, instant acquisition and near-field motion regressions."""

import unittest

import numpy as np

from audio_engine import CARRIERS, HOP, MAX_GAIN, Transmitter
from nearfield import FRAME_RATE, METRES_PER_SAMPLE, PERIOD, RANGES, NearField, NearFieldDemo, TorsoRate


class NearFieldTests(unittest.TestCase):
    def scene(self, distances, count=10):
        camera = NearField(2, CARRIERS)
        spectrum = np.fft.fft(camera.probe.wave, axis=0)
        phase = -2j*np.pi*np.fft.fftfreq(PERIOD)
        paths = .04*np.exp(92*phase)
        for distance in distances:
            paths += .012*np.exp(phase*(92+(distance-.075)/METRES_PER_SAMPLE))
        audio = np.fft.ifft(spectrum*paths[:, None], axis=0).real
        for i in range(count):
            camera.analyze(audio, 0, (i+1)/FRAME_RATE)
        return camera

    def test_stationary_near_echo_without_empty_room_calibration(self):
        camera = self.scene([.44])
        self.assertTrue(camera.locked)
        self.assertLess(camera.first_lock, .5)
        row, _ = np.unravel_index(camera.grid.argmax(), camera.grid.shape)
        self.assertAlmostEqual(RANGES[row], .44, delta=.015)
        self.assertGreater(camera.grid.max(), .4)

    def test_close_and_far_echoes_do_not_enter_trackers(self):
        for distance in (.10, .13, .952, 1.10, 1.60):
            with self.subTest(distance=distance):
                camera = self.scene([distance])
                self.assertTrue(camera.locked)
                self.assertFalse(camera.grid.any())
                self.assertFalse(camera.events)
                self.assertIsNone(camera.breath.cell)

    def test_noise_without_transmitted_reference(self):
        camera = NearField(2, CARRIERS)
        random = np.random.default_rng(8)
        for i in range(25):
            camera.analyze(random.normal(0, .0001, (PERIOD, 2)), 0, i/FRAME_RATE)
        self.assertFalse(camera.locked)
        self.assertFalse(camera.grid.any())

    def test_mti_fades_stationary_echo(self):
        early = self.scene([.60], count=10)
        late = self.scene([.60], count=240)
        self.assertLess(late.grid.max(), .15*early.grid.max())

    def test_breath_and_four_gestures_from_stereo_pcm(self):
        camera = NearField(2, CARRIERS)
        demo = NearFieldDemo(camera.probe)
        for i in range(14*FRAME_RATE):
            camera.analyze(demo.block(i*PERIOD, PERIOD), 0, (i+1)/FRAME_RATE)
        self.assertEqual([e.name for e in camera.events],
                         ["Push", "Pull", "Swipe Right", "Swipe Left"])
        self.assertIsNotNone(camera.breath.bpm)
        self.assertAlmostEqual(camera.breath.bpm, 19.2, delta=1)

    def test_chirp_output_respects_peak_and_mute(self):
        tx = Transmitter(gain=MAX_GAIN, imaging="chirp")
        out = np.zeros((HOP, 2), np.float32)
        tx.arm()
        for i in range(12):
            self.assertTrue(tx.fill(out, i*HOP))
            self.assertLessEqual(abs(out).max(), MAX_GAIN+1e-8)
            self.assertFalse(out[:, 1-i//2%2].any())
        tx.mute()
        self.assertFalse(tx.fill(out, 12*HOP))
        self.assertFalse(out.any())

    def test_breath_lock_requires_enough_cycles(self):
        for bpm, expected in ((18, (7, 8)), (8, (14, 19))):
            with self.subTest(bpm=bpm):
                track = TorsoRate()
                first = None
                for i in range(19*FRAME_RATE):
                    t = i/FRAME_RATE
                    iq = np.exp(1j*.45*np.sin(2*np.pi*bpm*t/60))
                    track.push(t, (0, 0, .58), iq)
                    if track.bpm is not None:
                        first = t
                        break
                self.assertIsNotNone(first)
                self.assertTrue(expected[0] <= first <= expected[1], first)
                self.assertAlmostEqual(track.bpm, bpm, delta=1)
