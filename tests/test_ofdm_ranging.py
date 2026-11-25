import unittest

import numpy as np

from audio_engine import Block, HOP, RATE, Transmitter
from main import Worker
from nearfield import METRES_PER_SAMPLE
from ofdm_ranging import OfdmSounder, OfdmTracker, SYMBOL_SAMPLES
from visualizer_cam import render_echo_map


class OfdmRangingTests(unittest.TestCase):
    def test_tracks_reflector_from_pcm_and_clears_on_mute(self):
        sounder = OfdmSounder(HOP)
        tracker = OfdmTracker(sounder, 1, .15)
        worker = Worker(None, recording=np.empty(0), ofdm=tracker)
        spectrum = np.fft.rfft(sounder.symbol)
        phase = -2j*np.pi*np.fft.rfftfreq(SYMBOL_SAMPLES)
        room = np.exp(phase*101.3) + .2*np.exp(phase*269.7)
        rng = np.random.default_rng(23)
        start = 0

        def feed(periods, distance=None, active=True, quiet=False):
            nonlocal start
            path = room
            if distance is not None:
                delay = 101.3+(distance-.075)/METRES_PER_SAMPLE
                path = room+.19*np.exp(phase*delay)
            signal = np.fft.irfft(spectrum*path)*(.01 if active and not quiet else 0)
            for _ in range(periods):
                samples = np.tile(signal, len(sounder.wave)//SYMBOL_SAMPLES)
                samples += rng.normal(0, .00003, len(samples))
                for i in range(0, len(samples), HOP):
                    worker._ofdm(Block(samples[i:i+HOP, None], start, active, False, 0, 0))
                    start += HOP
                worker._publish()

        feed(tracker.calibration_frames+4)
        blank = worker.latest
        self.assertTrue(blank["room_ready"])
        self.assertIsNone(blank["tracked_range"])
        for distance in (.72, .46):
            feed(6, distance)
            self.assertAlmostEqual(worker.latest["tracked_range"], distance, delta=.045)
            recent_peak = worker.latest["echo_history"][-1].argmax()
            self.assertAlmostEqual(tracker.ranges[recent_peak], distance, delta=.045)
        near = worker.latest
        visible = render_echo_map(near, {})
        hidden = render_echo_map(blank, {})
        self.assertEqual(visible.shape, (720, 1280, 3))
        self.assertGreater(np.count_nonzero(visible[192:555, 116:813] != hidden[192:555, 116:813]), 1000)

        feed(1, quiet=True)
        self.assertIsNone(worker.latest["tracked_range"])
        self.assertFalse(worker.latest["echo_history"].size)
        feed(6, .68)
        self.assertIsNotNone(worker.latest["tracked_range"])
        feed(1, active=False)
        muted = worker.latest
        self.assertFalse(muted["room_ready"])
        self.assertIsNone(muted["tracked_range"])

    def test_transmitter_repeats_ofdm_through_callback_boundaries(self):
        tx = Transmitter(imaging="ofdm")
        samples = np.zeros((HOP, 2), np.float32)
        tx.arm()
        for start in range(0, 2*len(tx.probe.wave), HOP):
            self.assertTrue(tx.fill(samples, start))
            self.assertLessEqual(abs(samples).max(), tx.gain + 1e-8)
            if start >= 2*HOP:
                offset = start % len(tx.probe.wave)
                expected = tx.gain*tx.probe.wave[offset:offset+HOP]
                np.testing.assert_allclose(samples, expected, atol=1e-9)
        tx.mute()
        self.assertFalse(tx.fill(samples, 2*len(tx.probe.wave)))
        self.assertFalse(samples.any())
