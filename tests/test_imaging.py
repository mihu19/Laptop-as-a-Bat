import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_engine import CARRIERS, HOP, MAX_GAIN, Transmitter
from dsp_processor import DSPProcessor
from sonar_imaging import DELAYS, FPS, PERIOD, RATE, RANGES, RangeCamera, demo_echo
from visualizer_cam import render_sonar


class ImagingTests(unittest.TestCase):
    def test_distances_and_static_clutter(self):
        camera = RangeCamera(2, CARRIERS)
        wave = camera.probe.wave
        room = .05*np.roll(wave, 130, axis=0) + .018*np.roll(wave, 524, axis=0)
        echo_l = np.roll(wave[:, 0], 130+100)
        echo_r = np.roll(wave[:, 1], 130+245)
        scene = room + .012*np.column_stack((echo_l+.08*echo_r, .08*echo_l+echo_r))
        for _ in range(50):
            camera.process(room, learn=True)
        for _ in range(20):
            camera.process(scene, learn=False)
        self.assertTrue(camera.locked)
        strength = camera.grid.max(axis=1)
        for delay in (100, 245):
            row = int(np.argmin(abs(DELAYS-delay)))
            nearby = np.argmax(strength[row-3:row+4])+row-3
            self.assertLess(abs(DELAYS[nearby]-delay), 9)
            self.assertGreater(strength[nearby], .30)
        wall = int(np.argmin(abs(DELAYS-394)))
        self.assertLess(strength[wall], strength.max()*.3)
        self.assertLess(camera.grid[np.argmin(abs(DELAYS-100))].argmax(), 30)
        self.assertGreater(camera.grid[np.argmin(abs(DELAYS-245))].argmax(), 34)

    def test_breathing_and_render_from_pcm(self):
        camera = RangeCamera(2, CARRIERS)
        first_lock = None
        for i in range(18*FPS):
            timestamp = i/FPS
            camera.process(demo_echo(camera.probe, i*PERIOD, PERIOD, timestamp),
                           learn=timestamp < 1.4)
            if first_lock is None and camera.breath.bpm is not None:
                first_lock = timestamp
        self.assertIsNotNone(first_lock)
        self.assertLess(first_lock, 15)
        self.assertAlmostEqual(camera.breath.bpm, 15, delta=1)
        self.assertTrue(.3 <= RANGES[camera.breath.cell[2]] <= 1)
        snapshot = DSPProcessor().snapshot()
        snapshot.update(depth=camera.grid, depth_locked=True, range_calibrated=True,
                        range_bpm=camera.breath.bpm, range_quality=camera.breath.quality,
                        range_wave=camera.breath.wave)
        health = dict(reason="muted", remaining=120, simulated=True)
        pure = render_sonar(snapshot, health, hud=False)
        frame = render_sonar(snapshot, health)
        self.assertEqual(frame.shape, (720,1280,3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(pure.std(), 10)
        self.assertGreater(np.count_nonzero(pure.max(axis=2) > 100), 2000)
        self.assertLess(np.count_nonzero(np.any(frame != pure, axis=2))/(720*1280), .15)

    def test_guided_marks_bind_to_route_and_survive_reload(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"calibration.json"
            camera = RangeCamera(2, CARRIERS, ["route"], path)
            camera.locked = True
            camera.calibrate(0)
            for t in np.arange(0, 3.05, .04):
                camera.cue = .55 if t < 1.5 else -.50
                camera.guide_step(t)
            restored = RangeCamera(2, CARRIERS, ["route"], path)
            self.assertTrue(restored.calibrated)
            self.assertEqual(restored.marks.tolist(), [.55, -.50])
            self.assertFalse(RangeCamera(2, CARRIERS, ["other"], path).calibrated)

    def test_guided_marks_from_moving_pcm(self):
        camera = RangeCamera(2, CARRIERS)
        tx = camera.probe.wave
        room = .035*np.roll(tx, 117, axis=0)
        for _ in range(38):
            camera.process(room, learn=True)
        camera.calibrate(0)
        for i in range(3*FPS+2):
            left = i < 1.5*FPS
            echo = .011*np.roll(tx[:, int(not left)], 240+round(10*np.sin(i/3)))
            side = np.array([1, .25]) if left else np.array([.25, 1])
            camera.process(room+echo[:, None]*side, learn=False)
            camera.guide_step(i/FPS)
        self.assertTrue(camera.calibrated, camera.calibration_status)
        self.assertLess(camera.marks[0], camera.marks[1])

    def test_reference_loss_clears_output(self):
        camera = RangeCamera(1, CARRIERS)
        camera.process(np.zeros((PERIOD, 1)), learn=False)
        self.assertFalse(camera.locked)
        self.assertFalse(camera.grid.any())

    def test_muting_retains_room_calibration_and_clears_measurements(self):
        camera = RangeCamera(2, CARRIERS)
        for i in range(60):
            t = i/FPS
            camera.process(demo_echo(camera.probe, i*PERIOD, PERIOD, t), learn=t < 1.4)
        self.assertGreater(camera.grid.max(), 0)
        background = camera.background.copy()
        camera.reset(keep_background=True)
        np.testing.assert_array_equal(camera.background, background)
        self.assertFalse(camera.grid.any())
        self.assertIsNone(camera.breath.bpm)
        camera.reset()
        self.assertIsNone(camera.background)


class DemoTimerTests(unittest.TestCase):
    def test_demo_timer_waits_two_seconds_and_honors_manual_mute(self):
        tx = Transmitter(max_seconds=.02, auto_cooldown=True)
        out = np.zeros((HOP, 2), np.float32)
        tx.arm()
        self.assertTrue(tx.fill(out, 0))
        self.assertTrue(tx.fill(out, HOP))
        self.assertFalse(tx.fill(out, 2*HOP))
        self.assertFalse(tx.fill(out, 2*RATE+HOP))
        self.assertTrue(tx.fill(out, 2*RATE+2*HOP))
        tx.mute()
        self.assertFalse(tx.fill(out, 4*RATE+4*HOP))

    def test_cooldown_obeys_exposure_budget_and_cap(self):
        tx = Transmitter(gain=MAX_GAIN, session=.04, max_seconds=.02,
                         auto_cooldown=True, imaging=True)
        out = np.zeros((HOP, 2), np.float32)
        self.assertTrue(tx.arm())
        self.assertTrue(tx.fill(out, 0))
        self.assertTrue(tx.fill(out, HOP))
        self.assertLessEqual(abs(out).max(), MAX_GAIN+1e-8)
        self.assertFalse(tx.fill(out, 2*HOP))
        self.assertFalse(out.any())
        self.assertFalse(tx.fill(out, 2*RATE))
        self.assertFalse(tx.fill(out, 2*RATE+2*HOP))
        self.assertTrue(tx.fill(out, 181*RATE))
        tx.mute()
        self.assertFalse(tx.fill(out, 181*RATE+HOP))

    def test_manual_reset_cannot_reset_rolling_usage(self):
        tx = Transmitter(session=.02)
        out = np.empty((HOP,2), np.float32)
        tx.arm()
        self.assertTrue(tx.fill(out, 0))
        self.assertTrue(tx.fill(out, HOP))
        self.assertFalse(tx.fill(out, 2*HOP))
        self.assertFalse(tx.arm(reset_demo=True))
        self.assertFalse(tx.fill(out, 3*HOP))
