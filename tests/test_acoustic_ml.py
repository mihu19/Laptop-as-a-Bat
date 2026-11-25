import unittest

import cv2
import numpy as np

from main import Worker
from ml_acoustic_model import AcousticModel, load_model
from nearfield import NearField, NearFieldDemo, PERIOD, RANGES
from sonar_settings import ML_BAND
from visualizer_cam import render_echo_map, render_ml


class AcousticMLTests(unittest.TestCase):
    def test_sonar_trail_uses_measured_echoes_and_clears_on_mute_or_timing_loss(self):
        near = NearField(2, ML_BAND.carriers, settings=ML_BAND)
        model = AcousticModel(load_model())
        worker = Worker(None, nearfield=near, model=model)
        worker.active = True
        health = {"simulated": True}
        plot = np.s_[192:555, 116:813]
        empty = render_echo_map(worker.snapshot(), health)[plot].copy()

        model.grid = np.ones_like(near.grid)
        demo = NearFieldDemo(near.probe)
        for i in range(25):
            near.analyze(demo.block(i*PERIOD, PERIOD), 90, (i+1)/25)
            worker._publish()
        snapshot = worker.latest
        active = render_echo_map(snapshot, health)
        self.assertEqual(active.shape, (720, 1280, 3))
        self.assertTrue(active.flags.c_contiguous)
        self.assertGreater(np.count_nonzero(active[plot] != empty), 1000)
        snapshot["depth"].fill(0)
        np.testing.assert_array_equal(render_echo_map(snapshot, health), active)

        for unqualified in (dict(state="muted"), dict(depth_locked=False)):
            with self.subTest(fields=tuple(unqualified)):
                cleared = render_echo_map(dict(snapshot, **unqualified), health)
                np.testing.assert_array_equal(cleared[plot], empty)

    def test_ambient_display_is_visible_without_fabricating_presence(self):
        cv2.setNumThreads(1)
        near = NearField(2, ML_BAND.carriers, settings=ML_BAND)
        model = AcousticModel(load_model())
        worker = Worker(None, nearfield=near, model=model)
        for _ in range(12):
            model.observe(near, RANGES)
        snap = worker.snapshot()
        self.assertIsNone(snap["subject"])
        self.assertEqual(snap["ml_state"], "No_Human")
        frame = render_ml(snap, {"reason": "muted", "simulated": False}, "hud")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.assertGreater(gray.mean(), 30)
        self.assertTrue(.15 <= np.mean(frame.max(axis=2) >= 35) <= .65)

    def test_trained_model_accepts_measured_simulated_torso(self):
        near = NearField(2, ML_BAND.carriers, settings=ML_BAND)
        model = AcousticModel(load_model())
        demo = NearFieldDemo(near.probe)
        count = 0
        for i in range(23):
            near.analyze(demo.block(i*PERIOD, PERIOD), 90, (i+1)/25)
            model.observe(near, RANGES)
            if model.subject is not None:
                count += 1
        self.assertGreater(count, 5)
        self.assertAlmostEqual(model.subject[1], .62, delta=.10)
        self.assertGreater(model.weights["metrics"]["synthetic_validation_accuracy"], .92)
        gestures = set()
        for i in range(23, 145):
            near.analyze(demo.block(i*PERIOD, PERIOD), 90, (i+1)/25)
            model.observe(near, RANGES)
            if model.label.startswith("Hand_"):
                gestures.add(model.label)
        self.assertEqual(gestures, {"Hand_Push", "Hand_Pull", "Hand_Swipe_Left", "Hand_Swipe_Right"})
        near.reset()
        model.clear()
        self.assertIsNone(model.subject)
