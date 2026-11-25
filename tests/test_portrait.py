import unittest

import cv2
import numpy as np

from main import Worker
from nearfield import FRAME_RATE, PERIOD, NearField, NearFieldDemo
from sonar_settings import LOW_BAND
from visualizer_cam import render_portrait


class PortraitTests(unittest.TestCase):
    def test_qualified_synthetic_echo_drives_model(self):
        cv2.setNumThreads(1)
        camera = NearField(2, LOW_BAND.carriers, settings=LOW_BAND)
        simulated = NearFieldDemo(camera.probe)
        for i in range(FRAME_RATE):
            camera.analyze(simulated.block(i*PERIOD, PERIOD), 90, (i+1)/FRAME_RATE)
        worker = Worker(None, nearfield=camera)
        worker.active = True
        snap = worker.snapshot()
        self.assertIsNotNone(snap["subject"])
        self.assertAlmostEqual(snap["subject"][1], .62, delta=.06)
        snap["depth"].fill(0)
        health = dict(reason="synthetic", remaining=1, simulated=True)
        active = render_portrait(snap, health, "pure")
        self.assertEqual(active.shape, (720, 1280, 3))
        self.assertGreater(np.count_nonzero(active.max(axis=2) >= 35), 92160)
        snap["subject"] = None
        self.assertFalse(render_portrait(snap, health, "pure").any())
