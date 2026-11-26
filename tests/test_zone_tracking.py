from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nearfield import RANGES
from zone_tracking import (
    EXAMPLES_PER_TRIAL, FEATURES, ROOM_FRAMES, ROOM_WARMUP, SETTLE_FRAMES,
    TRIALS, VALIDATION_START, WINDOW, ZoneTracker, fit_profile, load_profile,
)


class ZoneTrackingTests(unittest.TestCase):
    def test_guided_capture_saves_validated_profile_and_clears_on_signal_loss(self):
        identity = {"devices": ["test mic", "test speakers"], "channels": 2}
        rng = np.random.default_rng(738)
        valid = np.array([[True, False], [False, True]])
        base = np.full((2, 2, len(RANGES)), .04+0.01j)
        tracker = ZoneTracker(identity, learning=True)

        def feed(target, frames):
            for _ in range(frames):
                iq = base.copy()
                if target:
                    width = .026
                    center = .53+rng.uniform(-.022, .022)
                    offset = .039*(target-2)
                    iq[0, 0] += .20*np.exp(-.5*((RANGES-center-offset)/width)**2)
                    iq[1, 1] += .14*np.exp(-.5*((RANGES-center+offset)/width)**2)
                iq += rng.normal(0, .001, iq.shape)
                tracker.update(iq, valid)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"tracker.json"
            with patch("zone_tracking.PROFILE_PATH", path):
                feed(0, ROOM_WARMUP+ROOM_FRAMES)
                self.assertTrue(tracker.ready)
                for trial, label in enumerate(TRIALS):
                    tracker.begin()
                    feed(label, SETTLE_FRAMES+WINDOW)
                    if trial == 0:
                        tracker.update(base, np.zeros_like(valid))
                        self.assertEqual(len(tracker.capture), 1)
                    feed(label, WINDOW*(EXAMPLES_PER_TRIAL-1))
                self.assertTrue(tracker.finished)
                self.assertTrue(tracker.saved)
                self.assertEqual(tracker.validation, 1.)
                self.assertIsNone(load_profile({"channels": 1}))
                tracker = ZoneTracker(identity, profile=load_profile(identity))
                feed(0, ROOM_WARMUP+ROOM_FRAMES)
                feed(1, 2*WINDOW)
                self.assertEqual(tracker.snapshot(True)["position"], "Left")
                tracker.update(base, np.zeros_like(valid))
                self.assertIsNone(tracker.snapshot(True)["position"])

    def test_rejects_when_separate_round_does_not_match_training(self):
        patterns = np.zeros((4, FEATURES))
        for i in range(4):
            patterns[i, 6*i:6*i+6] = 1.
        trials = [np.repeat(patterns[label if index < VALIDATION_START else (label+1) % 4][None],
                            EXAMPLES_PER_TRIAL, axis=0)
                  for index, label in enumerate(TRIALS)]
        result, passed = fit_profile(trials, {"channels": 2})
        self.assertEqual(result["validation_accuracy"], 0.)
        self.assertFalse(passed)
