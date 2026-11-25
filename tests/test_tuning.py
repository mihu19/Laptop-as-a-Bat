import tempfile
import unittest
from pathlib import Path

import numpy as np

from nearfield import RANGES
from scripts.tune_and_verify import assess, foreground, tune
from sonar_settings import ML_BAND, SonarSettings, load_settings, save_settings


class TuningTests(unittest.TestCase):
    def test_blank_frames_do_not_count_as_stable_foreground(self):
        mask, bounds = foreground(np.zeros((90, 160, 3), np.uint8))
        self.assertFalse(mask.any())
        self.assertIsNone(bounds)
        row = self.sample()
        row.update(echo=0., locked=False, energy=np.zeros(2), bpm=None)
        scored = assess([row], [0.], [0.], [0.])
        for check in ("iou", "snr", "stereo", "breathing"):
            self.assertFalse(scored["checks"][check])
        self.assertGreater(tune(SonarSettings(), scored).gain, SonarSettings().gain)

    def test_wave_updates_and_separate_signal_checks(self):
        rows = [dict(self.sample(), wave=.05+i/10000) for i in range(35)]
        self.assertTrue(all(assess(rows, [.1]*35, [.81]*34, [.5]*35)["checks"].values()))
        self.assertFalse(assess([self.sample()]*35, [.1], [.81], [.5])["bpm_valid"])
        one_sided = [dict(row, energy=np.array([1., 0.])) for row in rows]
        self.assertFalse(assess(one_sided, [.1], [.81], [.5])["checks"]["stereo"])
        self.assertFalse(assess(rows, [.1], [.9], [.95])["checks"]["coverage"])

    def test_configuration_round_trip_and_peak_validation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"sonar_config.json"
            self.assertEqual(load_settings(path), ML_BAND)
            settings = SonarSettings(gain=.007, persistence=.8, low_hz=17000, high_hz=22000)
            save_settings(settings, path)
            self.assertEqual(load_settings(path), settings)
            path.write_text('{"gain": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(path)

    @staticmethod
    def sample():
        return dict(echo=.16, noise=.001, energy=np.ones(2), locked=True,
                    cir=np.zeros_like(RANGES), grid=np.zeros((len(RANGES), 64)),
                    bpm=18., wave=.05)
