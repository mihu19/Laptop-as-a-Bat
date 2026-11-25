"""Opt-in consumer check. Set the DirectShow index of OBS Virtual Camera first."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import cv2
import numpy as np


@unittest.skipUnless(sys.platform == "win32" and "LAPTOPBAT_OBS_INDEX" in os.environ,
                     "requires the DirectShow index of the installed OBS Virtual Camera")
class WebcamReceiverTest(unittest.TestCase):
    def test_receives_changing_720p_dashboard(self):
        index = int(os.environ["LAPTOPBAT_OBS_INDEX"])
        main = Path(__file__).resolve().parents[1] / "main.py"
        with subprocess.Popen([sys.executable, str(main), "--headless", "--simulate",
                               "--duration", "16"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) as producer:
            cap = None
            try:
                time.sleep(3)
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        break
                    cap.release()
                    time.sleep(0.15)
                self.assertTrue(cap.isOpened(), "Could not open the selected DirectShow input")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                deadline = time.monotonic()+7
                while True:
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    interior = frame[120:650, 80:1200]
                    warm = (interior[:, :, 2] > 120) & (interior[:, :, 0] < 110)
                    if np.count_nonzero(warm) > 600:
                        break
                    self.assertLess(time.monotonic(), deadline, "No sonar returns reached the receiver")
                    time.sleep(.04)
                previous = None
                changes = 0
                for _ in range(45):
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape, (720, 1280, 3))
                    stamp = frame[120:650:2, 80:1200:4].astype(np.float32)
                    if previous is not None and abs(stamp-previous).mean() > 0.035:
                        changes += 1
                    previous = stamp
                    time.sleep(.035)
                self.assertGreater(changes, 7)
                self.assertGreater(frame[120:650, 80:1200].std(), 8)
            finally:
                if cap is not None:
                    cap.release()
                out, err = producer.communicate(timeout=22)
            self.assertEqual(producer.returncode, 0, err)
            report = json.loads(out)
            self.assertEqual(report["camera"], "OBS Virtual Camera")
            self.assertGreater(report["sent"], 150)
            self.assertEqual(report["output_seconds"], 0)
            self.assertTrue(report["range_locked"])
            print(f"DirectShow received 45 frames at 1280x720, {changes} changing sonar frames.")
