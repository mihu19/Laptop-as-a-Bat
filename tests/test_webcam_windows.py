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
                               "--duration", "8"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) as producer:
            cap = None
            try:
                time.sleep(2)
                deadline = time.monotonic()+4
                while time.monotonic() < deadline:
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        break
                    cap.release()
                    time.sleep(0.15)
                self.assertTrue(cap.isOpened(), "Could not open the selected DirectShow input")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                previous = None
                changes = 0
                for _ in range(45):
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape, (720, 1280, 3))
                    stamp = frame[17:52, 630:815].astype(np.float32)
                    if previous is not None and abs(stamp-previous).mean() > 0.05:
                        changes += 1
                    previous = stamp
                self.assertGreater(changes, 7)
                self.assertGreater(frame[15:56, 26:200].std(), 15)
            finally:
                if cap is not None:
                    cap.release()
                out, err = producer.communicate(timeout=18)
            self.assertEqual(producer.returncode, 0, err)
            report = json.loads(out)
            self.assertEqual(report["camera"], "OBS Virtual Camera")
            self.assertGreater(report["sent"], 150)
            self.assertEqual(report["output_seconds"], 0)
            print(f"DirectShow received 45 frames at 1280x720, {changes} timestamp changes.")
