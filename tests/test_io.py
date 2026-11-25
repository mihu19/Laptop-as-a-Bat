import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from scipy.io import wavfile

from audio_engine import CARRIERS, CaptureRing, HOP, MAX_GAIN, RATE, Transmitter, raw_settings
from dsp_processor import DSPProcessor
from main import parser, run, wav_input
from visualizer_cam import CameraOutput, render


class AudioTests(unittest.TestCase):
    def test_mute_carrier_routing_phase_continuity_and_cap(self):
        tx = Transmitter(gain=MAX_GAIN)
        out = np.ones((HOP, 2), np.float32)
        self.assertFalse(tx.fill(out, 0))
        self.assertFalse(out.any())
        tx.arm()
        capture = []
        for i in range(12):
            self.assertTrue(tx.fill(out, (i+1)*HOP))
            capture.append(out.copy())
        complete = np.concatenate(capture)
        self.assertLessEqual(float(abs(complete).max()), MAX_GAIN+1e-8)
        steady = complete[2*HOP:]
        expected = MAX_GAIN*np.sin(2*np.pi*np.arange(3*HOP,13*HOP)[:, None]*CARRIERS/RATE)
        np.testing.assert_allclose(steady, expected, atol=3e-8)
        freqs = np.fft.rfftfreq(len(steady), 1/RATE)
        np.testing.assert_array_equal(freqs[np.argmax(abs(np.fft.rfft(steady, axis=0)), axis=0)], CARRIERS)
        tx.mute("emergency mute")
        self.assertFalse(tx.fill(out, 13*HOP))
        self.assertFalse(out.any())

    def test_time_budget_survives_rearm_and_expires(self):
        tx = Transmitter(session=0.04)
        out = np.empty((HOP, 2), np.float32)
        tx.arm()
        for i in range(2):
            self.assertTrue(tx.fill(out, i*HOP))
        tx.mute()
        self.assertFalse(tx.fill(out, 2*HOP))
        tx.arm()
        self.assertTrue(tx.fill(out, 3*HOP))
        self.assertTrue(tx.fill(out, 4*HOP))
        self.assertFalse(tx.fill(out, 5*HOP))
        self.assertFalse(tx.arm())
        self.assertFalse(out.any())
        self.assertFalse(tx.fill(out, RATE*181))
        self.assertTrue(tx.arm())
        self.assertTrue(tx.fill(out, RATE*181+HOP))
        self.assertEqual(tx.total_emitted, 5*HOP)
        self.assertEqual(tx.used, HOP)

    def test_timeout_and_fault_fail_closed(self):
        tx = Transmitter()
        out = np.ones((HOP, 2), np.float32)
        tx.arm()
        tx.heartbeat -= 1
        self.assertFalse(tx.fill(out, 0))
        self.assertFalse(tx.armed)
        tx.arm()
        with tx.lock:
            self.assertFalse(tx.fill(out, HOP, fault=True))
        self.assertFalse(tx.fill(out, 2*HOP))
        self.assertEqual(tx.reason, "stream fault")
        self.assertFalse(out.any())
        for args in (dict(gain=1), dict(session=121), dict(window=179), dict(gain=float("nan"))):
            with self.assertRaises(ValueError):
                Transmitter(**args)
        self.assertEqual(raw_settings(False)._streaminfo.streamOption, 1)

    def test_ring_overrun_preserves_order_and_data_ownership(self):
        ring = CaptureRing(1, capacity=2)
        samples = np.zeros((HOP, 1), np.float32)
        for i in range(3):
            samples.fill(i)
            ring.put(samples, (i*HOP, True, False, 0., 0.))
        first = ring.pop()
        samples.fill(42)
        ring.put(samples, (3*HOP, False, False, 0., 0.))
        self.assertTrue((first.samples == 1).all())
        self.assertEqual([first.start, ring.pop().start, ring.pop().start], [HOP,2*HOP,3*HOP])
        self.assertEqual(ring.drops, 1)
        self.assertIsNone(ring.pop())

    def test_consumer_copy_does_not_drop_incoming_audio(self):
        reading, release = Event(), Event()

        class SlowCopy(np.ndarray):
            def copy(self):
                reading.set()
                if not release.wait(2):
                    raise TimeoutError("test reader did not release the slot")
                return np.asarray(self).copy()

        ring = CaptureRing(1, capacity=3)
        ring.data = ring.data.view(SlowCopy)
        block = np.ones((HOP, 1), np.float32)
        ring.put(block, (0, False, False, 0., 0.))
        result = []
        reader = Thread(target=lambda: result.append(ring.pop()))
        reader.start()
        try:
            self.assertTrue(reading.wait(1))
            ring.put(2*block, (HOP, False, False, 0., 0.))
            ring.put(3*block, (2*HOP, False, False, 0., 0.))
        finally:
            release.set()
            reader.join()
        self.assertEqual(ring.drops, 0)
        self.assertTrue((result[0].samples == 1).all())
        self.assertTrue((ring.pop().samples == 2).all())
        self.assertTrue((ring.pop().samples == 3).all())


class OutputTests(unittest.TestCase):
    def test_missing_camera_library_falls_back(self):
        with patch.dict("sys.modules", pyvirtualcam=None), CameraOutput() as output:
            output.thread.join(timeout=2)
        self.assertIn("Preview only", output.status)
        self.assertIsNotNone(output.error)

    def test_fallback_still_displays_valid_frames(self):
        def fail(**kwargs):
            raise RuntimeError("fake camera busy")
        stub = SimpleNamespace(Camera=fail, PixelFormat=SimpleNamespace(BGR=0))
        opts = parser().parse_args(["--simulate", "--duration", "0.35"])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", pyvirtualcam=stub))
            for fn in ("namedWindow", "resizeWindow", "destroyAllWindows"):
                stack.enter_context(patch("main.cv2."+fn))
            stack.enter_context(patch("main.cv2.waitKey", return_value=-1))
            stack.enter_context(patch("main.cv2.getWindowProperty",
                                      side_effect=[1, 1, cv2.error("window closed")]))
            seen = []
            stack.enter_context(patch("main.cv2.imshow", side_effect=lambda name, f: seen.append(f.shape)))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            run(opts)
        summary = json.loads(output.getvalue())
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(shape == (720, 1280, 3) for shape in seen))
        self.assertIn("busy", summary["camera_error"])
        self.assertEqual(summary["dtype"], "uint8")
        self.assertEqual(summary["output_seconds"], 0)

    def test_send_failure_closes_camera(self):
        class BrokenCamera:
            device = "test receiver"
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.closed = True

            def send(self, frame):
                raise OSError("receiver disconnected")

        backend = BrokenCamera()
        stub = SimpleNamespace(Camera=lambda **kw: backend, PixelFormat=SimpleNamespace(BGR=0))
        with patch.dict("sys.modules", pyvirtualcam=stub), CameraOutput() as output:
            output.latest = np.empty((720,1280,3), np.uint8)
            until = monotonic()+2
            while output.error is None and monotonic()<until:
                sleep(0.01)
        self.assertTrue(backend.closed)
        self.assertIn("disconnected", output.error)
        self.assertFalse(output.thread.is_alive())

    def test_wav_input_and_eof_teardown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trial.wav"
            data = np.full(HOP*25, -16384, np.int16)
            wavfile.write(path, RATE, data)
            np.testing.assert_array_equal(wav_input(path), np.full((len(data), 1), -0.5))
            opts = parser().parse_args(["--replay", str(path), "--headless", "--no-virtual-cam"])
            with contextlib.redirect_stdout(io.StringIO()) as output:
                run(opts)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["blocks"], 25)
            self.assertIsNone(summary["stream_closed"])
            wavfile.write(path, 44100, data)
            with self.assertRaisesRegex(ValueError, "48000"):
                wav_input(path)

    def test_frame_layout_and_updated_pixels(self):
        dsp = DSPProcessor()
        health = dict(source="Synthetic",camera="Preview only",peak=0.,used=0.,budget=120.,
                      remaining=120.,reason="muted",xruns=0,drops=0)
        first = render(dsp.snapshot(), health)
        snap = dsp.snapshot()
        snap["radar"][40, 30] = 1
        snap["target"] = (0,40)
        snap["bpm"] = 15
        snap["waveform"] = np.sin(np.arange(1200)/20)
        second = render(snap, health)
        self.assertEqual(second.shape, (720,1280,3))
        self.assertEqual(second.dtype, np.uint8)
        self.assertTrue(second.flags.c_contiguous)
        self.assertGreater(np.count_nonzero(first != second), 1000)
