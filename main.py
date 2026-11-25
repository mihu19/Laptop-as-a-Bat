"""Run the live sonar dashboard, a muted hardware check, or offline replay."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from queue import Empty, SimpleQueue
from threading import Event, Thread
from time import monotonic, sleep

import cv2
import numpy as np
from scipy.io import wavfile
import sounddevice as sd

from audio_engine import AudioEngine, Block, CARRIERS, DEFAULT_GAIN, HOP, MAX_GAIN, RATE
from dsp_processor import DSPProcessor
from sonar_imaging import RangeCamera, demo_echo
from visualizer_cam import CameraOutput, FPS, HEIGHT, WIDTH, render, render_sonar


def wav_input(path):
    rate, samples = wavfile.read(path)
    if samples.ndim == 1:
        samples = samples[:, None]
    if rate != RATE or samples.ndim != 2 or samples.shape[1] not in (1, 2):
        raise ValueError("Replay requires a mono or stereo 48000 Hz WAV capture.")
    if len(samples) < HOP:
        raise ValueError("Replay is shorter than one audio block.")
    dtype = samples.dtype
    if dtype == np.uint8:
        samples = (samples.astype(np.float32)-128)/128
    elif np.issubdtype(dtype, np.signedinteger):
        samples = samples.astype(np.float32) / float(-int(np.iinfo(dtype).min))
    elif not np.isfinite(samples).all():
        raise ValueError("Replay contains invalid floating point samples.")
    return samples.astype(np.float32, copy=False)


def demo_block(start, calibration, quiet, carriers):
    """Simulate a stereo capture. This source has no playback device."""
    t = (start + np.arange(HOP))/RATE
    result = np.random.default_rng(start).normal(0, 0.00008, (HOP, 2)).astype(np.float32)
    if start / RATE < calibration:
        return result
    since = t - calibration - quiet
    for i, fc in enumerate(carriers):
        phase = 2*np.pi*fc*t
        breath = 0.42*np.sin(2*np.pi*0.25*since)
        echo = 0.045*np.cos(phase) + 0.008*np.sin(phase + np.where(since > 15, breath, 0))
        for onset, duration, left, right, offset in (
                (0.15, 1.60, -0.85, -0.85, 145), (2.35, 1.60, 0, 0, 145),
                (4.55, 1.60, 0.85, 0.85, 145), (7.0, 0.30, 0, 0, 170),
                (8.1, 0.30, 0, 0, -165), (9.2, 0.50, 0.95, -0.95, 155),
                (10.5, 0.50, -0.95, 0.95, -150),
                (12.0, 0.08, 0, 0, 180), (12.08, 0.08, 0, 0, -180),
                (12.36, 0.08, 0, 0, 180), (12.44, 0.08, 0, 0, -180)):
            if since[-1] < onset or since[0] >= onset + duration:
                continue
            u = (since-onset)/duration
            x = left + (right-left)*u
            amp = 0.014*(1 + (1 if i else -1)*x)
            edge = np.clip(np.minimum(u, 1-u)*12, 0, 1)
            echo += amp*edge*np.sin(phase + 2*np.pi*offset*t)
        result += echo[:, None]*np.array([1.0, 0.9])
    return result


class Worker:
    def __init__(self, dsp, engine=None, recording=None, range_cam=None):
        self.dsp = dsp
        self.engine = engine
        self.recording = recording
        self.range_cam = range_cam
        self.range_start = None
        self.expected = None
        self.stop = Event()
        self.ended = Event()
        self.commands = SimpleQueue()
        self.latest = self.snapshot()
        self.error = None
        self.blocks = 0
        self.thread = Thread(target=self._run, name="sonar-dsp")

    def __enter__(self):
        self.thread.start()
        return self

    def snapshot(self):
        snap = self.dsp.snapshot()
        cam = self.range_cam
        if cam is not None:
            snap.update(depth=cam.grid.copy(), depth_locked=cam.locked,
                        depth_span=(float(cam.ranges[0]), float(cam.ranges[-1])),
                        range_calibrated=cam.calibrated, guide=cam.calibration_status,
                        range_bpm=cam.breath.bpm, range_quality=cam.breath.quality,
                        range_wave=cam.breath.wave.copy(), range_seconds=cam.breath.count/25)
        return snap

    def _range(self, block):
        cam = self.range_cam
        if cam is None:
            return
        if not block.active or block.gap or block.start != self.expected:
            if self.range_start is not None:
                cam.reset()
            self.range_start = None
        self.expected = block.start + HOP
        if not block.active or block.gap:
            return
        if self.range_start is None:
            self.range_start = block.start
        elapsed = (block.start-self.range_start)/RATE
        if elapsed >= .12:
            cam.process(block.samples, learn=elapsed < 1.4)
            cam.guide_step(block.start/RATE)

    def _run(self):
        try:
            epoch = monotonic()
            while not self.stop.is_set():
                try:
                    command = self.commands.get_nowait()
                    if command == "guided" and self.range_cam is not None:
                        self.range_cam.calibrate(self.dsp.expected/RATE)
                    elif command == "cw center":
                        self.dsp.mark_zone("center")
                    else:
                        self.dsp.mark_zone(command)
                except Empty:
                    pass
                if self.engine is not None:
                    block = self.engine.ring.pop()
                    if block is None:
                        self.stop.wait(0.002)
                        continue
                else:
                    start = self.blocks * HOP
                    t = start / RATE
                    if self.stop.wait(max(0, epoch + t - monotonic())):
                        break
                    if self.recording is not None:
                        data = self.recording[start:start+HOP]
                        if len(data) < HOP:
                            break
                    else:
                        data = demo_block(start, self.dsp.calibration, self.dsp.quiet, self.dsp.carriers)
                        if self.range_cam is not None and t >= self.dsp.calibration:
                            data += demo_echo(self.range_cam.probe, start, HOP, t-self.dsp.calibration)
                        for i, zone in enumerate(("left", "center", "right")):
                            mark = self.dsp.calibration + self.dsp.quiet + i*2.2
                            if self.blocks == round(mark * RATE / HOP):
                                self.dsp.mark_zone(zone)
                    block = Block(data, start, t >= self.dsp.calibration, False, t, t)
                self._range(block)
                self.dsp.process(block.samples, block.active, block.start, block.gap,
                                 self.range_cam.position if self.range_cam is not None else None)
                self.blocks += 1
                if self.blocks % 3 == 0:
                    self.latest = self.snapshot()
            self.latest = self.snapshot()
        except Exception as exc:
            self.error = exc
        finally:
            self.ended.set()

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def run(args):
    if args.list_devices:
        print(sd.query_devices())
        return
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("Duration must be a finite positive number.")
    if not (0 < args.max_volume <= MAX_GAIN):
        raise ValueError(f"Max volume must be positive and at most {MAX_GAIN:.8f}.")
    if not 0 <= args.direct_path_m <= 1:
        raise ValueError("Direct-path length must be between 0 and 1 metre.")
    recording = wav_input(args.replay) if args.replay else None
    carriers = (args.left_freq, args.right_freq)
    if recording is None and not args.simulate:
        engine = AudioEngine(args.input_device, args.output_device, carriers, args.max_volume,
                             args.exclusive, args.session_seconds, args.duty_window,
                             args.max_seconds, args.auto_cooldown, not args.doppler_only)
        source = "Live / WASAPI RAW requested"
    else:
        engine = None
        source = "WAV replay / TX off" if args.replay else "Synthetic demo / TX off"
    frame_count = 0
    with ExitStack() as stack:
        if engine is not None:
            stack.enter_context(engine)
            channels = engine.channels
            print(f"WASAPI input {engine.devices[0]}, output {engine.devices[1]}, "
                  f"{channels} in / 2 out, 48000 Hz, RAW requested", flush=True)
        else:
            channels = recording.shape[1] if recording is not None else 2
        dsp = DSPProcessor(channels, carriers, args.calibration_duration)
        route = None
        if engine is not None:
            route = [str(sd.query_devices(i)["name"]) for i in engine.devices]
            route.extend([channels, *carriers, args.exclusive])
        has_probe = not args.doppler_only and (recording is None or args.ranging_replay)
        range_cam = (RangeCamera(channels, carriers, route, args.calibration_file if engine else None,
                                 args.direct_path_m)
                     if has_probe else None)
        worker = stack.enter_context(Worker(dsp, engine, recording, range_cam))
        cam = stack.enter_context(CameraOutput(not args.no_virtual_cam))
        if not args.headless:
            cv2.namedWindow("Bat Cam", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Bat Cam", WIDTH, HEIGHT)
            stack.callback(cv2.destroyAllWindows)
        if engine is not None:
            stack.callback(engine.tx.mute)
        begin = deadline = monotonic()
        view = args.view or ("debug" if range_cam is None else "hud")
        last_audio = begin
        previous_blocks = 0
        while True:
            now = monotonic()
            if worker.error is not None:
                raise RuntimeError("DSP worker failed") from worker.error
            if engine is not None:
                engine.tx.heartbeat = now
                if engine.blocks != previous_blocks:
                    previous_blocks, last_audio = engine.blocks, now
                if not engine.stream.active or now-last_audio > 2:
                    raise RuntimeError("Audio stream stopped delivering callbacks.")
            snap = worker.latest
            tx = engine.tx if engine else None
            health = dict(source=source, camera=cam.status, peak=tx.peak if tx else 0,
                          used=tx.used/RATE if tx else 0, budget=tx.session/RATE if tx else 120,
                          remaining=(tx.session-tx.used)/RATE if tx else 120,
                          reason=tx.reason if tx else "playback disabled",
                          xruns=engine.xruns if engine else 0,
                          drops=engine.ring.drops if engine else 0,
                          simulated=args.simulate)
            if view == "debug":
                frame = render(snap, health)
            else:
                frame = render_sonar(snap, health, hud=view=="hud")
            cam.latest = frame
            frame_count += 1
            if not args.headless:
                cv2.imshow("Bat Cam", frame)
                key = cv2.waitKey(1) & 255
                try:
                    visible = cv2.getWindowProperty("Bat Cam", cv2.WND_PROP_VISIBLE) >= 1
                except cv2.error:
                    # Win32 may already have destroyed the native window on X.
                    visible = False
                if key in (27, ord("q"), ord("Q")) or not visible:
                    if tx:
                        tx.mute()
                    break
                key = key | 32
                if tx and key in (32, ord("m")):
                    if tx.armed or tx.resume_at is not None:
                        tx.mute("muted")
                    elif snap["ready"]:
                        tx.arm()
                if tx and key == ord("r") and snap["ready"]:
                    tx.arm(reset_demo=True)
                if key == ord("v"):
                    view = {"pure": "hud", "hud": "debug", "debug": "pure"}[view]
                if key == ord("h"):
                    view = "hud" if view == "pure" else "pure"
                if key == ord("c"):
                    worker.commands.put("guided")
                if key == ord("1"):
                    worker.commands.put("left")
                if key == ord("2"):
                    worker.commands.put("cw center")
                if key == ord("3"):
                    worker.commands.put("right")
            if worker.ended.is_set() or (args.duration and now - begin >= args.duration):
                break
            deadline += 1 / FPS
            if now - deadline > 1 / FPS:
                deadline = now
            sleep(max(0, deadline - monotonic()))
        if tx:
            tx.mute()
        if worker.error is not None:
            raise RuntimeError("DSP worker failed") from worker.error
        elapsed = monotonic() - begin
    if not worker.blocks or frame_count == 0:
        raise RuntimeError("No audio blocks or frames received.")
    print(json.dumps(dict(source=source, frames=frame_count, fps=round(frame_count/elapsed, 2),
                          shape=list(frame.shape), dtype=str(frame.dtype), blocks=worker.blocks,
                          stream_closed=engine.stream is None if engine else None,
                          output_seconds=engine.tx.total_emitted/RATE if engine else 0,
                          xruns=health["xruns"], drops=health["drops"], camera=cam.status,
                          camera_error=cam.error, sent=cam.frames, state=worker.latest["state"],
                          gestures=[e.name for e in worker.latest["events"]],
                          bpm=worker.latest.get("range_bpm", worker.latest["bpm"]),
                          range_locked=worker.latest.get("depth_locked", False),
                          range_peak=float(worker.latest["depth"].max()) if range_cam else None)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--input-device", type=int, help="WASAPI microphone ID")
    p.add_argument("--output-device", type=int, help="WASAPI stereo speaker ID")
    p.add_argument("--left-freq", type=float, default=CARRIERS[0])
    p.add_argument("--right-freq", type=float, default=CARRIERS[1])
    p.add_argument("--max-volume", type=float, default=DEFAULT_GAIN, help="linear peak amplitude")
    p.add_argument("--session-seconds", type=float, default=120,
                   help="maximum continuous and cumulative on-time in the duty window, up to 120 s")
    p.add_argument("--duty-window", type=float, default=180,
                   help="rolling exposure window in seconds, 180..3600")
    p.add_argument("--max-seconds", type=float, default=300, help="demo timer, independent of exposure budget")
    p.add_argument("--auto-cooldown", action="store_true", help="resume after at least 2 s when exposure permits")
    p.add_argument("--view", choices=("hud", "pure", "debug"))
    p.add_argument("--calibration-file", type=Path, default=Path("calibration.json"))
    p.add_argument("--ranging-replay", action="store_true", help="WAV contains the Phase 3 multitone probe")
    p.add_argument("--doppler-only", action="store_true", help="run the original CW-only transmitter and debug view")
    p.add_argument("--direct-path-m", type=float, default=0,
                   help="approximate speaker-to-mic path length for the range axis")
    p.add_argument("--calibration-duration", type=float, default=3, help="muted ambient seconds")
    p.add_argument("--exclusive", action="store_true", help="request exclusive WASAPI with RAW")
    p.add_argument("--no-virtual-cam", "--preview-only", action="store_true")
    p.add_argument("--headless", action="store_true", help="render without a window; live TX stays muted")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--simulate", action="store_true", help="generate a repeatable virtual scene")
    group.add_argument("--replay", type=Path, metavar="CAPTURE.wav")
    return p


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, RuntimeError, sd.PortAudioError, cv2.error) as error:
        sys.exit(f"Bat Cam: {error}")
