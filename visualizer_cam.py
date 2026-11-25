"""OpenCV dashboard and an independent latest-frame virtual camera producer."""

from threading import Event, Thread

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 1280, 720, 30
BG = (22, 21, 18)
PANEL = (32, 31, 27)
LINE = (67, 65, 56)
TEXT = (235, 238, 236)
DIM = (167, 172, 162)
AMBER = (87, 192, 252)
MINT = (169, 216, 118)


def label(frame, text, x, y, size=0.5, color=TEXT, thickness=1):
    cv2.putText(frame, str(text), (x, y), cv2.FONT_HERSHEY_DUPLEX, size,
                color, thickness, cv2.LINE_AA)


def trace(frame, values, left, top, width, height, low, high, color):
    if len(values) < 2:
        return
    indices = np.linspace(0, len(values)-1, min(len(values), width)).astype(int)
    x = np.linspace(left, left+width-1, len(indices))
    y = top + (1-np.clip((values[indices]-low)/(high-low), 0, 1))*(height-1)
    points = np.column_stack((x, y)).astype(np.int32)
    cv2.polylines(frame, [points], False, color, 1, cv2.LINE_AA)


def panel(frame, x, y, w, h, title):
    cv2.rectangle(frame, (x, y), (x+w, y+h), PANEL, -1)
    label(frame, title, x+16, y+26, 0.53)


def render(snapshot, health):
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = BG
    label(frame, "Bat Cam", 27, 49, 1.24, TEXT, 2)
    label(frame, health["source"], 235, 43, 0.55, MINT)
    label(frame, f'{snapshot["time"]:07.2f} s', 645, 43, 0.65)
    label(frame, health["camera"][:39], 845, 43, 0.48, DIM)
    cv2.line(frame, (26, 66), (1254, 66), LINE)

    panel(frame, 26, 82, 600, 412, "Motion radar")
    lateral = snapshot["lateral"]
    label(frame, "calibrated L/R" if lateral else "lateral uncalibrated", 422, 108, 0.43, DIM)
    x0, y0, w, h = 94, 128, 511, 310
    image = cv2.applyColorMap(np.uint8(np.clip(snapshot["radar"], 0, 1)*255),
                              cv2.COLORMAP_INFERNO)
    frame[y0:y0+h, x0:x0+w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    for j, speed in enumerate(("+4", "+2", "0", "-2", "-4")):
        y = y0 + round((h-1)*j/4)
        cv2.line(frame, (x0, y), (x0+w-1, y), LINE)
        label(frame, speed, 48, y+5, 0.46, DIM)
    for j, direction in enumerate(("Left", "Center", "Right")):
        x = x0+round((w-1)*j/2)
        cv2.line(frame, (x, y0), (x, y0+h-1), LINE)
        label(frame, direction, x-(30 if j else 0), y0+h+20, 0.43, DIM)
    if snapshot["target"] is not None:
        pos, row = snapshot["target"]
        center = (round(x0+(pos+1)*(w-1)/2), round(y0+row*(h-1)/80))
        cv2.circle(frame, center, 11, TEXT, 1, cv2.LINE_AA)
        cv2.drawMarker(frame, center, TEXT, cv2.MARKER_CROSS, 32, 1)
    label(frame, "+ toward / - away     approx. radial m/s", x0, 482, 0.42, DIM)

    panel(frame, 642, 82, 612, 255, "Doppler spectrum")
    label(frame, "0-50 dB over floor  /  4 s waterfall", 944, 108, 0.42, DIM)
    for i, (name, color) in enumerate((("L", AMBER), ("R", MINT))):
        x = 661+i*295
        freq = snapshot["carriers"][i] / 1000
        label(frame, f"{name}   {freq:.2f} kHz", x, 134, 0.45, color)
        shift = snapshot["shifts"][i]
        if np.isfinite(shift):
            label(frame, f"{shift:+.0f} Hz", x+190, 134, 0.44, color)
        cv2.line(frame, (x+136, 140), (x+136, 294), LINE)
        trace(frame, snapshot["spectrum"][i], x, 139, 273, 62, 0, 50, color)
        fall = np.uint8(np.clip(snapshot["waterfall"][:, i, :]/50, 0, 1)*255)
        strip = cv2.applyColorMap(fall, cv2.COLORMAP_VIRIDIS)
        frame[207:294, x:x+273] = cv2.resize(strip, (273, 87))
        label(frame, "-400", x, 314, 0.44, DIM)
        label(frame, "0", x+132, 314, 0.44, DIM)
        label(frame, "+400 Hz", x+210, 314, 0.44, DIM)

    panel(frame, 642, 352, 612, 195, "Breathing")
    bpm = snapshot["bpm"]
    label(frame, f"{bpm:.1f}" if bpm is not None else "--", 658, 445, 1.52, MINT, 2)
    label(frame, "breaths/min", 660, 474, 0.48, DIM)
    breath_state = snapshot["breath_state"]
    if breath_state == "acquiring":
        breath_state += f' {snapshot["clean_seconds"]:.0f}s'
    label(frame, breath_state, 660, 519, 0.47, MINT)
    wave = snapshot["waveform"]
    x, y = 846, 402
    cv2.line(frame, (x, y+45), (1232, y+45), LINE)
    if len(wave):
        span = max(0.001, float(np.percentile(abs(wave), 97)))
        occupied = max(2, round(386*len(wave)/1200))
        trace(frame, wave, x+386-occupied, y, occupied, 90, -span, span, MINT)
    label(frame, f'60 s window    quality {snapshot["quality"]:.0%}', x, 519, 0.44, DIM)

    panel(frame, 26, 508, 600, 133, "Gesture")
    events = snapshot["events"]
    latest = events[-1] if events else None
    if latest and snapshot["time"] - latest.time <= 1.5 and snapshot["state"] == "sensing":
        label(frame, latest.name, 42, 580, 1.02, AMBER, 2)
        label(frame, f"{latest.confidence:.0%}", 42, 615, 0.52, AMBER)
    else:
        label(frame, "Listening" if snapshot["state"] == "sensing" else "Idle", 42, 581, 0.82, DIM)
    for i, g in enumerate(reversed(events[-4:])):
        label(frame, f"{g.time:6.1f}s   {g.name}   {g.confidence:.0%}", 330, 542+23*i, 0.42)

    panel(frame, 642, 560, 612, 81, "Digital output")
    peak = health["peak"]
    db = 20*np.log10(max(peak, 1e-8))
    label(frame, f'{db:.1f} dBFS' if peak else "Muted", 660, 623, 0.53, AMBER)
    label(frame, f'{health["used"]:.1f} s on / {health["remaining"]:.1f} s left',
          844, 585, 0.46)
    cv2.rectangle(frame, (846, 602), (1229, 615), LINE, -1)
    fraction = np.clip(health["used"]/health["budget"], 0, 1)
    if fraction:
        cv2.rectangle(frame, (846, 602), (846+round(383*fraction), 615), AMBER, -1)

    status = snapshot["state"]
    if "calibration" in status:
        status += f' {snapshot["progress"]:.0%}'
    if snapshot["zone"]:
        status += " | sampling " + snapshot["zone"]
    elif snapshot["zones"]:
        status += " | marks " + "/".join(snapshot["zones"])
    label(frame, f'{status}  |  {health["reason"]}', 26, 672, 0.51, MINT)
    label(frame, f'SNR L/R {snapshot["snr"][0]:.0f}/{snapshot["snr"][1]:.0f} dB'
          f'    xruns {health["xruns"]}    drops {health["drops"]}', 796, 672, 0.44, DIM)
    label(frame, "M  arm / mute      R  resume      C  calibrate      1 / 2 / 3  Doppler zones      V  view",
          26, 703, 0.44, DIM)
    return frame


def render_sonar(snapshot, health, hud=True):
    grid = snapshot.get("depth")
    if grid is None:
        grid = np.zeros((113, 64), np.float32)
    # At the bottom of the image, returns are closest to the laptop.
    soft = cv2.bilateralFilter(np.ascontiguousarray(grid[::-1]), 5, .12, 2)
    gray = np.uint8(np.clip(soft*255, 0, 255))
    gray = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    if not hud:
        return frame

    label(frame, "BAT CAM   /   RANGE VIEW", 28, 40, .65)
    status = snapshot["state"]
    if snapshot.get("depth_locked"):
        status = ("L/R calibrated" if snapshot.get("range_calibrated")
                  else "L/R estimate   C to calibrate")
    elif status == "sensing":
        status = "No range reference   V for Doppler"
    label(frame, status, 30, 64, .45, DIM)
    span = snapshot.get("depth_span")
    if span and snapshot.get("depth_locked"):
        label(frame, f"{span[0]:.2f}-{span[1]:.2f} m approx.", 30, 88, .45, DIM)
    if health.get("simulated"):
        label(frame, "SYNTHETIC", 30, 111, .43, AMBER)
    guide = snapshot.get("guide")
    if guide:
        label(frame, guide, 30, 135 if health.get("simulated") else 111, .49, AMBER)

    t = snapshot["time"]
    bpm = snapshot.get("range_bpm")
    wave = snapshot.get("range_wave", ())
    if bpm is not None:
        pulse = 4 + round(2*(1+np.sin(2*np.pi*t*bpm/60)))
        cv2.circle(frame, (1089, 36), pulse, MINT, -1, cv2.LINE_AA)
        label(frame, f"{bpm:.1f} BPM", 1106, 42, .67, MINT)
        label(frame, f'Confidence {snapshot["range_quality"]:.0%}', 1090, 65, .46, DIM)
    else:
        elapsed = snapshot.get("range_seconds", 0)
        label(frame, f"Breath lock {elapsed:.0f}s" if elapsed else "Waiting for torso echo",
              1040, 40, .52, DIM)
    if len(wave) > 1:
        span = max(.01, float(np.max(abs(wave))))
        trace(frame, wave, 1089, 77, 158, 27, -span, span, MINT)

    hot = np.uint8(gray > 62)
    contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patches = [c for c in contours if cv2.contourArea(c) > 160]
    cv2.drawContours(frame, patches, -1, (160, 146, 224), 1, cv2.LINE_AA)
    if patches and snapshot.get("range_calibrated"):
        x, y, w, h = cv2.boundingRect(max(patches, key=cv2.contourArea))
        cv2.rectangle(frame, (x, y), (x+w-1, y+h-1), (214, 209, 211), 1)
    events = snapshot["events"]
    if events and t-events[-1].time < 1.5 and snapshot["state"] == "sensing":
        tag = events[-1].name.upper().replace("LEFT", "L").replace("RIGHT", "R")
        label(frame, tag, 30, HEIGHT-55, .9, AMBER, 2)
    label(frame, f'{health["reason"]}   {health["remaining"]:.0f}s available',
          30, HEIGHT-29, .48, DIM)
    label(frame, "C  calibrate     M  mute     R  resume     H  hide     V  view",
          850, HEIGHT-29, .47, DIM)
    return frame


class CameraOutput:
    """The UI remains responsive during backend open, pacing and send failures."""

    def __init__(self, enabled=True):
        self.status = "Opening virtual camera" if enabled else "Preview only"
        self.error = None
        self.latest = None
        self.stop = Event()
        self.frames = 0
        self.thread = Thread(target=self._run, name="webcam", daemon=True) if enabled else None

    def __enter__(self):
        if self.thread:
            self.thread.start()
        return self

    def _run(self):
        try:
            import pyvirtualcam as pvc
            failures = []
            for backend in ("obs", "unitycapture"):
                try:
                    cam = pvc.Camera(width=WIDTH, height=HEIGHT, fps=FPS,
                                     fmt=pvc.PixelFormat.BGR, backend=backend)
                    break
                except RuntimeError as exc:
                    failures.append(str(exc))
            else:
                raise RuntimeError(" / ".join(failures))
            with cam:
                self.status = cam.device
                while not self.stop.is_set():
                    frame = self.latest
                    if frame is None:
                        self.stop.wait(0.005)
                        continue
                    cam.send(frame)
                    self.frames += 1
                    cam.sleep_until_next_frame()
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            self.error = str(exc)
            self.status = "Preview only - camera unavailable"

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread:
            self.thread.join()
