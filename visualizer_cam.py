"""OpenCV dashboard and an independent latest-frame virtual camera producer."""

from threading import Event, Thread
from functools import lru_cache

import cv2
import numpy as np

from ofdm_ranging import ROOM_SECONDS

WIDTH, HEIGHT, FPS = 1280, 720, 30
ECHO_HISTORY_FRAMES = 100
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
          768, HEIGHT-29, .47, DIM)
    return frame


def paint_echoes(area, levels):
    gray = cv2.resize(np.uint8(255*levels), (area.shape[1], area.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    np.copyto(area, cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), where=(gray > 0)[:, :, None])


def plot_echo_history(area, history):
    width = max(1, round(area.shape[1]*len(history)/ECHO_HISTORY_FRAMES))
    paint_echoes(area[:, -width:], history.T[::-1])


def render_echo_map(snapshot, health):
    """Render the live radar measurements shared by the preview and virtual camera."""
    frame = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
    profile = snapshot["echo_profile"]
    ranges = snapshot["echo_ranges"]
    peak = int(profile.argmax())
    active = snapshot["state"] == "sensing"
    locked = active and snapshot["depth_locked"]
    has_echo = locked and profile[peak] > 0
    room_ready = snapshot.get("room_ready")
    tracked_range = snapshot.get("tracked_range")
    spatial = snapshot.get("spatial")
    zones = snapshot.get("zones")

    label(frame, "Bat Cam", 32, 58, 1.26, TEXT, 2)
    if not active:
        status = ("Clear the area, then press M to calibrate"
                  if room_ready is not None or spatial is not None else "Probe muted")
    elif spatial is not None:
        status = spatial["status"]
        if status == "Move an object into range" and has_echo:
            status = "Range echoes detected   /   waiting for matching paths"
    elif not locked:
        status = "Waiting for probe timing"
    elif room_ready is False:
        status = f"Measuring the empty room   /   stand clear for {ROOM_SECONDS:.0f}s"
    elif tracked_range is not None:
        status = "Tracking an echo"
    elif not has_echo:
        status = "Waiting for an echo   /   move into range" if room_ready else "No echo in range"
    else:
        status = "Echo detected" if room_ready is None else "Checking echo stability"
    status_ready = has_echo if spatial is None else spatial["footprint"].any()
    if zones and (zones["learning"] or not status_ready):
        status = zones["status"]
        status_ready = zones["saved"] if zones["learning"] else zones["position"] in ("Left", "Center", "Right")
    if health.get("auto_arm") and spatial is not None:
        status = "Keep the area clear   /   starting in a moment"
    label(frame, status, 35, 91, .60, MINT if status_ready else DIM)
    if health.get("simulated"):
        source = "SIMULATION"
    elif health.get("replay"):
        source = "WAV REPLAY"
    else:
        source = "LIVE MICROPHONE"
    source_size = cv2.getTextSize(source, cv2.FONT_HERSHEY_DUPLEX, .56, 1)[0][0]
    label(frame, source, WIDTH-34-source_size, 59, .56, AMBER if source != "LIVE MICROPHONE" else DIM)

    panel(frame, 32, 116, 807, 536, "Estimated echo footprint" if spatial is not None else "Echo trail")
    left, right, top, bottom = 116, 812, 192, 554
    if spatial is not None:
        left, right = 202, 726
    label(frame, "Approx. distance (cm)", 52, 173, .47, DIM)
    graph = frame[top:bottom+1, left:right+1]
    graph[:] = BG
    vertical = ranges if spatial is None else spatial["distance"]
    ticks = np.linspace(vertical[0], vertical[-1], 5)
    for i, distance in enumerate(ticks):
        y = bottom-round(i*(bottom-top)/(len(ticks)-1))
        cv2.line(frame, (left, y), (right, y), LINE, 1)
        label(frame, f"{distance*100:.0f}", left-43, y+5, .49, DIM)
    for x in (left, (left+right)//2, right):
        cv2.line(frame, (x, top), (x, bottom), LINE, 1)
    history = snapshot.get("echo_history")
    if active and spatial is not None:
        paint_echoes(graph, spatial["footprint"])
    elif spatial is None and locked and history is not None and len(history):
        plot_echo_history(graph, history)
    if spatial is not None:
        label(frame, f'{spatial["x"][0]*100:.0f} cm', left-24, 580, .45, DIM)
        label(frame, "0", (left+right)//2-5, 580, .45, DIM)
        label(frame, f'{spatial["x"][-1]*100:.0f} cm', right-23, 580, .45, DIM)
        label(frame, "Top view near laptop height", 52, 628, .47, DIM)
        if zones and not zones["learning"]:
            label(frame, "Learned zone", 52, 204, .47, DIM)
            position = zones["position"]
            label(frame, position or "--", 52, 239, .70, MINT if position else DIM)
            if position:
                label(frame, zones["detail"], 52, 263, .37, DIM)
    else:
        label(frame, "3 s ago", left, 580, .51, DIM)
        label(frame, "Now", right-35, 580, .51, DIM)
    legend = cv2.applyColorMap(np.uint8(np.linspace(30, 245, 106))[None, :], cv2.COLORMAP_TURBO)
    frame[616:629, 625:731] = legend
    label(frame, "Weak", 573, 628, .45, DIM)
    label(frame, "Strong", 744, 628, .45, DIM)

    panel(frame, 858, 116, 390, 536, "Echo profile")
    label(frame, "Relative strength by distance", 874, 172, .47, DIM)
    chart_x, chart_y, chart_w, chart_h = 879, 195, 347, 105
    cv2.line(frame, (chart_x, chart_y+chart_h-1), (chart_x+chart_w-1, chart_y+chart_h-1), LINE)
    if has_echo:
        trace(frame, np.sqrt(profile/profile[peak]), chart_x, chart_y, chart_w, chart_h, 0, 1, MINT)
    label(frame, f"{ranges[0]*100:.0f} cm", chart_x, 323, .47, DIM)
    label(frame, f"{ranges[-1]*100:.0f} cm", chart_x+chart_w-56, 323, .47, DIM)

    label(frame, "Strongest echo", 874, 369, .48, DIM)
    label(frame, f"~{ranges[peak]*100:.0f} cm" if has_echo else "--", 874, 405, .82)
    label(frame, "Echo SNR", 1072, 369, .48, DIM)
    noise = snapshot["echo_noise_power"]
    snr = 10*np.log10(profile[peak]/noise) if has_echo and noise > 0 else None
    label(frame, f"{snr:.1f} dB" if snr is not None else "--", 1072, 405, .76)

    events = snapshot["events"]
    if spatial is not None:
        label(frame, "Range history   /   last 3 s", 874, 462, .48, DIM)
        history_plot = frame[478:610, chart_x:chart_x+chart_w]
        history_plot[:] = BG
        if locked and history is not None and len(history):
            plot_echo_history(history_plot, history)
        label(frame, f'{spatial["paths"]}/4 timing paths   /   {spatial["echo_paths"]}/4 echo paths',
              874, 633, .44, DIM)
    elif room_ready is None:
        gesture = events[-1].name if has_echo and events and snapshot["time"]-events[-1].time < 1.5 else "--"
        label(frame, "Recent gesture", 874, 462, .48, DIM)
        label(frame, gesture, 874, 498, .73, AMBER if gesture != "--" else TEXT)
        label(frame, "Breathing", 874, 552, .48, DIM)
        bpm = snapshot["range_bpm"] if has_echo else None
        label(frame, f"{bpm:.1f} BPM" if bpm is not None else "--", 874, 588, .73)
    else:
        label(frame, "Tracked range", 874, 462, .48, DIM)
        label(frame, f"~{tracked_range*100:.0f} cm" if tracked_range is not None else "--", 874, 498, .73)
    controls = "M  mute / arm     Esc  exit"
    if zones and zones["learning"]:
        controls = zones["detail"] if zones["finished"] else f'{zones["detail"]}     C  capture     M  mute'
    label(frame, controls, 33, 689, .48, DIM)
    if spatial is not None:
        label(frame, "Estimated laptop geometry", 1001, 689, .46, DIM)
    return frame


def render_nearfield(snapshot, health, view):
    grid = np.ascontiguousarray(snapshot["depth"][::-1], dtype=np.float32)
    filtered = cv2.bilateralFilter(grid, 5, .14, 2)
    if snapshot.get("smoothing"):
        filtered = cv2.GaussianBlur(filtered, (0, 0), snapshot["smoothing"])
    # Keep the mesh on measured support. Depth controls brightness and perspective.
    slope = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    brightness = np.clip(filtered + .12*slope, 0, 1)
    gray = np.uint8(brightness*255)
    rows, columns = gray.shape
    src = np.float32([(0, 0), (columns-1, 0), (columns-1, rows-1), (0, rows-1)])
    canvas = (WIDTH//2, HEIGHT//2)
    dst = np.float32([(119, 42), (520, 42), (canvas[0]-1, canvas[1]-1), (0, canvas[1]-1)])
    projection = cv2.getPerspectiveTransform(src, dst)
    projected = cv2.warpPerspective(gray, projection, canvas, flags=cv2.INTER_CUBIC)
    frame = cv2.applyColorMap(projected, cv2.COLORMAP_INFERNO)
    frame[projected == 0] = 0
    if view == "hud":
        mask = np.uint8(projected > 70)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines = [c for c in contours if cv2.contourArea(c) > 55]
        cv2.drawContours(frame, outlines, -1, (115, 130, 204), 1, cv2.LINE_AA)
        lines = ((np.arange(rows)[:, None] % 8 == 0) | (np.arange(columns) % 7 == 0))
        mesh = np.uint8(lines & (gray > 81))
        mesh = cv2.warpPerspective(mesh, projection, canvas, flags=cv2.INTER_NEAREST)
        selected = mesh > 0
        frame[selected] = np.minimum(255, frame[selected].astype(np.uint16)+(19, 25, 34))
        if snapshot["position"] and np.max(gray) > 110:
            lateral, depth = snapshot["position"]
            point = np.float32([[[(lateral+1)*(columns-1)/2, (.95-depth)*(rows-1)/.80]]])
            x, y = np.rint(cv2.perspectiveTransform(point, projection)[0, 0]).astype(int)
            if 0 <= x < canvas[0] and 0 <= y < canvas[1] and projected[y, x] > 50:
                cv2.circle(frame, (x, y), 6, TEXT, 1, cv2.LINE_AA)
    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    if view == "pure":
        return frame
    if view == "debug":
        label(frame, "NEAR FIELD   /   GATED RETURN HISTORY", 28, 42, .64)
        history = snapshot["gated_waterfall"]
        if history.size:
            for side in range(2):
                x = 45+side*635
                data = np.uint8(np.clip(history[:, side, :]*255/32, 0, 255))
                strip = cv2.applyColorMap(data, cv2.COLORMAP_INFERNO)
                frame[285:630, x:x+552] = cv2.resize(strip, (552, 345))
                trace(frame, snapshot["gated_spectrum"][side], x, 168, 552, 100, 0, 32, MINT)
                label(frame, "Left" if side == 0 else "Right", x, 147, .56)
                label(frame, "0.15 m                        0.95 m", x, 655, .48, DIM)
    else:
        label(frame, "BAT CAM   /   0.95 m", 28, 42, .64)
    status = ("Echo tracking" if snapshot["depth_locked"] else "Waiting for a range reference")
    if snapshot["state"] == "muted":
        status = "Muted"
    if health.get("simulated"):
        status += "   /   synthetic"
    label(frame, status, 29, 68, .42, DIM)
    bpm = snapshot["range_bpm"]
    if bpm is not None:
        label(frame, f'{bpm:.1f} BPM  {snapshot["range_quality"]:.0%}', 1040, 42, .62, MINT)
    elif snapshot["range_seconds"] > 0:
        label(frame, f'Breathing  {snapshot["range_seconds"]:.1f}s', 1040, 42, .46, DIM)
    wave = snapshot["range_wave"]
    if len(wave) > 1:
        scale = max(.01, float(np.percentile(abs(wave), 95)))
        trace(frame, wave, 1042, 53, 199, 23, -scale, scale, MINT)
    events = snapshot["events"]
    if events and snapshot["state"] == "sensing" and snapshot["time"]-events[-1].time < 1.4:
        label(frame, events[-1].name.upper(), 29, HEIGHT-58, .81, AMBER, 2)
    label(frame, f'{health["reason"]}   {health["remaining"]:.0f}s available',
          29, HEIGHT-28, .47, DIM)
    label(frame, "V  view     H  hide     M  mute     R  reset timer", 849, HEIGHT-28, .47, DIM)
    return frame


PORTRAIT_Y, PORTRAIT_X = np.mgrid[-1:1:180j, -1:1:320j].astype(np.float32)


def draw_portrait(frame, lateral, depth, chest_displacement=0.):
    scale = np.clip(.57/depth, .65, 1.35)
    x = (PORTRAIT_X-.42*lateral)/scale
    y = (PORTRAIT_Y-.04)/scale
    yaw = .16*lateral
    chest = np.clip(chest_displacement/.004, -.5, .5)

    def shape(cx, cy, width, height):
        return np.sqrt(np.maximum(1-((x-cx)/width)**2-((y-cy)/height)**2, 0))

    head = shape(.03*yaw, -.34, .27, .49)
    jaw = shape(yaw*.12, -.13, .205, .32)
    shoulders = shape(0, .62, .73*(1+.045*chest), .44)
    neck = shape(0, .19, .15, .31)
    surface = np.maximum.reduce((.70*head, .63*jaw, .49*shoulders, .44*neck))
    face = np.maximum(head, jaw)
    nose = np.exp(-((x-.37*yaw)/.055)**2-((y+.28)/.14)**2)
    brows = np.exp(-((abs(x-.19*yaw)-.105)/.085)**2-((y+.48)/.045)**2)
    cheeks = np.exp(-((abs(x-.19*yaw)-.145)/.093)**2-((y+.22)/.10)**2)
    mouth = np.exp(-((x-.19*yaw)/.13)**2-((y+.095)/.016)**2)
    surface += face*(.23*nose+.10*brows+.13*cheeks-.08*mouth)
    surface = surface.astype(np.float32, copy=False)
    dx = cv2.Sobel(surface, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(surface, cv2.CV_32F, 0, 1, ksize=3)
    light = np.clip(.78+.28*dx-.24*dy, .48, 1.12)
    intensity = np.uint8(255*np.clip(surface*light*(1.12-.48*depth), 0, 1))
    portrait = cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO)
    portrait[intensity == 0] = 0
    np.maximum(frame, cv2.resize(portrait, (WIDTH, HEIGHT)), out=frame)


def render_portrait(snapshot, health, view):
    """Shade a generic relief model located by acoustic range and lateral estimates."""
    if view == "debug":
        return render_nearfield(snapshot, health, view)
    frame = cv2.convertScaleAbs(render_nearfield(snapshot, health, "pure"), alpha=.28)
    subject = snapshot.get("subject")
    if subject is not None and snapshot["state"] == "sensing":
        draw_portrait(frame, *subject, snapshot.get("chest_displacement", 0.))
    events = snapshot["events"]
    age = snapshot["time"]-events[-1].time if events else 10.
    position = snapshot.get("position")
    if position and 0 <= age < .8 and snapshot["state"] == "sensing":
        x, depth = position
        center = (round(WIDTH*(.50+.37*x)), round(HEIGHT*(1-(depth-.15)/.8)))
        radius = min(90, round(20+60*age+12*abs(snapshot["velocity"])))
        cv2.circle(frame, center, radius, (96, 121, 235), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, center, (154, 187, 255), cv2.MARKER_CROSS, 17, 1)
    if view == "pure":
        return frame
    label(frame, "BAT CAM   /   ACOUSTIC MODEL", 29, 41, .60)
    status = f"Z {subject[1]*100:.0f} cm" if subject else "Waiting for a stable torso echo"
    label(frame, status if snapshot["state"] == "sensing" else "Muted", 29, 68, .45, DIM)
    if health.get("simulated"):
        label(frame, "SYNTHETIC", 29, 92, .43, AMBER)
    bpm = snapshot["range_bpm"]
    if bpm is not None:
        label(frame, f'{bpm:.1f} BPM  {snapshot["range_quality"]:.0%}', 1040, 42, .62, MINT)
    if events and age < 1.4 and snapshot["state"] == "sensing":
        label(frame, events[-1].name.upper(), 29, HEIGHT-58, .79, AMBER, 2)
    label(frame, f'{health["reason"]}   {health["remaining"]:.0f}s available', 29, HEIGHT-28, .47, DIM)
    label(frame, "V  view     H  hide     M  mute     R  reset timer", 849, HEIGHT-28, .47, DIM)
    return frame


@lru_cache(maxsize=1)
def ambient_scan():
    """A display reference pattern, independent of microphone evidence."""
    r = np.sqrt((PORTRAIT_X/.82)**2+((PORTRAIT_Y+.06)/.70)**2)
    scan = np.uint8(np.clip(65+36*(1-r)+8*np.cos(r*43), 0, 255))
    small = cv2.applyColorMap(scan, cv2.COLORMAP_TURBO)
    small[r>1] = 0
    return cv2.resize(small, (WIDTH, HEIGHT))


def render_ml(snapshot, health, view, portrait=None):
    if view == "debug":
        frame = render_nearfield(snapshot, health, view)
    else:
        if portrait is None or (snapshot.get("display_subject") and snapshot.get("subject") is None):
            model = render_portrait(dict(snapshot, subject=snapshot.get("display_subject")),
                                    health, "pure")
        else:
            model = portrait
        opacity = snapshot.get("opacity", 0.)
        if opacity < 1 and snapshot.get("display_subject"):
            model = cv2.convertScaleAbs(model, alpha=opacity)
        background = (cv2.convertScaleAbs(ambient_scan(), alpha=1-.66*opacity)
                      if opacity > 0 else ambient_scan())
        frame = cv2.max(background, model)
        position = snapshot.get("position")
        state = snapshot["ml_state"]
        if position and state.startswith("Hand_"):
            x, depth = position
            point = (round(WIDTH*(.5+.37*x)), round(HEIGHT*(1-(depth-.15)/.8)))
            direction = {"Hand_Push": (0, 45), "Hand_Pull": (0, -45),
                         "Hand_Swipe_Left": (-65, 0), "Hand_Swipe_Right": (65, 0)}[state]
            start = (point[0]-direction[0], point[1]-direction[1])
            cv2.line(frame, start, point, (70, 154, 249), 8, cv2.LINE_AA)
            cv2.circle(frame, point, 13, (200, 245, 255), -1, cv2.LINE_AA)
        if view == "pure":
            return frame
    subject = snapshot.get("subject")
    status = snapshot["ml_state"].replace("_", " ") if subject else "Waiting for a qualified echo"
    label(frame, "BAT CAM   /   ACOUSTIC MODEL", 29, 41, .58)
    label(frame, status, 29, 68, .47, MINT if subject else DIM)
    label(frame, "MODELED PORTRAIT" if subject else "AMBIENT DISPLAY", 29, 91, .39, DIM)
    if health.get("simulated"):
        label(frame, "SYNTHETIC", 29, 115, .42, AMBER)
    if subject:
        label(frame, f'Z {subject[1]*100:.0f} cm   {snapshot["confidence"]:.0%}', 1010, 41, .54)
    if snapshot["range_bpm"] is not None:
        label(frame, f'{snapshot["range_bpm"]:.1f} BPM', 1090, 70, .55, MINT)
    signal_wave = snapshot["echo_wave"]
    span = max(.0004, float(np.percentile(abs(signal_wave), 92)))
    trace(frame, signal_wave, 1000, 635, 244, 30, -span, span, MINT)
    label(frame, f'Mic {snapshot["mic_dbfs"]:.0f} dBFS', 1000, 690, .45, DIM)
    label(frame, f'{health["reason"]}    V view   H hide   M mute   R reset timer',
          29, 690, .45, DIM)
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
