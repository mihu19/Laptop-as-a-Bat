"""Measure real near-field returns and save settings only after confirmation."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import monotonic, sleep

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_engine import AudioEngine, MAX_GAIN, RATE
from main import Worker
from ml_acoustic_model import AcousticModel, load_model
from nearfield import FAR, RANGES, NearField
from sonar_settings import load_settings, save_settings
from visualizer_cam import CameraOutput, render_ml, render_nearfield, render_portrait

MAX_TEST_GAIN = min(.010, MAX_GAIN)


def foreground(frame):
    mask = np.uint8(np.maximum(frame[:, :, 0], np.maximum(frame[:, :, 1], frame[:, :, 2])) >= 35)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count == 1:
        return mask != 0, None
    index = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index, stats[index, :4].tolist()


def assess(samples, coverage, ious, widths, portrait=False):
    ratios = [max(-100., 10*np.log10(s["echo"]/s["noise"]))
              if s["locked"] and s["echo"] > 0 and s["noise"] > 0 else -100. for s in samples]
    energies = np.sum([s["energy"] for s in samples], axis=0)
    total = float(energies.sum())
    imbalance = float(abs(energies[0]-energies[1])/total) if total else 1.
    waves = [s["wave"] for s in samples if s["bpm"] is not None]
    bpm_valid = bool(len(waves) >= 30 and np.ptp(waves) > 1e-8)
    # Range checks inspect the gated products. Residual acoustic noise is
    # measured separately, before accepting any range bins.
    outside = RANGES > FAR
    leakage = max(float(s["cir"][outside].sum()+s["grid"][outside].sum()) for s in samples)
    result = dict(snr_db=float(np.median(ratios)), peak_snr_db=max(ratios),
                  detection_fraction=float(np.mean(np.asarray(ratios) > -100)),
                  silhouette_coverage=float(np.median(coverage)),
                  frame_iou=float(np.median(ious)) if ious else 0.,
                  localized_fraction=float(np.mean((np.asarray(widths) > 0)
                                                     & (np.asarray(widths) <= .80))),
                  stereo_imbalance=imbalance, stereo_energy=energies.tolist(),
                  bpm_valid=bpm_valid, bpm=samples[-1]["bpm"], out_of_gate_energy=leakage,
                  lock_fraction=float(np.mean([s["locked"] for s in samples])))
    result["checks"] = dict(snr=result["snr_db"] >= 12,
                            coverage=(.15 if portrait else .08) <= result["silhouette_coverage"]
                            <= (.35 if portrait else .45)
                            and result["localized_fraction"] >= .75,
                            iou=result["frame_iou"] >= (.80 if portrait else .75),
                            stereo=imbalance <= .15 and total > 0,
                            breathing=bpm_valid, range_gate=leakage == 0.)
    if portrait:
        result["subject_fraction"] = float(np.mean([s["subject"] for s in samples]))
        result["checks"]["subject"] = result["subject_fraction"] >= .75
    return result


def capture(settings, seconds, folder, camera_enabled):
    samples, coverage, ious, widths = [], [], [], []
    previous = None
    best_image = None
    best_coverage = 0.
    folder.mkdir(parents=True, exist_ok=True)
    health = dict(reason="diagnostic", remaining=seconds, simulated=False)
    ml = AcousticModel(load_model()) if settings.ml_band else None
    with AudioEngine(carriers=settings.carriers, gain=settings.gain, imaging="chirp", max_seconds=seconds,
                     settings=settings) as engine:
        near = NearField(engine.channels, settings.carriers, settings=settings, probe=engine.tx.probe)
        with Worker(None, engine, nearfield=near, diagnostics=True, model=ml) as worker, CameraOutput(camera_enabled) as camera:
            if ml:
                sleep(3)
            engine.tx.arm()
            start = deadline = monotonic()
            next_diagnostic = start
            ml_peak = 0.
            rendered = 0
            while monotonic()-start < seconds:
                engine.tx.heartbeat = monotonic()
                if worker.error is not None:
                    raise RuntimeError("DSP worker failed") from worker.error
                snap = worker.latest
                draw = render_portrait if settings.low_band else render_nearfield
                frame = draw(snap, health, "pure")
                display = render_ml(snap, health, "hud", portrait=frame) if ml else frame
                camera.latest = display
                rendered += 1
                if ml:
                    ml_peak = max(ml_peak, snap["ml_ms"])
                    if monotonic() >= next_diagnostic:
                        snr = snap["echo_snr_db"]
                        snr_text = f"{snr:.1f} dB" if snr is not None else "unavailable"
                        print(f'Peak Mic {snap["mic_dbfs"]:.1f} dBFS | Echo SNR {snr_text} | '
                              f'Input Channels {engine.channels} | {snap["ml_state"]}', flush=True)
                        next_diagnostic += 1
                if monotonic()-start >= .5:
                    mask, bounds = foreground(frame)
                    fraction = float(mask.mean())
                    coverage.append(fraction)
                    widths.append(bounds[2]/frame.shape[1] if bounds else 0.)
                    if fraction > best_coverage:
                        best_image, best_coverage = frame, fraction
                    if previous is not None:
                        union = np.count_nonzero(previous | mask)
                        ious.append(np.count_nonzero(previous & mask)/union if union else 0.)
                    previous = mask
                    sample = snap["tuning"]
                    if not samples or sample["time"] > samples[-1]["time"]:
                        samples.append(sample)
                deadline += 1/30
                sleep(max(0, deadline-monotonic()))
            engine.tx.mute()
        output_seconds = engine.tx.total_emitted/RATE
        faults, drops = engine.xruns, engine.ring.drops
    if not samples:
        raise RuntimeError("Diagnostic did not receive microphone frames.")
    report = assess(samples, coverage, ious, widths, settings.low_band)
    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    report.update(display_coverage=float(np.mean(display.max(axis=2) >= 35)),
                  display_mean_brightness=float(gray.mean()), peak_ml_ms=ml_peak,
                  render_fps=round(rendered/seconds, 2))
    if ml:
        report["checks"]["display"] = (.15 <= report["display_coverage"] <= .65
                                       and report["display_mean_brightness"] > 30)
    report.update(settings=asdict(settings), duration=seconds, output_seconds=output_seconds,
                  xruns=faults, drops=drops, camera_frames=camera.frames, camera_error=camera.error,
                  camera=camera.status, frames_scored=len(coverage), face_verified=False,
                  contour_bbox=bounds, peak_coverage=best_coverage,
                  pixel_histogram=np.histogram(frame.max(axis=2), bins=np.arange(0, 257, 16))[0].tolist())
    report["checks"]["stream"] = faults == drops == 0 and (
        not camera_enabled or (camera.frames > 0 and camera.error is None))
    report["passed"] = all(report["checks"].values())
    best = max(samples, key=lambda sample: sample["echo"])
    chart = np.zeros((240, 640, 3), np.uint8)
    profile = best["cir"]/max(float(best["cir"].max()), 1e-20)
    points = np.column_stack((np.linspace(25, 615, len(profile)), 200-165*profile)).astype(np.int32)
    cv2.polylines(chart, [points], False, (90, 230, 240), 1)
    cv2.putText(chart, "Gated CIR   0.15 to 0.95 m", (25, 226),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1)
    radar = cv2.applyColorMap(np.uint8(np.clip(best["grid"][::-1], 0, 1)*255),
                             cv2.COLORMAP_INFERNO)
    for name, image in (("debug_cir.png", chart), ("debug_radar.png", radar),
                        ("debug_output_cam.png", display),
                        ("debug_best_cam.png", best_image if best_image is not None else frame)):
        if not cv2.imwrite(str(folder / name), image):
            raise OSError(f"Could not write {name}")
    (folder/"debug_metrics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def tune(settings, result):
    minimum, maximum = (.15, .35) if settings.low_band else (.08, .45)
    if result["silhouette_coverage"] > maximum or result["out_of_gate_energy"] != 0.:
        return replace(settings, blanking_m=.18,
                       noise_multiplier=min(6, settings.noise_multiplier+.6))
    if not result["checks"]["snr"] or result["silhouette_coverage"] < minimum:
        return replace(settings, gain=min(MAX_TEST_GAIN, settings.gain*1.55),
                       alpha=max(.004, settings.alpha*.67),
                       noise_multiplier=max(3., settings.noise_multiplier-.30))
    if not result["checks"]["iou"]:
        return replace(settings, persistence=min(.86, settings.persistence+.08),
                       smoothing=min(1.4, settings.smoothing+.45))
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("sonar_debug"))
    parser.add_argument("--iterations", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--preset", choices=("default", "standard", "wide", "stable"), default="default")
    parser.add_argument("--start-gain", type=float)
    parser.add_argument("--no-virtual-cam", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    settings = load_settings()
    gain = settings.gain if args.start_gain is None else args.start_gain
    settings = replace(settings, gain=min(MAX_TEST_GAIN, gain))
    if args.preset == "standard":
        settings = replace(settings, low_hz=17500, high_hz=21500)
    if args.preset in ("wide", "stable"):
        settings = replace(settings, low_hz=17000, high_hz=22000)
    if args.preset == "stable":
        settings = replace(settings, persistence=.8, smoothing=.8)
    reports = []
    for i in range(args.iterations):
        result = capture(settings, 6, args.output/f"burst-{i+1}", not args.no_virtual_cam)
        reports.append(result)
        print(json.dumps(result), flush=True)
        if all(value for key, value in result["checks"].items() if key != "breathing"):
            result = capture(settings, 20, args.output/"confirmation", not args.no_virtual_cam)
            reports.append(result)
            print(json.dumps(result), flush=True)
            if result["passed"]:
                save_settings(settings)
            break
        if not result["checks"]["stream"]:
            break
        next_settings = tune(settings, result)
        if next_settings == settings:
            break
        settings = next_settings
    (args.output/"debug_metrics.json").write_text(json.dumps(reports, indent=2)+"\n", encoding="utf-8")
    return 0 if reports[-1]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
