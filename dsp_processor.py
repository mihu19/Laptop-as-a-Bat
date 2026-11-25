"""Streaming CW sonar. All DSP state is owned by one consumer of audio blocks."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal

from audio_engine import CARRIERS, HOP, RATE, validate_carriers

NFFT = 3840
BIN_HZ = RATE / NFFT
OFFSETS = np.arange(-32, 33)
EPS = 1e-16


@dataclass(frozen=True)
class Gesture:
    name: str
    time: float
    confidence: float


class Gestures:
    def __init__(self):
        self.log = deque(maxlen=8)
        self.reset()

    def reset(self):
        self.rows = []
        self.quiet = 0
        self.pending = None
        self.cooldown = 0
        self.overlong = False

    def update(self, t, sides, x, lateral):
        if sides.sum() > 0:
            if not self.overlong and t >= self.cooldown:
                self.rows.append((t, x, *sides.sum(axis=0), *sides.flatten()))
                if len(self.rows) > 110:
                    self.overlong = True
                    self.rows.clear()
            self.quiet = 0
            return
        self.quiet += 1
        if self.quiet < 8:
            if len(self.rows) < 3:
                self.rows.clear()
            return
        self.overlong = False
        if len(self.rows) >= 3:
            self._classify(t, lateral)
        self.rows.clear()
        if self.pending is not None and t - self.pending > 0.48:
            self.pending = None

    def _emit(self, name, t, confidence):
        self.log.append(Gesture(name, t, float(np.clip(confidence, 0, 1))))
        self.cooldown = t + 0.30
        self.pending = None

    def _classify(self, t, lateral):
        rows = np.asarray(self.rows)
        energy = rows[:, 2] + rows[:, 3]
        rows = rows[energy > 0.10 * energy.max()]
        times, x, pos, neg = rows[:, :4].T
        total = pos + neg
        duration = times[-1] - times[0] + HOP / RATE
        if duration < 0.07:
            return
        third = max(1, len(x) // 3)
        first, last = np.mean(x[:third]), np.mean(x[-third:])
        onset, release = times[0], times[-1]
        if (duration <= 0.27 and max(abs(first), abs(last)) < 0.6
                and pos[:third].sum() > 2 * neg[:third].sum()
                and neg[-third:].sum() > 2 * pos[-third:].sum()):
            if self.pending is not None and 0.09 <= onset - self.pending <= 0.41:
                self._emit("Double Tap", t, min(pos[:third].sum() / total[:third].sum(),
                                                neg[-third:].sum() / total[-third:].sum()))
            else:
                self.pending = release
            return
        left_peak = times[np.argmax(total * (1 - x))]
        right_peak = times[np.argmax(total * (1 + x))]
        if lateral and 0.12 <= duration <= 0.80:
            if first < -0.25 and last > 0.25 and right_peak - left_peak >= 0.03:
                self._emit("Swipe Right", t, (last - first) / 2)
                return
            if first > 0.25 and last < -0.25 and left_peak - right_peak >= 0.03:
                self._emit("Swipe Left", t, (first - last) / 2)
                return
        if duration <= 0.60:
            by_carrier = rows[:, 4:].sum(axis=0).reshape(2, 2)
            strong = by_carrier.sum(axis=1) > 0.05 * total.sum()
            for column, name in enumerate(("Push", "Pull")):
                if np.all(by_carrier[strong, column] > 2 * by_carrier[strong, 1-column]):
                    self._emit(name, t, by_carrier[:, column].sum() / total.sum())
                    return


class Breathing:
    """Demodulated slow time at 20 Hz, independent of the gesture FFT bins."""

    def __init__(self, paths):
        self.paths = paths
        self.lp = signal.butter(4, 3, fs=100, output="sos")
        self.bp = signal.butter(2, (0.1, 0.6), btype="bandpass", fs=20, output="sos")
        self.reset("acquiring")

    def reset(self, state):
        self.state = state
        self.bpm = None
        self.quality = 0.0
        self.history = deque(maxlen=1200)
        self.recent = deque(maxlen=3)
        self.lp_state = np.zeros((len(self.lp), 2, self.paths), complex)
        self.bp_state = np.zeros((len(self.bp), 2, 4 * self.paths))
        self.count = 0
        self.previous = self.phase = None
        self.best = 0

    def push(self, iq, baseline, noise, usable):
        # Normalize direct-path phase and gain before comparing independent paths.
        z = (iq / np.where(np.abs(baseline) > 1e-8, baseline, 1)).ravel()
        slow, self.lp_state = signal.sosfilt(self.lp, (z - 1)[None, :],
                                             axis=0, zi=self.lp_state)
        self.count += 1
        if self.count % 5:
            return
        residual = slow[0]
        carrier = 1 + residual
        if self.previous is None:
            self.phase = np.angle(carrier)
        else:
            self.phase += np.angle(carrier * self.previous.conj())
        self.previous = carrier
        # Signed quadratures survive a fade or a static-return subtraction null.
        features = np.concatenate((self.phase, np.abs(carrier)-1,
                                   residual.real, residual.imag))
        filtered, self.bp_state = signal.sosfilt(self.bp, features[None, :],
                                                 axis=0, zi=self.bp_state)
        if self.count <= 1000:
            return
        self.history.append(filtered[0])
        if self.count % 100 == 0 and len(self.history) >= 600:
            self._estimate(noise.ravel(), usable.ravel())

    @staticmethod
    def period(wave):
        wave = signal.detrend(wave)
        scale = wave.std()
        if scale < 1e-5:
            return None
        peaks, _ = signal.find_peaks(wave, distance=32, prominence=0.7 * scale)
        if len(peaks) < 4:
            return None
        period = np.median(np.diff(peaks))
        if not 32 <= period <= 205:
            return None
        # Look near the independently counted cycle to avoid octave aliases.
        lags = np.arange(max(32, int(period*0.86)), min(201, int(period*1.14)) + 1)
        if len(lags) < 3:
            return None
        corr = signal.correlate(wave, wave, mode="full", method="fft")[len(wave)-1:]
        energy = np.r_[0, np.cumsum(wave**2)]
        corr = corr[lags] / np.sqrt(energy[-1-lags] * (energy[-1] - energy[lags]) + EPS)
        i = int(np.argmax(corr))
        lag = float(lags[i])
        if 0 < i < len(corr)-1:
            a, b, c = corr[i-1:i+2]
            if abs(a - 2*b + c) > 1e-8:
                lag += np.clip(0.5*(a-c)/(a-2*b+c), -0.5, 0.5)
        bpm = 1200 / lag
        if corr[i] < 0.55 or abs(bpm - 1200 / period) > 2 or not 5.9 <= bpm <= 36.2:
            return None
        return float(np.clip(bpm, 6, 36)), float(np.clip(corr[i], 0, 1)), scale

    def _estimate(self, noise, usable):
        matrix = np.asarray(self.history)
        estimates = []
        for i, wave in enumerate(matrix.T):
            path = i % self.paths
            if not usable[path]:
                continue
            result = self.period(wave)
            if result:
                bpm, quality, scale = result
                if scale > max(0.00025, 1.5 * noise[path]):
                    estimates.append((quality, bpm, i))
        estimates.sort(reverse=True)
        if estimates:
            for quality, bpm, i in estimates:
                agree = [b for _, b, j in estimates if abs(b-bpm) <= 2
                         and (j % self.paths != i % self.paths or j < self.paths)]
                if agree:
                    self.best = i
                    self.recent.append(float(np.median([bpm, *agree])))
                    self.bpm = float(np.median(self.recent))
                    self.quality = quality
                    self.state = "tracking" if len(matrix) == 1200 else "provisional"
                    return
        self.bpm, self.quality, self.state = None, 0.0, "weak signal"
        self.recent.clear()

    def waveform(self):
        return np.fromiter((v[self.best] for v in self.history), dtype=np.float32)


class DSPProcessor:
    def __init__(self, channels=2, carriers=CARRIERS, calibration=3.0, quiet=5.0):
        self.carriers = np.array(validate_carriers(carriers))
        if channels not in (1, 2) or not (0.1 <= calibration <= 30 and 0.1 <= quiet <= 30):
            raise ValueError("Use one or two input channels and 0.1..30 s calibration.")
        self.channels = channels
        self.calibration, self.quiet = calibration, quiet
        self.window = signal.windows.hann(NFFT, sym=False)[:, None]
        self.bins = np.rint(self.carriers / BIN_HZ).astype(int)[:, None] + OFFSETS
        self.hz = self.bins * BIN_HZ - self.carriers[:, None]
        self.ref = np.exp(-2j*np.pi * self.carriers[:, None] * np.arange(HOP) / RATE)
        self.ref *= signal.windows.hann(HOP, sym=False) * (4 / HOP)
        self.buffer = np.zeros((NFFT, channels), np.float32)
        self.radar = np.zeros((81, 64), np.float32)
        self.x_axis = np.linspace(-1, 1, 64)
        self.velocity_rows = np.clip(np.rint(40 - 10 * 343 * self.hz /
                                             (2*self.carriers[:, None])).astype(int), 0, 80)
        self.spectrum = np.zeros((2, len(OFFSETS)), np.float32)
        self.shifts = np.full(2, np.nan)
        self.bandwidths = np.zeros((2, 2))
        self.waterfall = deque(maxlen=100)
        self.breath = Breathing(2 * channels)
        self.gestures = Gestures()
        self.ambient = None
        self.learning = []
        self.iq_learning = []
        self.active = False
        self.expected = self.hops = self.learn_hops = 0
        self.state = "ambient calibration"
        self.zone = None
        self.zones = {}
        self.cue_maps = []
        self.target = None
        self.snr = np.zeros(2)

    @property
    def ready(self):
        return self.ambient is not None

    def mark_zone(self, name):
        if name not in ("left", "center", "right") or self.state != "sensing":
            return False
        self.zone = (name, self.expected / RATE + 2, [])
        self.gestures.reset()
        return True

    def _transition(self, active):
        self.active = active
        self.state = "quiet carrier calibration" if active else "muted"
        self.learning.clear()
        self.iq_learning.clear()
        self.learn_hops = 0
        self.buffer.fill(0)
        self.breath.reset("acquiring" if active else "muted")
        self.gestures.reset()
        self.radar.fill(0)
        self.spectrum.fill(0)
        self.shifts.fill(np.nan)
        self.bandwidths.fill(0)
        self.waterfall.clear()
        self.target = self.zone = None

    def process(self, samples, active=True, start=None, gap=False):
        if samples.shape != (HOP, self.channels):
            raise ValueError(f"Expected {HOP} frames and {self.channels} channels.")
        start = self.expected if start is None else start
        if gap or start != self.expected:
            self._transition(active and self.ready)
            if not self.ready:
                self.state = "ambient calibration"
        if self.ready and active != self.active:
            self._transition(active)
        self.expected = start + HOP
        self.hops += 1
        self.buffer[:-HOP] = self.buffer[HOP:]
        self.buffer[-HOP:] = samples
        fft = np.fft.rfft(self.window * self.buffer, axis=0)
        power = np.abs(fft[self.bins].transpose(0, 2, 1) * (4/NFFT)) ** 2
        if not self.ready:
            if active:
                self.learning.clear()
                self.learn_hops = 0
                return
            self.learn_hops += 1
            self.learning.append(power)
            if self.learn_hops >= round(self.calibration * RATE / HOP):
                self.ambient = np.median(self.learning, axis=0) + EPS
                self._transition(False)
            return
        if not active:
            return
        iq = (self.ref @ samples) * np.exp(-2j*np.pi*self.carriers*start/RATE)[:, None]
        if self.state == "quiet carrier calibration":
            self.learn_hops += 1
            if self.learn_hops >= 8:
                self.learning.append(power)
                self.iq_learning.append(iq)
            if self.learn_hops >= round(self.quiet * RATE / HOP):
                self._calibrate()
            return
        t = self.expected / RATE
        corrected = iq * np.exp(-1j * (self.drift*(t-self.origin) + self.correction))
        self._analyze(power, corrected, samples, t)

    def _calibrate(self):
        data = np.asarray(self.learning)
        floor = np.median(data, axis=0)
        self.floor = np.maximum(floor, self.ambient)
        self.mad = np.median(abs(data - floor), axis=0)
        trace = np.asarray(self.iq_learning)
        time = (np.arange(len(trace)) - len(trace) + 1) * HOP / RATE
        centered = time - time.mean()
        phases = np.unwrap(np.angle(trace), axis=0)
        self.drift = np.sum(centered[:, None, None] * phases, axis=0) / max(EPS, centered @ centered)
        aligned = trace * np.exp(-1j*time[:, None, None]*self.drift)
        self.baseline = aligned.mean(axis=0)
        self.correction = np.zeros_like(self.drift)
        self.origin = self.expected / RATE
        self.gains = np.maximum(abs(self.baseline)**2, 1e-12)
        self.noise = np.median(abs(aligned-self.baseline), axis=0) / np.sqrt(self.gains)
        # An unrelated pilot already present while muted cannot qualify a TX path.
        ambient_carrier = self.ambient[:, :, 31:34].max(axis=2)
        self.path_snr = 10*np.log10(self.gains / np.maximum(
            ambient_carrier, np.median(self.ambient, axis=2)))
        self.snr = self.path_snr.max(axis=1)
        self.usable = self.path_snr >= 15
        self.state = "sensing"
        self.learning.clear()
        self.iq_learning.clear()

    def _position(self, cues):
        if not self.cue_maps:
            return np.zeros(np.shape(cues)[1:])
        values, weights = [], []
        for i, marks, weight in self.cue_maps:
            orientation = np.sign(marks[2] - marks[0])
            values.append(np.interp(orientation*cues[i], orientation*marks, [-1, 0, 1]))
            weights.append(weight)
        return np.average(values, axis=0, weights=weights)

    def _sample_zone(self, t, cues, motion):
        name, end, readings = self.zone
        if motion:
            readings.append(cues.copy())
        if t < end:
            return
        self.zone = None
        self.zones.pop(name, None)
        self.cue_maps.clear()
        if len(readings) >= 10:
            self.zones[name] = np.asarray(readings)
        if len(self.zones) == 3:
            zones = [self.zones[k] for k in ("left", "center", "right")]
            medians = np.asarray([np.median(z, axis=0) for z in zones])
            spread = np.mean([np.median(abs(z-m), axis=0) for z, m in zip(zones, medians)], axis=0)
            for i, marks in enumerate(medians.T):
                diffs = np.diff(marks)
                if diffs.prod() > 0 and min(abs(diffs)) > max(0.08, 2*spread[i]):
                    self.cue_maps.append((i, marks, 1/max(0.01, spread[i])**2))

    @staticmethod
    def _contrasts(energy):
        speakers = energy.sum(axis=1)
        mics = energy.sum(axis=0)
        denom = speakers.sum(axis=0) + EPS
        return np.array([(speakers[1]-speakers[0])/denom, (mics[-1]-mics[0])/denom])

    def _track_shifts(self, spectra):
        for i, power in enumerate(spectra):
            active = np.flatnonzero(power)
            if not active.size:
                self.shifts[i] = np.nan
                self.bandwidths[i] = 0
                continue
            offsets = self.hz[i, active]
            self.bandwidths[i] = (max(0, -offsets.min()), max(0, offsets.max()))
            peak = int(power.argmax())
            estimate = self.hz[i, peak]
            if 0 < peak < len(power)-1:
                left, center, right = power[peak-1:peak+2]
                curvature = left - 2*center + right
                if curvature < -EPS:
                    estimate += BIN_HZ*np.clip(0.5*(left-right)/curvature, -0.5, 0.5)
            previous = self.shifts[i]
            if np.isfinite(previous) and abs(estimate-previous) < 50:
                estimate = 0.65*estimate + 0.35*previous
            self.shifts[i] = estimate

    def _analyze(self, power, iq, samples, t):
        threshold = np.maximum.reduce((12*self.floor, self.floor + 8*self.mad,
                                       np.broadcast_to(self.gains[:, :, None]*2e-5, power.shape)))
        mask = (power > threshold) & (abs(self.hz[:, None]) >= 36) & (abs(self.hz[:, None]) <= 400)
        adjacent = np.zeros_like(mask)
        adjacent[:, :, 1:] |= mask[:, :, :-1]
        adjacent[:, :, :-1] |= mask[:, :, 1:]
        residual = np.where(mask & adjacent & self.usable[:, :, None],
                            (power-self.floor)/self.gains[:, :, None], 0)
        self._track_shifts(residual.sum(axis=1))
        ratio = np.maximum(power / np.maximum(self.floor, EPS), 1)
        self.spectrum = np.clip(10*np.log10(ratio.max(axis=1)), 0, 50)
        if self.hops % 4 == 0:
            self.waterfall.append(self.spectrum.copy())
        energy = residual.sum(axis=2)
        motion = energy.sum() > 0
        cues = self._contrasts(energy)
        x = float(self._position(cues))
        if self.zone is not None:
            self._sample_zone(t, cues, motion)
        else:
            sides = np.stack([(residual*(self.hz[:, None] > 0)).sum(axis=(1, 2)),
                              (residual*(self.hz[:, None] < 0)).sum(axis=(1, 2))], axis=1)
            self.gestures.update(t, sides, x, bool(self.cue_maps))
        self.radar *= np.exp(-0.01 / 0.35)
        self.target = None
        if motion:
            rows = np.zeros((2, self.channels, 81))
            for speaker, indices in enumerate(self.velocity_rows):
                for mic in range(self.channels):
                    np.add.at(rows[speaker, mic], indices, residual[speaker, mic])
            row_power = rows.sum(axis=(0, 1))
            positions = self._position(self._contrasts(rows))
            lobes = np.exp(-0.5*((self.x_axis - positions[:, None])/0.14)**2) if self.cue_maps else 1
            strength = np.clip(np.log1p(row_power*10000)/5, 0, 1)[:, None]
            np.maximum(self.radar, strength*lobes, out=self.radar)
            row = int(row_power.argmax())
            if self.cue_maps:
                self.target = (float(positions[row]), row)
        else:
            self.floor += (np.minimum(power, 3*self.floor) - self.floor) / 3000
        present = self.usable & (abs(iq) > 0.15*np.sqrt(self.gains))
        if motion or np.max(np.abs(samples)) >= 0.99 or not np.any(present):
            reason = "motion" if motion else "weak signal"
            if self.breath.count or self.breath.state != reason:
                self.breath.reset(reason)
        else:
            if self.breath.count == 0:
                self.breath.state = "acquiring"
            phase_error = np.angle(iq * self.baseline.conj())
            self.correction += 0.0006 * phase_error
            self.breath.push(iq, self.baseline, self.noise, present)

    def snapshot(self):
        fall = np.zeros((100, 2, 65), np.float32)
        if self.waterfall:
            fall[-len(self.waterfall):] = self.waterfall
        progress = min(1, self.learn_hops / (100*(self.quiet if self.active else self.calibration)))
        return dict(time=self.expected/RATE, state=self.state, ready=self.ready, progress=progress,
                    radar=self.radar.copy(), target=self.target, spectrum=self.spectrum.copy(),
                    shifts=self.shifts.copy(), bandwidths=self.bandwidths.copy(),
                    waterfall=fall, waveform=self.breath.waveform(), bpm=self.breath.bpm,
                    quality=self.breath.quality, breath_state=self.breath.state,
                    clean_seconds=self.breath.count/100,
                    events=tuple(self.gestures.log), lateral=bool(self.cue_maps),
                    zones=tuple(self.zones), zone=self.zone[0] if self.zone else None,
                    snr=self.snr.copy(), carriers=self.carriers.copy())
