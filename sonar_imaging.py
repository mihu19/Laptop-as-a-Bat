"""Coherent multitone pulse compression on a repeating stereo probe."""
from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy import signal

RATE = 48000
PERIOD = 1920
PROFILE = PERIOD // 2
SPEED = 343.
DELAYS = np.arange(32, 482, 4)
RANGES = DELAYS * SPEED / (2 * RATE)
X = np.linspace(-1, 1, 64)
FPS = RATE // PERIOD


class Probe:
    def __init__(self, carriers):
        bins = np.arange(720, 861)
        hz = bins * RATE / PERIOD
        bins = bins[np.min(abs(hz[:, None]-np.asarray(carriers)), axis=1) > 575]
        self.bins = [bins[bins % 2 == i] for i in range(2)]
        self.wave = np.empty((PERIOD, 2))
        self.inverse = []
        self.kernels = []
        for i, indices in enumerate(self.bins):
            k = np.arange(len(indices))
            spectrum = np.zeros(PERIOD//2+1, complex)
            spectrum[indices] = np.exp(1j*np.pi*k*(k+1)/len(k))
            wave = np.fft.irfft(spectrum, n=PERIOD)
            self.wave[:, i] = wave / np.max(abs(wave))
            self.inverse.append(1 / np.fft.rfft(self.wave[:, i])[indices])
            relative_delay = DELAYS[:, None]-DELAYS
            kernel = np.exp(2j*np.pi/PERIOD * relative_delay[:, :, None] * indices).mean(axis=2)
            self.kernels.append(kernel)

    def profiles(self, samples):
        fft = np.fft.rfft(samples, axis=0)
        profiles = []
        for indices, inverse in zip(self.bins, self.inverse):
            spec = np.zeros((PERIOD, samples.shape[1]), complex)
            spec[indices] = fft[indices]*inverse[:, None]
            profiles.append(np.fft.ifft(spec, axis=0).T)
        return np.array(profiles)

    def separate(self, residual, floor):
        """Subtract the known sidelobes of a few isolated foreground echoes."""
        remaining = residual.copy()
        echoes = np.zeros_like(remaining)
        for spk, kernel in enumerate(self.kernels):
            gate = np.maximum(floor[spk], .14*abs(remaining[spk]).max(axis=1)[:, None])
            for mic in range(remaining.shape[1]):
                row = remaining[spk, mic]
                for _ in range(4):
                    index = int(np.argmax(abs(row)))
                    peak = row[index]
                    if abs(peak) <= gate[mic, index]:
                        break
                    echoes[spk, mic, index] += peak
                    row -= peak * kernel[:, index]
        return echoes


class RangeBreath:
    def __init__(self, ranges):
        self.sos = signal.butter(2, [0.12, 0.55], fs=FPS, btype="bandpass", output="sos")
        self.torso_zone = (ranges >= .3) & (ranges <= 1)
        self.clear()

    def clear(self):
        self.history = deque(maxlen=30*FPS)
        self.cell = None
        self.bpm = None
        self.quality = 0.
        self.wave = np.empty(0)
        self.count = 0

    def update(self, z, strengths, valid):
        scores = strengths * self.torso_zone * valid[:, :, None]
        best = np.unravel_index(scores.argmax(), scores.shape)
        if scores[best] < .009:
            self.clear()
            return
        if self.cell is None or not valid[self.cell[:2]] or scores[self.cell] < .25*scores[best]:
            self.clear()
            self.cell = best
        self.history.append(z[self.cell])
        self.count += 1
        if self.count % FPS:
            return
        data = np.array(self.history)
        phase = np.unwrap(np.angle(data))
        amp = abs(data)
        self.bpm, self.quality = None, 0.
        for series, minimum in ((phase, .008), (amp/(amp.mean()+1e-12), .004)):
            detrended = signal.detrend(series)
            filtered = signal.sosfiltfilt(self.sos, detrended)
            if series is phase:
                self.wave = filtered
            if len(data) < 10*FPS or np.std(filtered) < minimum:
                continue
            freqs, power = signal.periodogram(detrended, fs=FPS, window="hann", nfft=4096)
            band = (freqs >= .12) & (freqs <= .55)
            freq = freqs[band][power[band].argmax()]
            period = round(FPS/freq)
            if len(data) < 2.1 * period:
                continue
            corr = np.corrcoef(filtered[:-period], filtered[period:])[0, 1]
            purity = power[band].sum() / (power.sum()+1e-20)
            quality = min(corr, purity)
            if np.isfinite(quality) and quality > max(.64, self.quality):
                self.bpm = 60*freq
                self.quality = quality
                self.wave = filtered


class RangeCamera:
    def __init__(self, channels, carriers, signature=None, path=None, direct_path=0):
        self.probe = Probe(carriers)
        self.samples = np.zeros((PERIOD, channels))
        self.ranges = RANGES + direct_path/2
        self.breath = RangeBreath(self.ranges)
        self.marks = np.array([-1., 1.])
        self.path = Path(path) if path is not None else None
        self.signature = signature
        self.calibrated = False
        self.calibration_status = ""
        if self.path is not None:
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                marks = np.asarray(saved["marks"], float)
                if (saved["route"] == signature and marks.shape == (2,)
                        and np.all(np.isfinite(marks)) and np.max(abs(marks)) <= 1
                        and abs(marks[1]-marks[0]) >= .15):
                    self.marks = marks
                    self.calibrated = True
            except (ValueError, OSError, KeyError, TypeError):
                pass
        self.reset()

    def reset(self, keep_background=False):
        self.cursor = 0
        if not keep_background:
            self.background = None
            self.noise = None
            self.activity = None
        elif self.activity is not None:
            self.activity.fill(0)
        self.reference = None
        self.previous_echo = None
        self.locked = False
        self.grid = np.zeros((len(RANGES), len(X)), np.float32)
        self.cue = None
        self.position = None
        if getattr(self, "guide", None) is not None:
            self.calibration_status = "Calibration interrupted - press C to retry"
        self.guide = None
        self.breath.clear()

    def calibrate(self, t):
        if self.locked:
            self.guide = (t, [], [])
            self.calibration_status = "Wave left"
        else:
            self.calibration_status = "Wait for range lock"

    def guide_step(self, t):
        if self.guide is None:
            return
        start, left, right = self.guide
        age = t-start
        if age < 3:
            self.calibration_status = "Wave left" if age < 1.5 else "Wave right"
            if self.cue is not None:
                (left if age < 1.5 else right).append(self.cue)
            return
        self.guide = None
        if min(len(left), len(right)) < 6:
            self.calibration_status = "Too few echoes - press C to retry"
            return
        marks = np.array([np.median(left), np.median(right)])
        scatter = max(np.median(abs(np.array(left)-marks[0])),
                      np.median(abs(np.array(right)-marks[1])))
        if abs(marks[1]-marks[0]) < max(.15, 3*scatter):
            self.calibration_status = "Left/right overlap - press C to retry"
            return
        self.marks = marks
        self.calibrated = True
        self.calibration_status = "Left/right calibrated"
        if self.path is not None:
            pending = self.path.with_suffix(".tmp")
            try:
                pending.write_text(json.dumps(dict(route=self.signature, marks=marks.tolist()),
                                              indent=2)+"\n", encoding="utf-8")
                pending.replace(self.path)
            except OSError:
                self.calibration_status = "Calibrated for this session"

    def process(self, samples, learn):
        n = len(samples)
        self.samples[self.cursor:self.cursor+n] = samples
        self.cursor += n
        if self.cursor != PERIOD:
            return
        self.cursor = 0
        raw = self.probe.profiles(self.samples)
        if self.reference is None:
            self.reference = raw.copy()
        else:
            self.reference += .35*(raw-self.reference)
        magnitude = abs(self.reference[:, :, :PROFILE])
        direct = magnitude.argmax(axis=2)
        peak = magnitude.max(axis=2)
        floor = np.median(magnitude, axis=2)
        valid = (peak > 3.6*floor) & (peak > 1e-10)
        raw_at_direct = np.take_along_axis(raw, direct[:, :, None], axis=2)
        valid &= abs(raw_at_direct[:, :, 0]) > .3*peak
        self.locked = bool(valid.any())
        if not self.locked:
            self.grid *= .7
            self.breath.clear()
            self.cue = None
            self.position = None
            self.previous_echo = None
            return
        # Each path uses its direct arrival as time and phase reference. This
        # removes arbitrary duplex latency and tracks slow sample-clock slip.
        ref = raw_at_direct
        bins = direct[:, :, None] + DELAYS
        z = np.take_along_axis(raw, bins, axis=2)/(ref+1e-15)
        # Relative complex phase across repetitions must be evaluated at a
        # consistent delay even when the direct sample index advances by one.
        z *= valid[:, :, None]
        if self.background is None:
            self.background = z.copy()
            self.noise = np.full(z.shape, .002)
            self.activity = np.zeros(z.shape)
        residual = (z - self.background)*valid[:, :, None]
        if learn:
            self.background += .06*residual
            self.noise += .06*(abs(residual)-self.noise)*valid[:, :, None]
            return
        threshold = 3*self.noise + .008
        separated = self.probe.separate(residual, threshold)
        strength = np.maximum(abs(separated) - threshold, 0)
        # Protect persistent foreground. Empty cells follow slow room changes.
        self.background += np.where(strength < .006, .005, .000015)*residual
        power = strength**2
        speakers = power.sum(axis=1)
        mics = power.sum(axis=0)
        total = speakers.sum(axis=0)
        activity = total
        if self.previous_echo is not None:
            delta = abs(z-self.previous_echo)
            activity = total * np.clip((delta**2).sum(axis=(0, 1))/(total+1e-12), 0, 1)
        self.previous_echo = z
        cues = []
        if valid.any(axis=1).all():
            cues.append((speakers[1]-speakers[0])/(total+1e-14))
        if mics.shape[0] == 2 and valid.any(axis=0).all():
            duplicate = np.linalg.norm(self.samples[:, 1]-self.samples[:, 0])
            if duplicate > .02*np.linalg.norm(self.samples):
                balance = (mics[1]-mics[0])/(total+1e-14)
                # The phase repeats every wavelength. Retain it only as a
                # bounded refinement of the unambiguous amplitude contrast.
                cross = np.sum(residual[:, 1]*residual[:, 0].conj(), axis=0)
                coherence = abs(cross)/(np.sqrt(mics.prod(axis=0))+1e-12)
                refinement = .12 * np.sin(np.angle(cross)) * np.clip(coherence, 0, 1)
                cues.append(np.clip(balance + refinement, -1, 1))
        self.cue = None
        self.position = None
        if cues:
            contrast = np.mean(cues, axis=0)
            if activity.max() > .0003:
                self.cue = float(contrast[activity.argmax()])
            sign = np.sign(self.marks[1]-self.marks[0])
            pos = np.interp(sign*contrast, sign*self.marks, [-1, 1])
            if self.cue is not None and self.calibrated:
                self.position = float(pos[activity.argmax()])
            lobes = np.exp(-.5*((X-pos[:, None])/.2)**2)
        else:
            lobes = 1.
        intensity = np.clip(np.log1p(6*np.sqrt(total))/(1+self.ranges*.5), 0, 1)
        paint = intensity[:, None]*lobes
        spread = .85*paint
        np.maximum(spread[1:], .65*paint[:-1], out=spread[1:])
        np.maximum(spread[:-1], .65*paint[1:], out=spread[:-1])
        np.maximum(spread[2:], .26*paint[:-2], out=spread[2:])
        np.maximum(spread[:-2], .26*paint[2:], out=spread[:-2])
        self.grid *= .72
        self.grid += .28*spread
        # Adjacent delay cells can trade peak strength during chest motion.
        breathing_strength = np.maximum(abs(residual) - threshold, 0)
        self.activity += .045*(breathing_strength-self.activity)
        self.breath.update(z, self.activity, valid)


def demo_echo(probe, start, frames, time):
    """Generate delayed microphone arrivals using the same PCM probe as hardware."""
    index = np.arange(frames) + start
    mic = np.zeros((frames, 2))
    for speaker in range(2):
        for channel in range(2):
            path = probe.wave[:, speaker]
            mic[:, channel] += .014*path[(index-87-4*speaker-2*channel) % PERIOD]
            mic[:, channel] += .005*path[(index-470-4*speaker-2*channel) % PERIOD]
            if time < 1.55:
                continue
            motion = .65*np.sin(2*np.pi*.37*time)
            for depth, lateral, gain in (
                    (.68 + .0015*np.sin(2*np.pi*.25*time), .22, .0045),
                    (.40+.06*np.sin(time*2.8), motion, .0028)):
                delay = 87+4*speaker+2*channel + 2*depth*RATE/SPEED
                offset = index-delay
                sample = np.floor(offset).astype(int)
                u = offset-sample
                echo = (1-u)*path[sample % PERIOD] + u*path[(sample+1) % PERIOD]
                side = (1 if speaker else -1)*.30 + (1 if channel else -1)*.55
                mic[:, channel] += gain*(1+lateral*side)*echo
    return mic
