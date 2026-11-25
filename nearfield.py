"""Stereo pulse compression, near-field MTI and echo-driven motion tracking."""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal
from sonar_settings import SonarSettings

RATE = 48000
PERIOD = 1920
SLOT = PERIOD // 2
CHIRP_SAMPLES = 512
SPEED = 343.
FRAME_RATE = RATE // PERIOD
NEAR, FAR = .15, .95
METRES_PER_SAMPLE = SPEED / (2 * RATE)
RANGES = np.linspace(NEAR, FAR, 161)
X = np.linspace(-1., 1., 64)
FRACTIONS = np.linspace(-.5, .5, 17)
FIT_LAGS = np.arange(-5, 6) % PERIOD


@dataclass(frozen=True)
class MotionEvent:
    name: str
    time: float
    confidence: float


class ChirpProbe:
    """One windowed chirp per speaker. CW pilots occupy the latter part of its slot."""

    def __init__(self, carriers, settings=None):
        settings = settings or SonarSettings()
        self.wave = np.zeros((PERIOD, 2))
        self.filters = []
        self.pilot_refs = []
        t = np.arange(CHIRP_SAMPLES)/RATE
        window = signal.windows.tukey(CHIRP_SAMPLES, .26)
        frequencies = np.fft.fftfreq(PERIOD, 1/RATE)
        sweep = (settings.low_hz, settings.high_hz)
        for side, ends in enumerate((sweep, sweep[::-1])):
            chirp = window*signal.chirp(t, ends[0], CHIRP_SAMPLES/RATE, ends[1])
            packet = self.wave[:, side]
            offset = SLOT*side
            packet[offset:offset+CHIRP_SAMPLES] = chirp
            template = np.fft.fft(packet)
            band = (frequencies >= sweep[0]-250) & (frequencies <= sweep[1]+250)
            # Remove the dominant pilot bins from the pulse compressor.
            for carrier in carriers:
                band &= abs(frequencies-carrier) > 110
            filt = template.conj()*band
            point = np.fft.ifft(template*filt)
            self.filters.append(filt/point[0].real)
            guard = np.arange(CHIRP_SAMPLES+40, SLOT)
            phase = 2*np.pi*carriers[side]*guard/RATE
            taper = signal.windows.tukey(len(guard), .55)
            packet[offset+guard] = .25*taper*np.sin(phase)
            self.pilot_refs.append((offset+guard, 8*taper*np.exp(-1j*phase)/sum(taper**2)))
        self.filters = np.array(self.filters)
        spectra = np.fft.fft(self.wave, axis=0).T
        delays = np.exp(-2j*np.pi*np.fft.fftfreq(PERIOD)*FRACTIONS[:, None])
        kernels = np.fft.ifft(
            self.filters[None, :, None, :]*spectra[:, None, None, :]*delays, axis=-1)
        self.kernels = np.tile(kernels, 2)
        self.fit = np.array([kernels[spk, spk][:, FIT_LAGS] for spk in range(2)])
        self.fit_energy = np.sum(abs(self.fit)**2, axis=2)

    def compress(self, audio):
        spectrum = np.fft.fft(audio, axis=0)
        return np.fft.ifft(self.filters[:, :, None]*spectrum[None, :, :], axis=1)


class TorsoRate:
    """Track one continuous complex return and qualify its periodic rate."""

    def __init__(self):
        self.sos = signal.butter(2, (.12, .55), fs=FRAME_RATE, btype="bandpass", output="sos")
        self.reset()

    def reset(self):
        self.history = deque(maxlen=30*FRAME_RATE)
        self.cell = None
        self.seconds = 0.
        self.next_update = 0.
        self.bpm = None
        self.quality = 0.
        self.wave = np.empty(0)

    def push(self, t, cell, iq):
        if self.cell is not None and (cell[:2] != self.cell[:2] or abs(cell[2]-self.cell[2]) > .045):
            self.reset()
        if self.cell is None:
            self.cell = cell
            self.next_update = t + .5
        self.history.append((t, iq))
        self.seconds = t-self.history[0][0]
        if len(self.history) < 25 or t < self.next_update:
            return
        self.next_update += .5
        z = np.array([value for _, value in self.history])
        candidates = []
        for values, minimum in ((np.unwrap(np.angle(z)), .009),
                                (abs(z)/(abs(z).mean()+1e-12), .006)):
            raw = signal.detrend(values)
            filtered = signal.sosfiltfilt(self.sos, raw)
            candidates.append((filtered, raw, minimum))
        self.wave = max(candidates, key=lambda item: np.std(item[0]))[0]
        if self.seconds < 7:
            return
        estimates = []
        for wave, raw, minimum in candidates:
            if np.std(wave) < minimum:
                continue
            freq, power = signal.periodogram(raw, fs=FRAME_RATE, window="hann", nfft=4096)
            band = (freq >= max(.12, 1.9/self.seconds)) & (freq <= .55)
            if not band.any():
                continue
            i = np.argmax(np.where(band, power, 0))
            period = round(FRAME_RATE/freq[i])
            if len(wave) < 2*period:
                continue
            corr = np.corrcoef(wave[:-period], wave[period:])[0, 1]
            purity = power[(freq >= .12) & (freq <= .55)].sum()/(power.sum()+1e-12)
            quality = min(corr, purity)
            if np.isfinite(quality) and quality >= .65:
                estimates.append((quality, freq[i]*60, wave))
        if estimates:
            self.quality, rate, self.wave = max(estimates, key=lambda value: value[0])
            self.bpm = rate if self.bpm is None else .68*self.bpm + .32*rate
        else:
            self.bpm, self.quality = None, 0.


class NearField:
    def __init__(self, channels, carriers, direct_path=.15, settings=None, probe=None):
        self.settings = settings or SonarSettings()
        self.probe = probe or ChirpProbe(carriers, self.settings)
        self.channels = channels
        self.carriers = np.asarray(carriers)
        self.direct_path = direct_path
        self.ranges = RANGES
        self.breath = TorsoRate()
        self.events = deque(maxlen=8)
        self.samples = np.zeros((PERIOD, channels))
        self.reset()

    def reset(self):
        self.cursor = 0
        self.grid = np.zeros((len(RANGES), len(X)), np.float32)
        self.reference = None
        self.ref_index = None
        self.confirmed = np.zeros((2, self.channels), bool)
        self.clutter = None
        self.torso_activity = None
        self.gain = None
        self.iq = None
        self.spec = np.zeros((2, 65))
        self.spec_floor = None
        self.waterfall = deque(maxlen=100)
        self.snr = np.zeros(2)
        self.echo_power = 0.
        self.noise_power = 0.
        self.stereo_energy = np.zeros(2)
        self.cir = np.zeros_like(RANGES)
        self.locked = False
        self.first_lock = None
        self.last_valid = None
        self.last_gesture = -10
        self.position = None
        self.track = deque()
        self.breath.reset()

    def push(self, block):
        offset = block.start % PERIOD
        if offset != self.cursor:
            self.cursor = 0
            return
        self.samples[offset:offset+len(block.samples)] = block.samples
        self.cursor = (offset+len(block.samples)) % PERIOD
        if self.cursor:
            return
        timing = (block.dac_time-block.adc_time)*RATE
        delay = round(timing) if np.isfinite(timing) else 0
        self.analyze(self.samples, delay, (block.start+len(block.samples))/RATE)

    def analyze(self, audio, nominal_lag, t):
        profiles = self.probe.compress(audio)
        if self.reference is None:
            self.reference = profiles.copy()
        else:
            self.reference += .50*(profiles-self.reference)
        if self.ref_index is None:
            # Some WASAPI devices report ADC/DAC time in whole callback blocks.
            # Search the possible buffer phases for each coded transmission.
            search = (nominal_lag + np.arange(0, PERIOD, 480)[:, None]
                      + np.arange(-72, 150)).ravel() % PERIOD
        else:
            search = (self.ref_index[:, :, None] + np.arange(-10, 11)) % PERIOD
        found = np.take_along_axis(abs(self.reference).transpose(0, 2, 1),
                                   np.broadcast_to(search, (2, self.channels, search.shape[-1])), axis=2)
        best = found.argmax(axis=2, keepdims=True)
        index = np.take_along_axis(np.broadcast_to(search, found.shape), best, axis=2)[:, :, 0]
        peak = found.max(axis=2)
        floor = np.median(abs(self.reference), axis=1)
        instantaneous = abs(profiles[np.arange(2)[:, None], index, np.arange(self.channels)])
        snr = peak/(floor+1e-10)
        coherent = (peak > 2e-7) & (instantaneous > .4*peak)
        # An acquired direct path may hold at a lower SNR on the same delay.
        # Acquiring a path from noise still requires the higher threshold.
        if self.ref_index is not None:
            displacement = (index-self.ref_index+PERIOD//2) % PERIOD-PERIOD//2
            self.confirmed &= abs(displacement) <= 3
        valid = coherent & ((snr >= 5) | (self.confirmed & (snr >= 3.2)))
        self.confirmed = valid
        self.snr = 20*np.log10(np.maximum(snr.max(axis=1), 1e-5))
        self.locked = bool(valid.any())
        if not self.locked:
            self.echo_power = self.noise_power = 0.
            self.stereo_energy.fill(0)
            self.cir.fill(0)
            self.ref_index = None
            self.grid *= .55
            self.position = None
            self.track.clear()
            if self.last_valid is None or t-self.last_valid > .12:
                self.breath.reset()
            return
        self.last_valid = t
        if self.first_lock is None:
            self.first_lock = t
        self.ref_index = np.where(valid, index, nominal_lag % PERIOD)
        pilots = np.zeros_like(peak)
        for spk, (offsets, weights) in enumerate(self.probe.pilot_refs):
            pilot = audio[(index[spk]+offsets[:, None]) % PERIOD, np.arange(self.channels)]
            pilots[spk] = abs(weights @ pilot)
        gain = np.maximum(peak, .6*np.minimum(pilots, 2.5*peak))
        self.gain = gain.copy() if self.gain is None else np.maximum(.98*self.gain, gain)
        rows = np.rint((RANGES-self.direct_path/2)/METRES_PER_SAMPLE).astype(int)
        ordered = profiles.transpose(0, 2, 1)
        arrivals = np.take_along_axis(ordered, (index[:, :, None]+rows) % PERIOD, axis=2)
        direct = np.take_along_axis(ordered, index[:, :, None], axis=2)
        self.iq = arrivals/(direct+1e-12)

        # Estimate and subtract out-of-gate paths as nuisance terms before
        # accepting any matched peak. They never enter the scene or trackers.
        offsets = np.arange(-72, 620)
        measured = self._echoes(profiles, index, peak, valid, offsets)
        squared = abs(measured)**2
        self.cir = squared.max(axis=(0, 1))
        self.echo_power = float(self.cir[RANGES >= .20].max())
        self.stereo_energy = squared[:, :, RANGES >= .20].sum(axis=(1, 2))
        clean = self._mti(measured, valid)
        self._scene(measured, clean, valid, t)
        self._diagnostics(clean)

    def _echoes(self, profiles, index, peak, valid, offsets):
        # Each accepted tap is tagged by its own interpolated path length.
        n = len(RANGES)
        measured = np.zeros((2, self.channels, n), complex)
        floors = []
        for mic in range(self.channels):
            usable = valid[:, mic]
            if not usable.any():
                continue
            anchors = index[:, mic]
            grid = (anchors[:, None]+offsets) % PERIOD
            residual = profiles[:, :, mic].copy()
            direct = np.ones(2, complex)
            # A single transmitted echo appears in both matched filters.
            # Subtract its complete two-filter response to prevent code leakage.
            for pass_index in range(2):
                for spk in np.flatnonzero(usable):
                    coefficient, _ = self._remove_tap(residual, spk, anchors[spk])
                    if pass_index == 0:
                        direct[spk] = coefficient
                    else:
                        direct[spk] += coefficient
            for _ in range(14):
                response = np.take_along_axis(abs(residual), grid, axis=1)*usable[:, None]
                threshold = np.maximum(.022*peak[:, mic],
                                       self.settings.noise_multiplier*np.median(response, axis=1))
                scores = response/(threshold[:, None]+1e-12)
                spk, i = np.unravel_index(scores.argmax(), scores.shape)
                if scores[spk, i] <= 1:
                    break
                coefficient, fraction = self._remove_tap(residual, spk, grid[spk, i])
                distance = (offsets[i]+fraction)*METRES_PER_SAMPLE + self.direct_path/2
                if self.settings.blanking_m <= distance <= FAR:
                    row = round((distance-NEAR)/(FAR-NEAR)*(n-1))
                    scale = max(self.gain[spk, mic], 1e-10)
                    value = coefficient*np.exp(-1j*np.angle(direct[spk]))/scale
                    if abs(value) > .043:
                        measured[spk, mic, row] += value
            for spk in np.flatnonzero(usable):
                residual_power = abs(residual[spk, grid[spk]])**2
                floors.append(float(np.median(residual_power))/max(self.gain[spk, mic]**2, 1e-20))
        self.noise_power = float(np.median(floors)) if floors else 0.
        return measured

    def _remove_tap(self, residual, spk, index):
        basis = self.probe.fit[spk]
        sample = residual[spk, (index+FIT_LAGS) % PERIOD]
        energy = self.probe.fit_energy[spk]
        coefficients = np.sum(basis.conj()*sample, axis=1)/energy
        i = int(np.argmax(abs(coefficients)**2*energy))
        coefficient = coefficients[i]
        origin = -int(index) % PERIOD
        residual -= coefficient*self.probe.kernels[spk, :, i, origin:origin+PERIOD]
        return coefficient, FRACTIONS[i]

    def _mti(self, measured, valid):
        if self.clutter is None:
            self.clutter = np.zeros_like(measured)
        # The tap locations can move by one bin as sub-sample delay changes.
        background = self.clutter
        clean = measured-background
        background += self.settings.alpha*clean*valid[:, :, None]
        return clean*valid[:, :, None]

    def _scene(self, measured, clean, valid, t):
        amplitude = abs(measured)
        power = amplitude**2
        energy = power.sum(axis=(0, 1))
        peaks, _ = signal.find_peaks(np.r_[0, energy, 0], distance=7, height=.0005)
        peaks -= 1
        target = np.zeros_like(self.grid)
        motion = []
        for row in peaks:
            near = slice(max(0, row-7), row+8)
            path = power[:, :, near].sum(axis=2)
            speaker = path.sum(axis=1)
            mic = path.sum(axis=0)
            ratios = []
            if valid.any(axis=1).all():
                ratios.append((speaker[1]-speaker[0])/(speaker.sum()+1e-10))
            if len(mic) == 2 and valid.any(axis=0).all():
                ratios.append((mic[1]-mic[0])/(mic.sum()+1e-10))
            lateral = float(np.clip(1.25*np.mean(ratios), -1, 1)) if ratios else 0.
            depth = RANGES[row]
            activity = float(abs(clean[:, :, near]).max())
            support = float(np.sqrt(energy[row]))
            # Keep an initial foreground view, then suppress stationary taps.
            level = np.clip(np.log1p(10*min(support, activity)), 0, 1)
            level *= 1.05-.48*(depth-NEAR)/(FAR-NEAR)
            if ratios:
                shape = np.exp(-.5*((X-lateral)/.16)**2)
            else:
                shape = np.ones_like(X)*.65
            depth_shape = np.exp(-.5*((RANGES-depth)/.024)**2)
            np.maximum(target, level*depth_shape[:, None]*shape, out=target)
            if activity > .018:
                motion.append((activity, depth, lateral, bool(ratios)))
        self.grid *= self.settings.persistence
        self.grid += (1-self.settings.persistence)*target
        self.position = None
        if motion:
            _, depth, x, horizontal = max(motion)
            self.position = (x, depth) if horizontal else None
            self._gesture(t, depth, x, horizontal)
        elif self.track and t-self.track[-1][0] > .14:
            self.track.clear()
        self._track_torso(measured, clean, valid, t)

    def _gesture(self, t, depth, x, horizontal):
        if self.track and (t-self.track[-1][0] > .14 or abs(depth-self.track[-1][1]) > .06):
            self.track.clear()
        self.track.append((t, depth, x, horizontal))
        while self.track and t-self.track[0][0] > .54:
            self.track.popleft()
        first, previous_depth, previous_x, previous_horizontal = self.track[0]
        if t-first < .16 or t-self.last_gesture < .65 or len(self.track) < 5:
            return
        dz, dx = depth-previous_depth, x-previous_x
        if horizontal and previous_horizontal and abs(dx) > .45 and abs(dz) < .10:
            name = "Swipe Right" if dx > 0 else "Swipe Left"
            confidence = min(1., abs(dx))
        elif abs(dz) > .075:
            name = "Pull" if dz > 0 else "Push"
            confidence = min(1., abs(dz)/.16)
        else:
            return
        self.events.append(MotionEvent(name, t, confidence))
        self.last_gesture = t
        self.track.clear()

    def _track_torso(self, measured, clean, valid, t):
        zone = (RANGES >= .25) & (RANGES <= .85)
        score = abs(measured)*zone
        variation = np.minimum(abs(clean), score)
        if self.torso_activity is None:
            self.torso_activity = variation.copy()
        else:
            self.torso_activity += .035*(variation-self.torso_activity)
        score *= .20+np.minimum(1, self.torso_activity/.04)
        best = np.unravel_index(score.argmax(), score.shape)
        if self.breath.cell is not None:
            spk, mic, depth = self.breath.cell
            local = score[spk, mic]*(abs(RANGES-depth) <= .035)
            row = int(local.argmax())
            competing = (self.breath.seconds > 2 and self.breath.bpm is None
                         and score[best] > 2.5*local[row])
            if valid[spk, mic] and abs(measured[spk, mic, row]) > .014 and not competing:
                anchor = int(np.argmin(abs(RANGES-depth)))
                self.breath.push(t, (spk, mic, depth), self.iq[spk, mic, anchor])
                return
        spk, mic, row = best
        if abs(measured[spk, mic, row]) > .022:
            self.breath.push(t, (spk, mic, RANGES[row]), self.iq[spk, mic, row])
        else:
            self.breath.reset()

    def _diagnostics(self, clean):
        # Temporal envelope spectra are derived only from accepted range taps.
        # Report a gate-limited delay profile for the debug view.
        profile = abs(clean).max(axis=1)
        profile = np.array([np.interp(np.linspace(NEAR,FAR,65), RANGES, row) for row in profile])
        self.spec_floor = profile.copy() if self.spec_floor is None else .98*self.spec_floor+.02*profile
        self.spec = np.clip(20*np.log10(1+profile/(self.spec_floor+.005)), 0, 50)
        self.waterfall.append(self.spec.copy())


class NearFieldDemo:
    def __init__(self, probe, direct_path=.15):
        self.direct = direct_path
        self.spectra = np.fft.fft(probe.wave, axis=0)
        self.omega = -2j*np.pi*np.fft.fftfreq(PERIOD)
        self.cached = None

    def block(self, start, frames):
        offset = start % PERIOD
        if offset == 0 or self.cached is None:
            t = start/RATE
            chest = .0018*np.sin(2*np.pi*.32*t)
            arm = None
            for onset, length, src, dst in (
                    (1., .55, (.59, -.32), (.31, -.32)),
                    (2., .55, (.29, .30), (.56, .30)),
                    (3., .50, (.40, -.9), (.40, .9)),
                    (4., .50, (.40, .9), (.40, -.9))):
                if onset <= t <= onset+length:
                    arm = np.asarray(src)+(np.asarray(dst)-src)*(t-onset)/length
            reflectors = [(.62+chest, .05, .009), (1.13, -.1, .012), (.085, .2, .003)]
            if arm is not None:
                reflectors.append((*arm, .021))
            scene = np.zeros((PERIOD, 2))
            for mic in range(2):
                paths = np.zeros(PERIOD, complex)
                for spk in range(2):
                    direct_lag = 92+4*spk+2*mic
                    paths += .040*self.spectra[:, spk]*np.exp(self.omega*direct_lag)
                    for depth, x, level in reflectors:
                        delay = direct_lag+(depth-self.direct/2)/METRES_PER_SAMPLE
                        side = (2*spk-1)*.42+(2*mic-1)*.37
                        paths += level*(1+x*side)*self.spectra[:, spk]*np.exp(self.omega*delay)
                scene[:, mic] = np.fft.ifft(paths).real
            noise = .000035*np.random.default_rng(start).normal(size=scene.shape)
            self.cached = (scene+noise).astype(np.float32)
        return self.cached[offset:offset+frames]
