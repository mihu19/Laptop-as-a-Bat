"""Live range tracking using the sounder from Laptop Acoustic Radar.

The periodic Newman-phase sounder and least-squares channel estimate follow
https://github.com/enricguaschmesia-lab/laptop-acoustic-radar/tree/b00695c5472ce705532aaa1cb02bf3a87d2cceba.

MIT License

Copyright (c) 2026 Enric Guasch Mesià

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import deque
from math import lcm

import numpy as np

from nearfield import METRES_PER_SAMPLE, RATE

SYMBOL_SAMPLES = 768
SUBCARRIERS = slice(96, 160)  # 6.0–9.94 kHz at 48 kHz
ROOM_SECONDS = 2.0
MIN_RANGE_M = .25
MAX_RANGE_M = 1.50
REFERENCE_RATIO = 5.0
REFERENCE_FRAMES = 3


class OfdmSounder:
    def __init__(self, block_samples):
        bins = np.arange(SUBCARRIERS.stop-SUBCARRIERS.start)
        spectrum = np.zeros(SYMBOL_SAMPLES//2+1, complex)
        spectrum[SUBCARRIERS] = np.exp(1j*np.pi*np.square(bins)/len(bins))
        self.symbol = np.fft.irfft(spectrum)
        self.symbol /= abs(self.symbol).max()
        self.reference = np.fft.rfft(self.symbol)[SUBCARRIERS]
        # A whole number of symbols and audio callbacks fits in the same buffer.
        size = lcm(SYMBOL_SAMPLES, block_samples)
        self.wave = np.zeros((size, 2), np.float32)
        self.wave[:, 0] = np.tile(self.symbol, size//SYMBOL_SAMPLES)
        self.taper = np.hanning(len(bins))

    def response(self, audio):
        periods = audio.reshape(-1, SYMBOL_SAMPLES, audio.shape[1])
        received = np.fft.rfft(periods, axis=1).mean(axis=0)[SUBCARRIERS]
        channel = received/self.reference[:, None]
        # Taper away direct-path sidelobes and interpolate the delay grid.
        return np.fft.ifft(channel*self.taper[:, None], n=SYMBOL_SAMPLES, axis=0)


class OfdmTracker:
    def __init__(self, sounder, channels, direct_path):
        self.sounder = sounder
        self.samples = np.empty((len(sounder.wave), channels), np.float32)
        self.direct_path = direct_path
        offsets = np.arange(SYMBOL_SAMPLES)*METRES_PER_SAMPLE + direct_path/2
        self.delays = np.flatnonzero((offsets >= MIN_RANGE_M) & (offsets <= MAX_RANGE_M))
        self.ranges = offsets[self.delays]
        self.calibration_frames = round(ROOM_SECONDS*RATE/len(self.samples))
        self.recent = deque(maxlen=4)
        self.reset()

    def reset(self):
        self.cursor = 0
        self.anchor = None
        self.mic = None
        self.timing_snr = None
        self.reference_frames = 0
        self.room_frames = []
        self.room = self.threshold = None
        self.echo_profile = np.zeros_like(self.ranges)
        self.echo_noise_power = 0.
        self.recent.clear()
        self.range_m = None

    def push(self, block):
        start = block.start % len(self.samples)
        if start != self.cursor:
            self.cursor = 0
            return
        end = start+len(block.samples)
        self.samples[start:end] = block.samples
        self.cursor = end % len(self.samples)
        if self.cursor == 0:
            self.analyze(self.samples)

    def analyze(self, audio):
        response = self.sounder.response(audio)
        amplitude = abs(response)
        if self.anchor is None:
            peaks = amplitude.argmax(axis=0)
            ratios = amplitude.max(axis=0)/np.maximum(np.median(amplitude, axis=0), 1e-12)
            mic = int(ratios.argmax())
            peak = int(peaks[mic])
        else:
            mic = self.mic
            search = (self.anchor+np.arange(-8, 9)) % SYMBOL_SAMPLES
            peak = int(search[amplitude[search, mic].argmax()])
        strength = amplitude[peak, mic]
        ratio = strength/max(float(np.median(amplitude[:, mic])), 1e-12)
        if ratio < REFERENCE_RATIO or strength < 2e-8:
            self.timing_snr = None
            self.reference_frames = 0
            self.echo_profile.fill(0)
            self.echo_noise_power = 0.
            self.range_m = None
            self.recent.clear()
            if self.room is None:
                self.room_frames.clear()
                self.anchor = self.mic = None
            return
        if self.anchor is None:
            self.anchor, self.mic = peak, mic
        self.reference_frames = min(REFERENCE_FRAMES, self.reference_frames+1)
        if self.reference_frames < REFERENCE_FRAMES:
            return
        self.timing_snr = 20*np.log10(ratio)
        # Remove per-frame direct-path gain and phase before comparing with the room.
        bins = (peak+self.delays) % SYMBOL_SAMPLES
        profile = response[bins, mic]/response[peak, mic]
        if self.room is None:
            self.room_frames.append(profile)
            if len(self.room_frames) == self.calibration_frames:
                self.room = np.mean(self.room_frames, axis=0)
                variation = abs(np.array(self.room_frames)-self.room)
                self.threshold = np.maximum(.012, 3*np.percentile(variation, 95, axis=0))
                self.room_frames.clear()
            return

        echo = abs(profile-self.room)
        noise = float(np.median(echo))
        supported = echo > np.maximum(self.threshold, 4*noise)
        self.echo_noise_power = max(noise, .004)**2
        self.echo_profile = np.where(supported, np.square(echo), 0.)
        if not np.any(supported):
            self.recent.clear()
            self.range_m = None
            return
        index = int(np.where(supported, echo, 0).argmax())
        candidate = float(self.ranges[index])
        if self.recent and abs(candidate-self.recent[-1]) > .14:
            self.recent.clear()
            self.range_m = None
        self.recent.append(candidate)
        if len(self.recent) == self.recent.maxlen:
            self.range_m = float(np.median(self.recent))


class OfdmDemo:
    """Generate microphone PCM with a room echo and a moving reflector."""

    def __init__(self, sounder, direct_path):
        self.period = len(sounder.wave)
        self.spectrum = np.fft.rfft(sounder.symbol)
        self.omega = -2j*np.pi*np.fft.rfftfreq(SYMBOL_SAMPLES)
        self.direct_range = direct_path/2

    def block(self, start, frames):
        offset = start % self.period
        if offset == 0:
            t = start/RATE
            room = np.exp(self.omega*112) + .16*np.exp(self.omega*331.3)
            if t >= ROOM_SECONDS+.4:
                distance = .68+.21*np.sin(2*np.pi*(t-ROOM_SECONDS-.4)/5)
                lag = 112+(distance-self.direct_range)/METRES_PER_SAMPLE
                room += .17*np.exp(self.omega*lag)
            signal = .009*np.fft.irfft(self.spectrum*room)
            stereo = np.column_stack((signal, .79*np.roll(signal, 2)))
            noise = np.random.default_rng(start).normal(0, .000025, (self.period, 2))
            self.samples = (np.tile(stereo, (self.period//SYMBOL_SAMPLES, 1))+noise).astype(np.float32)
        return self.samples[offset:offset+frames]
