"""48 kHz WASAPI duplex acquisition and callback-enforced transmission limits."""

from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic

import numpy as np
import sounddevice as sd
from sonar_imaging import Probe, PERIOD
from nearfield import ChirpProbe
from ofdm_ranging import OfdmSounder

RATE = 48_000
HOP = 480
CARRIERS = (18_500.0, 20_500.0)
MAX_GAIN = 10 ** (-30 / 20)
DEFAULT_GAIN = 10 ** (-48 / 20)


def validate_carriers(pair):
    left, right = pair
    if not (12_000 <= left < right <= 22_000 and right - left >= 1_000):
        raise ValueError("Choose two carriers in 12000..22000 Hz, at least 1000 Hz apart.")
    return float(left), float(right)


@dataclass(frozen=True)
class Block:
    samples: np.ndarray
    start: int
    active: bool
    gap: bool
    adc_time: float
    dac_time: float


class CaptureRing:
    """Transfer ownership of preallocated slots via thread-safe deque operations."""

    def __init__(self, channels, capacity=200):
        self.data = np.empty((capacity, HOP, channels), np.float32)
        self.metadata = [None] * capacity
        self.free = deque(range(capacity))
        self.ready = deque()
        self.drops = 0

    def put(self, samples, metadata):
        try:
            slot = self.free.popleft()
        except IndexError:
            self.drops += 1
            try:
                slot = self.ready.popleft()  # reclaim the oldest unread block
            except IndexError:
                return  # all slots currently owned by the consumer
        self.data[slot] = samples
        self.metadata[slot] = metadata
        self.ready.append(slot)

    def pop(self):
        try:
            slot = self.ready.popleft()
        except IndexError:
            return None
        try:
            return Block(self.data[slot].copy(), *self.metadata[slot])
        finally:
            self.free.append(slot)


class Transmitter:
    """Account in DAC frames. The same budget survives every manual toggle."""

    def __init__(self, carriers=CARRIERS, gain=DEFAULT_GAIN, session=120, window=180,
                 imaging=False, max_seconds=300, auto_cooldown=False, settings=None):
        if not (0 < gain <= MAX_GAIN and HOP/RATE <= session <= 120
                and 180 <= window <= 3600 and .01 <= max_seconds <= 86400):
            raise ValueError("Invalid output cap or time budget.")
        self.gain = gain
        self.session = round(session * RATE)
        self.window = round(window * RATE)
        self.demo_limit = round(max_seconds * RATE)
        self.auto_cooldown = auto_cooldown
        self.demo_start = None
        self.resume_at = None
        self.omega = 2 * np.pi * np.array(validate_carriers(carriers)) / RATE
        self.angles = np.arange(HOP)[:, None] * self.omega
        self.work = np.empty_like(self.angles)
        if imaging == "ofdm":
            self.probe = OfdmSounder(HOP)
        elif imaging in ("chirp", "spatial"):
            self.probe = ChirpProbe(carriers, settings, multicarrier=imaging == "spatial")
        else:
            self.probe = Probe(carriers) if imaging else None
        self.probe_only = imaging in ("chirp", "ofdm", "spatial")
        self.ramp = (1 - np.cos(np.pi * np.arange(2 * HOP) / (2 * HOP))) / 2
        self.lock = Lock()
        self.armed = False
        self.reason = "muted"
        self.heartbeat = monotonic()
        self.intervals = deque()
        self.peak = 0.0
        self.used = self.continuous = 0
        self.total_emitted = 0
        self.fault_pending = False

    def arm(self, reset_demo=False):
        with self.lock:
            if reset_demo:
                self.demo_start = None
            if self.used + HOP <= self.session and not self.armed:
                self.heartbeat = monotonic()
                self.armed = True
                self.continuous = 0
                self.reason = "armed"
                self.resume_at = None
            return self.armed

    def mute(self, reason="muted"):
        with self.lock:
            self.armed = False
            self.resume_at = None
            self.peak = 0.0
            self.reason = reason

    def fill(self, out, start, fault=False):
        out.fill(0)
        if fault:
            self.fault_pending = True
        if not self.lock.acquire(blocking=False):
            return False
        try:
            self.peak = 0.0
            cutoff = start - self.window
            while self.intervals and self.intervals[0][1] <= cutoff:
                self.intervals.popleft()
            if self.intervals and self.intervals[0][0] < cutoff:
                self.intervals[0][0] = cutoff
            self.used = sum(b - a for a, b in self.intervals)
            if self.fault_pending or (self.armed and monotonic() - self.heartbeat > 0.5):
                self.armed = False
                self.reason = "stream fault" if self.fault_pending else "control timeout"
                self.fault_pending = False
                self.resume_at = None
            if self.resume_at is not None and monotonic() - self.heartbeat > .5:
                self.resume_at = None
                self.reason = "control timeout"
            if self.resume_at is not None and start >= self.resume_at:
                if self.used + min(self.session, 15*RATE) <= self.session:
                    self.armed = True
                    self.continuous = 0
                    self.demo_start = None
                    self.resume_at = None
                    self.reason = "armed"
            if self.armed and self.demo_start is None:
                self.demo_start = start
            if self.armed and start + HOP - self.demo_start > self.demo_limit:
                self.armed = False
                self.reason = "demo timer - press R"
                if self.auto_cooldown:
                    self.resume_at = start + 2*RATE
                    self.reason = "cooldown"
            if self.armed and max(self.used, self.continuous) + len(out) > self.session:
                self.armed = False
                self.reason = "exposure budget - press R when ready"
                if self.auto_cooldown:
                    self.resume_at = start + 2*RATE
                    self.reason = "cooldown / exposure budget"
            if not self.armed:
                return False
            if self.probe_only:
                offset = start % len(self.probe.wave)
                self.work[:] = self.probe.wave[offset:offset+HOP]
            else:
                np.add(self.angles, np.remainder(start * self.omega, 2 * np.pi), out=self.work)
                np.sin(self.work, out=self.work)
            if self.probe is not None and not self.probe_only:
                offset = start % PERIOD
                self.work *= .57
                self.work += .43*self.probe.wave[offset:offset+HOP]
            np.multiply(self.work, self.gain, out=out)
            if self.continuous < len(self.ramp):
                ramp = self.ramp[self.continuous:self.continuous + len(out)]
                out[:len(ramp)] *= ramp[:, None]
            np.clip(out, -self.gain, self.gain, out=out)
            self.peak = float(max(out.max(), -out.min()))
            if self.intervals and self.intervals[-1][1] == start:
                self.intervals[-1][1] += len(out)
            else:
                self.intervals.append([start, start + len(out)])
            self.continuous += len(out)
            self.used += len(out)
            self.total_emitted += len(out)
            return True
        finally:
            self.lock.release()


def wasapi_devices(input_id=None, output_id=None):
    devices = sd.query_devices()
    hosts = sd.query_hostapis()
    indices = []
    for direction, chosen, name in (("input", input_id, "microphone"),
                                    ("output", output_id, "speakers")):
        valid = [i for i, d in enumerate(devices)
                 if hosts[d["hostapi"]]["name"] == "Windows WASAPI"
                 and d[f"max_{direction}_channels"] >= (2 if direction == "output" else 1)]
        if chosen is None:
            candidates = [i for i in valid if name in devices[i]["name"].lower()
                          and "realtek" in devices[i]["name"].lower()]
            if len(candidates) != 1:
                raise ValueError(f"Select an explicit WASAPI {direction} ID from --list-devices.")
            chosen = candidates[0]
        if chosen not in valid:
            raise ValueError(f"ID {chosen} is not a suitable WASAPI {direction}. See --list-devices.")
        indices.append(chosen)
    return tuple(indices)


def raw_settings(exclusive):
    # sounddevice 0.5.5 exposes PortAudio's ABI but lacks a public RAW keyword.
    settings = sd.WasapiSettings(exclusive=exclusive, auto_convert=False)
    settings._streaminfo.streamOption = sd._lib.eStreamOptionRaw
    return settings


class AudioEngine:
    def __init__(self, input_id=None, output_id=None, carriers=CARRIERS,
                 gain=DEFAULT_GAIN, exclusive=False, session=120, window=180,
                 max_seconds=300, auto_cooldown=False, imaging=True, settings=None):
        self.devices = wasapi_devices(input_id, output_id)
        self.channels = min(2, sd.query_devices(self.devices[0])["max_input_channels"])
        self.tx = Transmitter(carriers, gain, session, window, imaging=imaging,
                              max_seconds=max_seconds, auto_cooldown=auto_cooldown, settings=settings)
        self.exclusive = exclusive
        self.stream = None
        self.start = self.xruns = self.blocks = 0
        self.ring = CaptureRing(self.channels)

    def _callback(self, indata, outdata, frames, timing, status):
        outdata.fill(0)
        if frames != HOP:
            raise sd.CallbackAbort
        fault = bool(status)
        self.xruns += int(fault)
        active = self.tx.fill(outdata, self.start, fault)
        self.ring.put(indata, (self.start, active, fault,
                               timing.inputBufferAdcTime, timing.outputBufferDacTime))
        self.start += frames
        self.blocks += 1

    def __enter__(self):
        settings = raw_settings(self.exclusive)
        for count in range(self.channels, 0, -1):
            try:
                self.channels = count
                self.ring = CaptureRing(count)
                self.stream = sd.Stream(device=self.devices, channels=(count, 2),
                                        dtype="float32", samplerate=RATE, blocksize=HOP,
                                        latency="high", extra_settings=(settings, settings),
                                        callback=self._callback)
                self.stream.start()
                return self
            except sd.PortAudioError:
                self.close()
                if count == 1:
                    raise

    def close(self):
        self.tx.mute("closed")
        if self.stream is not None:
            try:
                self.stream.abort()
            finally:
                self.stream.close()
                self.stream = None

    def __exit__(self, *exc):
        self.close()
