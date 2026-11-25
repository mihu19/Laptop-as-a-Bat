"""Estimate a front projection from bistatic speaker-to-reflector-to-mic delays."""

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from nearfield import FRAME_RATE, PERIOD, RATE, RANGES, SLOT, SPEED

# Approximate laptop geometry in metres. X is right, Y is up, Z is toward the user.
SPEAKERS = np.array([[-.145, 0., .16], [.145, 0., .16]])
MICROPHONES = np.array([[-.04, .22, -.03], [.04, .22, -.03]])
ROOM_FRAMES = 2*FRAME_RATE
MIN_PATHS = 3


class SpatialImager:
    def __init__(self, direct_path, channels, speakers=SPEAKERS, microphones=MICROPHONES):
        self.speakers = np.asarray(speakers)
        self.microphones = np.asarray(microphones)
        self.direct_lengths = np.linalg.norm(self.speakers[:, None]-self.microphones[None], axis=-1)
        self.channels = channels
        self.x = np.linspace(-.55, .55, 45)
        self.y = np.linspace(.70, -.25, 39)
        depth = np.linspace(.24, 1., 39)
        points = np.stack(np.meshgrid(self.x, self.y, depth, indexing="xy"), axis=-1)
        to_speaker = np.linalg.norm(points-self.speakers[:, None, None, None, :], axis=-1)
        to_mic = np.linalg.norm(points-self.microphones[:, None, None, None, :], axis=-1)
        path = to_speaker[:, None]+to_mic[None]
        equivalent = (path-self.direct_lengths[:, :, None, None, None]+direct_path)/2
        in_range = (equivalent >= RANGES[0]) & (equivalent <= RANGES[-1])
        self.inside = in_range.all(axis=(0, 1))
        self.step = RANGES[1]-RANGES[0]
        self.delay_bins = (equivalent-RANGES[0])/self.step
        self.path_index = np.arange(4).reshape(2, 2, 1, 1, 1)
        self.reset()

    def reset(self):
        self.image = np.zeros((len(self.y), len(self.x)), np.float32)
        self.room = [[] for _ in range(4)]
        self.anchors = [[] for _ in range(4)]
        self.ceiling = np.full((4, len(RANGES)), np.inf)
        self.calibrated = np.zeros(4, bool)
        self.bins = None
        self.status = "Waiting for three timing paths"
        self.valid_paths = 0
        self.echo_paths = 0

    @property
    def ready(self):
        return bool(np.count_nonzero(self.calibrated) >= MIN_PATHS)

    def update(self, echoes, valid, noise_power, audio, indices=None):
        self.valid_paths = int(np.count_nonzero(valid))
        self.echo_paths = 0
        self.image.fill(0)
        if self.channels != 2 or np.allclose(audio[:, 0], audio[:, 1], rtol=.002, atol=2e-7):
            self.status = "Two independent mic channels required"
            return
        amplitude = abs(echoes).reshape(4, -1)
        if not self.ready:
            for path in np.flatnonzero(valid.ravel() & ~self.calibrated):
                self.room[path].append(amplitude[path])
                if indices is not None:
                    self.anchors[path].append(int(indices.flat[path]))
                if len(self.room[path]) == ROOM_FRAMES:
                    ceiling = np.percentile(self.room[path], 90, axis=0)
                    self.ceiling[path] = maximum_filter1d(ceiling, size=5)
                    self.calibrated[path] = True
                    self.room[path].clear()
        usable = valid.ravel() & self.calibrated
        if np.count_nonzero(usable) < MIN_PATHS:
            self.status = ("Learning the room   /   stand clear" if self.valid_paths >= MIN_PATHS
                           else "Waiting for three timing paths")
            return
        if self.bins is None:
            self._align_paths()

        # A location qualifies only where at least three paths support a new reflector.
        threshold = np.maximum(.014, 2.5*np.sqrt(noise_power))
        excess = np.maximum(amplitude-self.ceiling-threshold, 0)
        self.echo_paths = int(np.count_nonzero(usable & (excess.max(axis=1) > .015)))
        # Weight closer delay matches more heavily within the timing tolerance.
        likelihood = 5*gaussian_filter1d(excess, 2., axis=-1, mode="constant")
        views = likelihood[self.path_index, self.bins].reshape(4, *self.inside.shape)
        active = views[usable]
        third = len(active)-MIN_PATHS
        support = np.partition(active, third, axis=0)[third]
        projection = (support*self.inside).max(axis=-1)
        floor = max(.018, .16*projection.max())
        self.image[:] = np.clip((projection-floor)*6, 0, 1)
        self.status = "Corroborated echoes" if self.image.any() else "Waiting for a shared echo   /   move in range"

    def _align_paths(self):
        offsets = np.zeros(4)
        measured = np.array([len(indices) == ROOM_FRAMES for indices in self.anchors])
        if np.count_nonzero(measured) >= MIN_PATHS:
            for path in np.flatnonzero(measured):
                indices = np.asarray(self.anchors[path])
                first = indices[0]
                offsets[path] = first+np.median((indices-first+SLOT//2) % SLOT-SLOT//2)
            offsets[measured] -= self.direct_lengths.ravel()[measured]*(RATE/SPEED)
            first = offsets[measured][0]
            relative = (offsets[measured]-first+SLOT/2) % SLOT-SLOT/2
            offsets[measured] = (relative-np.median(relative))*SPEED/(2*RATE*self.step)
        shift = offsets.reshape(2, 2, 1, 1, 1)
        self.bins = np.clip(np.rint(self.delay_bins-shift), 0, len(RANGES)-1).astype(np.intp)

    def lost(self):
        self.valid_paths = 0
        self.echo_paths = 0
        self.image.fill(0)
        self.status = "Waiting for three timing paths"


class SpatialDemo:
    """Generate separate path delays for a few point reflectors, without a body template."""

    def __init__(self, probe, imager):
        self.imager = imager
        self.spectra = np.fft.rfft(probe.wave, axis=0).T
        self.omega = -2j*np.pi*np.fft.rfftfreq(PERIOD)
        self.cached = None

    def block(self, start, frames):
        offset = start % PERIOD
        if offset == 0 or self.cached is None:
            t = start/RATE
            points = [(.28, .11, .79)]
            levels = [.075]
            if t >= 2.6:
                shift = .13*np.sin(2*np.pi*(t-2.6)/4)
                points.extend([(shift-.11, .40, .51), (shift+.18, .17, .68)])
                levels.extend([.34, .30])
            targets = np.asarray(points)
            spk_dist = np.linalg.norm(targets-self.imager.speakers[:, None], axis=-1)
            mic_dist = np.linalg.norm(targets-self.imager.microphones[:, None], axis=-1)
            paths = spk_dist[:, None]+mic_dist[None]
            direct = self.imager.direct_lengths
            all_lengths = np.concatenate((direct[:, :, None], paths), axis=-1)
            delays = 71 + (RATE/SPEED)*all_lengths
            gains = np.asarray([1., *levels])
            transfer = (np.exp(delays[..., None]*self.omega)*gains[None, None, :, None]).sum(axis=2)
            scene = np.fft.irfft((.035*self.spectra[:, None]*transfer).sum(axis=0), axis=-1).T
            noise = np.random.default_rng(start).normal(0, .000025, scene.shape)
            self.cached = (scene+noise).astype(np.float32)
        return self.cached[offset:offset+frames]
