"""Synthetic-trained acoustic perception over ten gated echo frames."""

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np

MODEL_PATH = Path(__file__).with_name("models") / "acoustic_sonar_ml.joblib"
LABELS = ("No_Human", "Human_Present_Idle", "Hand_Push", "Hand_Pull",
          "Hand_Swipe_Left", "Hand_Swipe_Right")
SCHEMA = 2
WINDOW = 10
RANGE = slice(0, 16)
DOPPLER = slice(16, 20)
ILD, IPD, X, Z, HAND_X, HAND_Z, SNR, VALID = range(20, 28)
STATE_FEATURES = np.r_[16:20, 25:27, 29:31]


def extract_acoustic_features(cube):
    """Summarize gated range, signed Doppler proxy, stereo and motion history."""
    history = np.asarray(cube, dtype=np.float64)
    power = np.maximum(history[:, RANGE], 0.)
    energy = power.sum(axis=1)
    range_centers = np.linspace(.175, .925, 16)
    centroid = (power @ range_centers)/(energy+1e-12)
    late = history[-3:].mean(axis=0)
    signed = np.mean(history[:, DOPPLER], axis=0)
    motion = np.median(np.diff(history[:, HAND_X:HAND_Z+1], axis=0), axis=0)*7
    return np.r_[np.log1p(24*power.mean(axis=0)), signed,
                 history[:, ILD].mean(), np.sin(history[:, IPD]).mean(),
                 np.cos(history[:, IPD]).mean(), late[X], late[Z],
                 motion,
                 (energy[-3:].mean()-energy[:3].mean())/(.02+energy.mean()),
                 centroid[-3:].mean()-centroid[:3].mean(),
                 np.mean(history[:, SNR])/30, np.mean(history[:, VALID])]


def load_model():
    import joblib

    try:
        data = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        data = None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        from scripts.train_acoustic_ml import train
        data, _ = train()
    return data


class AcousticModel:
    def __init__(self, weights):
        self.weights = weights
        self.history = deque(maxlen=WINDOW)
        self.grid = None
        self.bounds = None
        self.clear()

    def clear(self):
        self.history.clear()
        self.grid = self.bounds = self.subject = None
        self.label = LABELS[0]
        self.confidence = 1.
        self.latency_ms = 0.
        self.display_subject = None
        self.opacity = 0.

    def update(self, row, grid):
        start = perf_counter()
        self.history.append(row)
        active = grid[grid > 1e-5]
        if active.size:
            p = np.percentile(active, (5, 98))
            self.bounds = p if self.bounds is None else .85*self.bounds+.15*p
            normalized = np.clip((grid-self.bounds[0])/max(.04, np.ptp(self.bounds)), 0, 1).astype(np.float32)
        else:
            normalized = np.zeros_like(grid)
        self.grid = normalized if self.grid is None else .75*self.grid+.25*normalized
        if len(self.history) == WINDOW:
            features = extract_acoustic_features(self.history)
            w = self.weights
            z = (features-w["mean"])/w["scale"]
            hidden = np.maximum(z[STATE_FEATURES] @ w["layers"][0] + w["biases"][0], 0)
            logits = hidden @ w["layers"][1] + w["biases"][1]
            scores = np.exp(logits-logits.max())
            scores /= scores.sum()
            index = int(scores.argmax())
            # Geometry and detection evidence stay independent of the learned prior.
            usable = features[-1] >= .6 and row[VALID] > 0 and .15 <= row[Z] <= .95
            if usable and index and scores[index] >= .50:
                estimated = w["spatial"] @ z + w["offset"]
                target = .65*np.array([row[X], row[Z]])+.35*estimated
                target = np.clip(target, [-1., .15], [1., .95])
                self.subject = target if self.subject is None else .72*self.subject+.28*target
                self.label, self.confidence = LABELS[index], float(scores[index])
            else:
                self.subject = None
                self.label, self.confidence = LABELS[0], float(scores[0])
        if self.subject is not None:
            self.display_subject = tuple(self.subject)
            self.opacity = min(1., self.opacity + .25)
        else:
            self.opacity = max(0., self.opacity - .25)
            if self.opacity == 0:
                self.display_subject = None
        self.latency_ms = (perf_counter()-start)*1000

    def observe(self, near, ranges):
        row = np.zeros(28)
        row[RANGE] = np.add.reduceat(near.cir, np.arange(16)*10)
        subject = near.subject if near.subject_age >= 4 else None
        if subject and near.locked and near.echo_power > 0 and near.noise_power > 0:
            x, depth = subject
            row[X], row[Z] = x, depth
            row[SNR] = np.clip(10*np.log10(near.echo_power/near.noise_power), 0, 40)
            row[VALID] = float(row[SNR] >= 10)
            levels = near.stereo_energy
            row[ILD] = (levels[1]-levels[0])/max(levels.sum(), 1e-10)
            if near.channels == 2 and near.iq is not None:
                at = int(np.argmin(abs(ranges-depth)))
                row[IPD] = np.angle(np.sum(near.iq[:, 1, at]*np.conj(near.iq[:, 0, at])))
            row[HAND_X], row[HAND_Z] = near.position or (x, depth)
            if near.position:
                # Range-gated path rates provide signed Doppler estimates.
                v = np.clip(near.velocity, -1.35, 1.35)
                balance = np.repeat([1-.5*row[HAND_X], 1+.5*row[HAND_X]], 2)
                row[DOPPLER] = np.tile([max(-v, 0), max(v, 0)], 2)*balance
        self.update(row, near.grid)
