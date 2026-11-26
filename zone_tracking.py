"""Learn coarse, room-specific position from short windows of measured echoes."""

from collections import deque
import json
import os
from pathlib import Path
from tempfile import mkstemp

import numpy as np

from nearfield import FRAME_RATE, RANGES

PROFILE_PATH = Path(__file__).with_name("models") / "zone_tracker.json"
SCHEMA = 1
LABELS = ("Empty", "Left", "Center", "Right")
TRIALS = (0, 1, 2, 3, 2, 0, 3, 1, 3, 2, 1, 0)
VALIDATION_START = 8
EXAMPLES_PER_TRIAL = 5
ROOM_FRAMES = 30
WINDOW = 10
MIN_PATHS = 2
CONFIDENCE = .55
SETTLE_FRAMES = 2*FRAME_RATE
ROOM_WARMUP = FRAME_RATE
# Group neighboring delay bins to reduce sensitivity to small timing shifts.
DELAY_BINS = np.flatnonzero((RANGES >= .225) & (RANGES < .925))
RANGE_GROUPS = 12
HALF_WINDOW_BINS = 5*RANGE_GROUPS//2
FEATURES = 2*4*RANGE_GROUPS


def load_profile(identity):
    try:
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        if data["schema"] != SCHEMA or data["identity"] != identity:
            return None
        shapes = {"mean": (FEATURES,), "scale": (FEATURES,),
                  "weights": (len(LABELS), FEATURES), "bias": (len(LABELS),)}
        for name, shape in shapes.items():
            values = np.asarray(data[name], dtype=float)
            if values.shape != shape or not np.isfinite(values).all():
                return None
            data[name] = values
        if np.any(data["scale"] <= 0):
            return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return data


def predict(profile, features):
    normalized = (features-profile["mean"])/profile["scale"]
    logits = normalized @ profile["weights"].T+profile["bias"]
    scores = np.exp(logits-logits.max(axis=-1, keepdims=True))
    probabilities = scores/scores.sum(axis=-1, keepdims=True)
    return probabilities.argmax(axis=-1), probabilities.max(axis=-1)


def fit_profile(samples, identity):
    from sklearn.linear_model import LogisticRegression

    train = np.concatenate(samples[:VALIDATION_START])
    test = np.concatenate(samples[VALIDATION_START:])
    labels = np.repeat(TRIALS, EXAMPLES_PER_TRIAL)
    split = len(train)
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), .025)
    model = LogisticRegression(C=.12).fit((train-mean)/scale, labels[:split])
    profile = dict(schema=SCHEMA, identity=identity, mean=mean, scale=scale,
                   weights=model.coef_, bias=model.intercept_)
    predictions, confidence = predict(profile, test)
    truth = labels[split:].reshape(-1, EXAMPLES_PER_TRIAL)[:, 1:]
    predictions = np.where(confidence >= CONFIDENCE, predictions, -1).reshape(-1, EXAMPLES_PER_TRIAL)
    correct = (predictions[:, 1:] == truth) & (predictions[:, 1:] == predictions[:, :-1])
    recalls = np.array([correct[truth == index].mean() for index in range(len(LABELS))])
    profile["validation_accuracy"] = float(correct.mean())
    profile["validation_recall"] = recalls.tolist()
    passed = bool(correct.mean() >= .75 and recalls.min() >= .60 and recalls[0] >= .80)
    return profile, passed


class ZoneTracker:
    def __init__(self, identity, learning=False, profile=None):
        self.identity = identity
        self.learning = learning
        self.profile = profile
        self.trials = []
        self.capture = None
        self.finished = False
        self.saved = False
        self.validation = None
        self.recall = None
        self.history = deque(maxlen=WINDOW)
        self.reset()

    def reset(self):
        self.room_samples = [[] for _ in range(4)]
        self.reference = np.zeros((4, len(DELAY_BINS)), dtype=complex)
        self.calibrated = np.zeros(4, dtype=bool)
        self.warmup = ROOM_WARMUP
        self.capture = None
        self.lost()

    def lost(self):
        self.history.clear()
        self.previous = None
        self.label = None
        self.confidence = None
        self.locked = False

    @property
    def ready(self):
        return np.count_nonzero(self.calibrated) >= MIN_PATHS

    def begin(self):
        if self.learning and not self.finished and self.ready and self.capture is None:
            self.history.clear()
            self.settle = SETTLE_FRAMES
            self.capture = []

    def update(self, iq, valid):
        paths = valid.ravel()
        if np.count_nonzero(paths) < MIN_PATHS:
            self.lost()
            return
        self.locked = True
        profile = iq.reshape(4, -1)[:, DELAY_BINS]
        if self.warmup:
            self.warmup -= 1
            return
        if not self.ready:
            for path in np.flatnonzero(paths & ~self.calibrated):
                self.room_samples[path].append(profile[path])
                if len(self.room_samples[path]) == ROOM_FRAMES:
                    self.reference[path] = np.mean(self.room_samples[path], axis=0)
                    self.calibrated[path] = True
                    self.room_samples[path].clear()
            return
        usable = paths & self.calibrated
        if np.count_nonzero(usable) < MIN_PATHS:
            self.lost()
            return
        if self.learning and self.capture is None:
            return
        if self.capture is not None and self.settle:
            self.settle -= 1
            return
        residual = np.where(usable[:, None], abs(profile-self.reference), 0)
        # Use a common delay origin so the model can learn differences between
        # paths across changing subject range.
        energy = np.convolve(residual.sum(axis=0), np.ones(5), mode="same")
        peak = int(energy.argmax())
        padded = np.pad(residual, ((0, 0), (HALF_WINDOW_BINS, HALF_WINDOW_BINS)))
        local = padded[:, peak:peak+2*HALF_WINDOW_BINS]
        pooled = local.reshape(4, RANGE_GROUPS, 5).mean(axis=-1)
        self.history.append(np.log1p(8*pooled).ravel())
        if len(self.history) < WINDOW:
            return
        window = np.asarray(self.history)
        features = np.r_[np.median(window, axis=0), np.percentile(window, 90, axis=0)]
        self.history.clear()
        if self.capture is not None:
            self.capture.append(features)
            if len(self.capture) == EXAMPLES_PER_TRIAL:
                self.trials.append(np.asarray(self.capture))
                self.capture = None
                if len(self.trials) == len(TRIALS):
                    self._finish()
        elif self.profile is not None:
            index, confidence = predict(self.profile, features)
            qualified = int(index) if confidence >= CONFIDENCE else None
            self.label = LABELS[qualified] if qualified is not None and qualified == self.previous else None
            self.confidence = float(confidence) if self.label is not None else None
            self.previous = qualified

    def _finish(self):
        profile, self.saved = fit_profile(self.trials, self.identity)
        self.validation = profile["validation_accuracy"]
        self.recall = profile["validation_recall"]
        if self.saved:
            PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in profile.items()}
            descriptor, name = mkstemp(dir=PROFILE_PATH.parent, prefix="zone_tracker-", suffix=".tmp")
            pending = Path(name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    json.dump(payload, file)
                pending.replace(PROFILE_PATH)
            finally:
                pending.unlink(missing_ok=True)
        self.finished = True

    def snapshot(self, active):
        if self.finished:
            status = (f"Calibration saved   /   validation {self.validation:.0%}" if self.saved
                      else f"Calibration did not qualify   /   validation {self.validation:.0%}")
            detail = "Restart in spatial mode to use the saved profile" if self.saved else "Esc to close"
        elif not active:
            status, detail = "Clear the area, then press M to calibrate", "Keep the laptop and lid in place"
        elif not self.ready:
            status, detail = "Keep the area clear for the room reference", "Waiting for stable audio timing"
        elif not self.locked:
            status, detail = "Waiting for probe timing", "Stay in position" if self.capture is not None else ""
        elif self.learning:
            zone = LABELS[TRIALS[len(self.trials)]]
            round_name = "Training" if len(self.trials) < VALIDATION_START else "Validation"
            detail = f"{round_name}   /   trial {len(self.trials)+1} of {len(TRIALS)}"
            if self.capture is None:
                status = ("Press C, then step clear" if zone == "Empty" else
                          f"Press C, then face the laptop from {zone.lower()}")
            elif self.settle:
                seconds = (self.settle+FRAME_RATE-1)//FRAME_RATE
                status = f"Get into position   /   {zone}   /   {seconds} s"
            else:
                status = ("Recording the clear area" if zone == "Empty" else
                          f"Recording {zone.lower()}   /   keep gently moving")
        else:
            status = ("Position tracked" if self.label in LABELS[1:] else
                      "Area clear   /   move into range" if self.label == LABELS[0] else "Position uncertain")
            detail = f"Model confidence {self.confidence:.0%}" if self.confidence is not None else ""
        return dict(status=status, detail=detail, learning=self.learning, finished=self.finished,
                    position=self.label if active else None, validation=self.validation,
                    recall=self.recall, saved=self.saved)
