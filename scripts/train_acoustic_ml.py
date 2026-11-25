"""Fit and validate the acoustic model using independent simulated rooms."""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_acoustic_model import (DOPPLER, HAND_X, HAND_Z, ILD, IPD, LABELS,
                               MODEL_PATH, RANGE, SCHEMA, SNR, VALID, WINDOW, X, Z,
                               extract_acoustic_features)


def simulate_rooms(seed, repetitions):
    rng = np.random.default_rng(seed)
    centers = np.linspace(.175, .925, 16)
    times = np.arange(WINDOW)/25
    features, states, locations = [], [], []
    for label in range(len(LABELS)):
        for _ in range(repetitions):
            x, z = rng.uniform(-.85, .85), rng.uniform(.38, .84)
            hand_z = rng.uniform(.30, .64)
            hand_x = rng.uniform(-.35, .35)
            vx = rng.choice((.0, .05))*rng.choice((-1, 1))
            vz = rng.normal(0, .021)
            if label in (2, 3):
                vz = rng.uniform(.37, .85)*(-1 if label == 2 else 1)
                if label == 3:
                    hand_z = min(hand_z, .56)
            if label in (4, 5):
                vx = rng.uniform(1.15, 2.45)*(-1 if label == 4 else 1)
            room_noise = rng.uniform(.0004, .005)
            multipath = sum(rng.uniform(.001, .012)*np.exp(-((centers-r)/.038)**2)
                            for r in rng.uniform(.16, .94, 3))
            row = rng.normal(0, .008, (WINDOW, 28))
            for i, t in enumerate(times):
                hx, hz = np.clip(hand_x+vx*t, -.98, .98), np.clip(hand_z+vz*t, .16, .94)
                torso = z+.002*np.sin(2*np.pi*.3*t)
                profile = multipath+rng.exponential(room_noise, 16)
                if label:
                    profile = profile + rng.uniform(.04, .26)*np.exp(-((centers-torso)/.058)**2)
                    if label > 1:
                        profile += rng.uniform(.055, .26)*np.exp(-((centers-hz)/.046)**2)
                row[i, RANGE] = profile
                velocity = vz+rng.normal(0, .055)
                gains = np.clip([1-hx*.5, 1+hx*.5], .3, 1.7)
                row[i, DOPPLER] += np.resize([max(-velocity, 0), max(velocity, 0)], 4)*np.repeat(gains, 2)
                row[i, ILD] += x + rng.normal(0, .16) if label else 0
                row[i, IPD] += (2*np.pi*.04*x*14250/343 + rng.normal(0, .5)) if label else 0
                row[i, X] += x+rng.normal(0, .12) if label else 0
                row[i, Z] += torso+rng.normal(0, .027) if label else 0
                row[i, HAND_X] += hx+rng.normal(0, .038)
                row[i, HAND_Z] += hz+rng.normal(0, .021)
                row[i, SNR] = rng.uniform(13, 34) if label else rng.uniform(0, 14)
                row[i, VALID] = float(label > 0 and rng.random()>.08)
            features.append(extract_acoustic_features(row))
            states.append(label)
            locations.append((x, z))
    return np.asarray(features), np.asarray(states), np.asarray(locations)


def train(path=MODEL_PATH):
    features, classes, xyz = simulate_rooms(4661, 360)
    validation, labels, locations = simulate_rooms(4739, 120)
    scaler = StandardScaler().fit(features)
    x = scaler.transform(features)
    test = scaler.transform(validation)
    classifier = MLPClassifier(hidden_layer_sizes=(24,), max_iter=300, random_state=20,
                               early_stopping=True, n_iter_no_change=16).fit(x, classes)
    spatial = Ridge(alpha=18).fit(x[classes > 0], xyz[classes > 0])
    error = np.abs(spatial.predict(test[labels > 0])-locations[labels > 0]).mean(axis=0)
    metrics = dict(synthetic_validation_accuracy=float(classifier.score(test, labels)),
                   validation_windows=len(labels), x_mae=float(error[0]),
                   z_mae_cm=float(100*error[1]))
    if metrics["synthetic_validation_accuracy"] <= .92:
        raise RuntimeError(f"Synthetic validation target missed: {metrics}")
    model = dict(schema=SCHEMA, mean=scaler.mean_, scale=scaler.scale_,
                 layers=classifier.coefs_, biases=classifier.intercepts_,
                 spatial=spatial.coef_, offset=spatial.intercept_, metrics=metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    joblib.dump(model, pending)
    pending.replace(path)
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    _, result = train(parser.parse_args().model)
    print(json.dumps(result))
