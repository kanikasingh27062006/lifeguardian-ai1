"""
LifeGuardian AI — Model Training Script
========================================
Trains a Random Forest classifier on the synthetic activity dataset and saves
the complete preprocessing + model pipeline.

Uses a pure-Python/NumPy Random Forest (ml/pure_rf.py) to avoid scipy DLL
dependency issues on some Windows environments.

Usage
-----
    cd lifeguardian/backend
    python ml/train.py

Output
------
    models/pipeline.pkl          -- serialized pipeline object
    models/metrics.json          -- held-out test metrics
    models/label_classes.json    -- ordered label classes

DISCLAIMER
----------
This model is trained on SYNTHETIC demonstration data only. Strong accuracy
simply means the model learned the synthetic labelling rules, NOT that it
can detect real falls or health emergencies.
"""

import json
import os
import sys
import pickle
import csv
from collections import Counter

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
ML_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND    = os.path.dirname(ML_DIR)
ROOT       = os.path.dirname(BACKEND)
DATA_FILE  = os.path.join(ROOT, "data", "synthetic_activity_data.csv")
MODELS_DIR = os.path.join(ROOT, "models")
PIPELINE_FILE      = os.path.join(MODELS_DIR, "pipeline.pkl")
METRICS_FILE       = os.path.join(MODELS_DIR, "metrics.json")
LABEL_CLASSES_FILE = os.path.join(MODELS_DIR, "label_classes.json")

SEED = 42
LABEL_ORDER = ["low", "medium", "high"]


# ── pure metrics (no scipy) ────────────────────────────────────────────
def accuracy(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def confusion_mat(y_true, y_pred, labels):
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def per_class_metrics(y_true, y_pred, labels):
    results = {}
    for cls in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[cls] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   support,
        }
    return results


def weighted_avg(per_class, y_true, metric):
    total = len(y_true)
    return sum(
        per_class[cls][metric] * per_class[cls]["support"] / total
        for cls in per_class if total > 0
    )


# ── data loading ──────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def to_arrays(rows):
    X, y = [], []
    for r in rows:
        X.append([
            r["activity_state"],
            float(r["inactivity_duration_minutes"]),
            float(r["movement_intensity"]),
            float(r["hour_of_day"]),
        ])
        y.append(r["risk_category"])
    return X, y


# ── train/test split ───────────────────────────────────────────────────
def stratified_split(X, y, test_frac=0.2, seed=SEED):
    """Simple stratified split without sklearn."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for i, label in enumerate(y):
        by_class.setdefault(label, []).append(i)

    train_idx, test_idx = [], []
    for label, indices in by_class.items():
        arr = np.array(indices)
        rng.shuffle(arr)
        n_test = max(1, int(len(arr) * test_frac))
        test_idx.extend(arr[:n_test].tolist())
        train_idx.extend(arr[n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_arr = np.array(X, dtype=object)
    y_arr = np.array(y)
    return (
        X_arr[train_idx], X_arr[test_idx],
        y_arr[train_idx].tolist(), y_arr[test_idx].tolist(),
    )


# Import shared classes from pipeline.py (pickle needs them importable by module path)
from pipeline import SimpleEncoder, ActivityPipeline


# ── main training function ─────────────────────────────────────────────
def train(data_path=DATA_FILE):
    # import pure RF here so the path is resolved from ml/ dir
    sys.path.insert(0, ML_DIR)
    from pure_rf import RandomForestClassifier

    print(f"Loading data: {data_path}")
    rows = load_csv(data_path)
    print(f"  {len(rows)} records")

    X_raw, y = to_arrays(rows)

    # Split BEFORE fitting (no leakage)
    X_train, X_test, y_train, y_test = stratified_split(X_raw, y, test_frac=0.2, seed=SEED)
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Train class counts: {Counter(y_train)}")

    enc = SimpleEncoder()
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=3,
        random_state=SEED,
    )

    pipeline = ActivityPipeline(enc, clf)
    print("Training Random Forest (pure-Python, no scipy)...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc  = accuracy(y_test, y_pred)
    present = [l for l in LABEL_ORDER if l in set(y_test)]
    pcm  = per_class_metrics(y_test, y_pred, present)
    prec = weighted_avg(pcm, y_test, "precision")
    rec  = weighted_avg(pcm, y_test, "recall")
    f1   = weighted_avg(pcm, y_test, "f1")
    cm   = confusion_mat(y_test, y_pred, present)

    metrics = {
        "accuracy":           round(acc,  4),
        "precision_weighted": round(prec, 4),
        "recall_weighted":    round(rec,  4),
        "f1_weighted":        round(f1,   4),
        "confusion_matrix":   cm,
        "confusion_labels":   present,
        "per_class":          pcm,
        "train_samples":      len(X_train),
        "test_samples":       len(X_test),
        "n_estimators":       clf.n_estimators,
        "max_depth":          clf.max_depth,
        "disclaimer": (
            "Metrics on a synthetic held-out test set. "
            "High accuracy indicates the model learned the synthetic labelling rules. "
            "This does NOT prove real-world fall detection ability or medical reliability."
        ),
    }

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print("\n  Per-class:")
    for cls in present:
        p = pcm[cls]
        print(f"    {cls:8s} P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f} n={p['support']}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(PIPELINE_FILE, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nPipeline saved: {PIPELINE_FILE}")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved:  {METRICS_FILE}")

    with open(LABEL_CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump(clf.classes_.tolist(), f)
    print(f"Classes saved:  {LABEL_CLASSES_FILE}")

    return metrics


if __name__ == "__main__":
    train()
