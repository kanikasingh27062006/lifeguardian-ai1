"""
LifeGuardian AI — Synthetic Dataset Generator
==============================================
Generates ~3 000 reproducible synthetic activity records for training the
risk-classification model.

IMPORTANT DISCLAIMER
--------------------
These labels are DEMONSTRATION labels only. They are assigned by explicit
rules so the model can learn something non-trivial, NOT by medical experts
or validated clinical criteria. The resulting model must never be used for
actual health decisions.

Labelling rules (documented for transparency):
  LOW risk
    - activity_state in (walking, sitting)
    - inactivity_duration_minutes < 20
    - movement_intensity >= 30

  MEDIUM risk  (one or more of the following with no HIGH trigger)
    - activity_state == inactive AND inactivity_duration_minutes in [20, 60)
    - activity_state == lying_down AND inactivity_duration_minutes in [10, 30)
    - movement_intensity in [10, 30) AND inactivity_duration_minutes >= 15
    - activity_state == inactive AND movement_intensity < 15

  HIGH risk  (any of the following)
    - activity_state == inactive AND inactivity_duration_minutes >= 60
    - activity_state == lying_down AND inactivity_duration_minutes >= 30 AND movement_intensity < 20
    - movement_intensity < 10 AND inactivity_duration_minutes >= 30
    - activity_state in (inactive, lying_down) AND movement_intensity < 5

Note: noise and borderline randomness are added to avoid a perfectly learnable
dataset (which would give unrealistically high accuracy numbers).

Target-leakage avoidance
------------------------
- No scenario identifier is included in features.
- No column is derived directly from the label.
- `hour_of_day` is included as a real-world proxy: elderly people are more
  likely to be inactive/resting at night, so it carries legitimate signal.

Usage
-----
    python generate_data.py          # saves data/synthetic_activity_data.csv
    python generate_data.py --rows 5000
"""

import argparse
import csv
import os
import random

SEED = 42
random.seed(SEED)

ACTIVITY_STATES = ["walking", "sitting", "lying_down", "inactive"]
TARGET_ROWS = 3000
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT_FILE = os.path.join(OUT_DIR, "synthetic_activity_data.csv")

FIELDNAMES = [
    "activity_state",
    "inactivity_duration_minutes",
    "movement_intensity",
    "hour_of_day",          # 0–23, legitimate contextual feature
    "risk_category",        # target: low | medium | high
]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _gauss(mu: float, sigma: float, lo: float, hi: float) -> float:
    return _clamp(random.gauss(mu, sigma), lo, hi)


def _label(activity: str, duration: float, intensity: float) -> str:
    """Assign a risk category using the documented rules."""

    # HIGH triggers — any one is sufficient
    if activity == "inactive" and duration >= 60:
        return "high"
    if activity == "lying_down" and duration >= 30 and intensity < 20:
        return "high"
    if intensity < 10 and duration >= 30:
        return "high"
    if activity in ("inactive", "lying_down") and intensity < 5:
        return "high"

    # MEDIUM triggers
    if activity == "inactive" and 20 <= duration < 60:
        return "medium"
    if activity == "lying_down" and 10 <= duration < 30:
        return "medium"
    if 10 <= intensity < 30 and duration >= 15:
        return "medium"
    if activity == "inactive" and intensity < 15:
        return "medium"

    return "low"


def _add_noise(label: str, p: float = 0.04) -> str:
    """Flip label to a neighbour with probability p — simulates labelling noise."""
    if random.random() >= p:
        return label
    neighbors = {"low": "medium", "medium": random.choice(["low", "high"]), "high": "medium"}
    return neighbors[label]


def generate(n_rows: int = TARGET_ROWS) -> list[dict]:
    """Return a list of synthetic record dicts."""
    # Desired class balance: roughly 50% low, 30% medium, 20% high
    profiles = [
        # (activity, duration_mu, duration_sigma, intensity_mu, intensity_sigma, weight)
        ("walking",    3,   3,  65, 15, 0.18),
        ("sitting",    8,   6,  45, 15, 0.18),
        ("sitting",    12,  8,  35, 12, 0.10),
        ("lying_down", 5,   4,  55, 20, 0.06),
        ("lying_down", 20, 12,  30, 15, 0.06),
        ("lying_down", 40, 15,  12,  8, 0.06),
        ("inactive",   5,   4,  40, 18, 0.06),
        ("inactive",   30, 15,  20, 12, 0.10),
        ("inactive",   70, 20,  10,  8, 0.12),
        ("inactive",   90, 25,   4,  4, 0.08),
    ]
    weights = [p[5] for p in profiles]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    records = []
    for _ in range(n_rows):
        # Pick a profile
        idx = random.choices(range(len(profiles)), weights=weights, k=1)[0]
        activity, d_mu, d_sig, i_mu, i_sig, _ = profiles[idx]

        duration  = round(_gauss(d_mu, d_sig, 0.0, 120.0), 1)
        intensity = round(_gauss(i_mu, i_sig, 0.0, 100.0), 1)
        hour      = random.randint(0, 23)

        label = _label(activity, duration, intensity)
        label = _add_noise(label)

        records.append({
            "activity_state":             activity,
            "inactivity_duration_minutes": duration,
            "movement_intensity":          intensity,
            "hour_of_day":                 hour,
            "risk_category":               label,
        })

    return records


def save(records: list[dict], path: str = OUT_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records to {path}")


def validate(records: list[dict]) -> dict:
    """Check basic data integrity and print a class-distribution summary."""
    counts = {"low": 0, "medium": 0, "high": 0}
    for r in records:
        counts[r["risk_category"]] += 1
    total = len(records)
    print("\nClass distribution:")
    for cls, cnt in counts.items():
        print(f"  {cls:8s}: {cnt:5d}  ({cnt/total*100:.1f}%)")
    assert total > 0, "No records generated"
    assert all(r["activity_state"] in ACTIVITY_STATES for r in records)
    assert all(0 <= r["inactivity_duration_minutes"] <= 120 for r in records)
    assert all(0 <= r["movement_intensity"] <= 100 for r in records)
    print("Validation passed.")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=TARGET_ROWS)
    parser.add_argument("--out",  type=str, default=OUT_FILE)
    args = parser.parse_args()

    records = generate(args.rows)
    validate(records)
    save(records, args.out)
