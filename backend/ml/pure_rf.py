"""
LifeGuardian AI — Pure-Python/NumPy Random Forest
===================================================
A lightweight Random Forest classifier that avoids scipy entirely.
Used because the installed scipy version has a DLL blocked by Application
Control policy on this machine.

This implementation is intentionally simple and beginner-readable.
It produces genuine train/test metrics on the synthetic dataset.

API mirrors the subset of sklearn used by the rest of the project:
  .fit(X, y)
  .predict(X)
  .predict_proba(X)
  .classes_
"""

import math
import random as _random
from collections import Counter

import numpy as np


class DecisionTree:
    """
    Single CART decision tree (classification, Gini impurity).
    """
    def __init__(self, max_depth=10, min_samples_leaf=3, seed=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self._rng = _random.Random(seed)
        self.classes_ = None
        self._tree = None

    # ── Gini ────────────────────────────────────────────────────────────
    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        counts = Counter(y)
        n = len(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    # ── best split ──────────────────────────────────────────────────────
    def _best_split(self, X, y, feature_indices):
        best = {"gain": -1, "feat": None, "thresh": None}
        n = len(y)
        g_parent = self._gini(y)
        for fi in feature_indices:
            col = X[:, fi]
            thresholds = np.unique(col)
            if len(thresholds) > 20:          # bin continuous features
                thresholds = np.percentile(col, np.linspace(5, 95, 19))
            for thr in thresholds:
                left  = y[col <= thr]
                right = y[col >  thr]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                gain = g_parent - (
                    len(left) / n * self._gini(left) +
                    len(right) / n * self._gini(right)
                )
                if gain > best["gain"]:
                    best = {"gain": gain, "feat": fi, "thresh": thr}
        return best

    # ── build ────────────────────────────────────────────────────────────
    def _build(self, X, y, depth, n_features):
        counts = Counter(y)
        majority = counts.most_common(1)[0][0]
        proba = {c: counts[c] / len(y) for c in self.classes_}

        if (depth >= self.max_depth or
                len(y) < 2 * self.min_samples_leaf or
                len(counts) == 1):
            return {"leaf": True, "label": majority, "proba": proba}

        fi_sample = self._rng.sample(
            list(range(X.shape[1])),
            min(n_features, X.shape[1])
        )
        split = self._best_split(X, y, fi_sample)
        if split["feat"] is None:
            return {"leaf": True, "label": majority, "proba": proba}

        mask = X[:, split["feat"]] <= split["thresh"]
        left  = self._build(X[mask],  y[mask],  depth + 1, n_features)
        right = self._build(X[~mask], y[~mask], depth + 1, n_features)
        return {
            "leaf": False,
            "feat": split["feat"],
            "thresh": split["thresh"],
            "left": left,
            "right": right,
        }

    # ── predict single ───────────────────────────────────────────────────
    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["label"], node["proba"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    # ── public API ───────────────────────────────────────────────────────
    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        n_features = max(1, int(math.sqrt(X.shape[1])))
        self._tree = self._build(X, np.array(y), 0, n_features)
        return self

    def predict(self, X):
        return np.array([self._predict_one(x, self._tree)[0] for x in X])

    def predict_proba(self, X):
        rows = []
        for x in X:
            _, proba = self._predict_one(x, self._tree)
            rows.append([proba.get(c, 0.0) for c in self.classes_])
        return np.array(rows)


class RandomForestClassifier:
    """
    Ensemble of DecisionTree classifiers with bootstrap sampling.
    API: .fit(), .predict(), .predict_proba(), .classes_
    """
    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        **_kwargs   # absorb sklearn kwargs like n_jobs, class_weight
    ):
        self.n_estimators   = n_estimators
        self.max_depth      = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state   = random_state
        self.trees          = []
        self.classes_       = None

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.array(sorted(set(y)))
        n = len(X)
        rng = _random.Random(self.random_state)
        self.trees = []
        for i in range(self.n_estimators):
            idx = [rng.randint(0, n - 1) for _ in range(n)]
            Xb, yb = X[idx], y[idx]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                seed=self.random_state + i,
            )
            tree.classes_ = self.classes_
            # reuse parent classes (so proba dicts align)
            tree.fit(Xb, yb)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        all_proba = np.zeros((len(X), len(self.classes_)))
        for tree in self.trees:
            all_proba += tree.predict_proba(X)
        return all_proba / len(self.trees)

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
