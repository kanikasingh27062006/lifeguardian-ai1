"""
LifeGuardian AI — Shared ML Pipeline Classes
=============================================
ActivityPipeline and SimpleEncoder must be defined here (not inside train.py)
so that pickle can find them when loading the saved model.
"""

import numpy as np


class SimpleEncoder:
    """Encode activity_state column to integer, pass numerics through."""
    ACTIVITY_MAP = {"walking": 0.0, "sitting": 1.0, "lying_down": 2.0, "inactive": 3.0}

    def fit(self, X):
        return self

    def transform(self, X):
        out = []
        for row in X:
            act = self.ACTIVITY_MAP.get(str(row[0]), -1.0)
            out.append([act, float(row[1]), float(row[2]), float(row[3])])
        return np.array(out, dtype=float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ActivityPipeline:
    """Wraps encoder + RandomForest. Pickle-friendly (no closures)."""

    def __init__(self, encoder, model):
        self.encoder  = encoder
        self.model    = model
        self.classes_ = None

    def fit(self, X, y):
        Xt = self.encoder.fit_transform(X)
        self.model.fit(Xt, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        Xt = self.encoder.transform(X)
        return self.model.predict(Xt)

    def predict_proba(self, X):
        Xt = self.encoder.transform(X)
        return self.model.predict_proba(Xt)
