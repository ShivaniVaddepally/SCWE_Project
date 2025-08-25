
import numpy as np
from dataclasses import dataclass
from typing import List
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV

@dataclass
class SCWEConfig:
    calibration_method: str = "isotonic"

class SCWE(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learners, config: SCWEConfig):
        self.base_learners = base_learners
        self.config = config
        self.models_ = []
        self.weights_ = None

    def fit(self, X_train, y_train, X_val, y_val, scorer):
        self.models_ = []
        aucs = []
        for est in self.base_learners:
            model = CalibratedClassifierCV(clone(est), method=self.config.calibration_method, cv=3)
            model.fit(X_train, y_train)
            self.models_.append(model)
            aucs.append(scorer(model, X_val, y_val))
        aucs = np.array(aucs)
        eps = 1e-6
        w = aucs + eps
        self.weights_ = w / w.sum()
        return self

    def predict_proba(self, X):
        probs = [m.predict_proba(X)[:, 1] for m in self.models_]
        P = np.vstack(probs)
        blended = (self.weights_.reshape(-1,1) * P).sum(axis=0)
        return np.vstack([1 - blended, blended]).T

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)[:, 1]
        return (p >= threshold).astype(int)
