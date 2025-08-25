
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_dataset():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    target_names = list(data.target_names)
    return X, y, target_names

def induce_imbalance(X, y, ratio, random_state=42):
    # Downsample the minority class in the TRAIN SPLIT ONLY.
    # ratio=1.0 keeps the class distribution; ratio=0.3 keeps 30% of minority samples.
    if ratio >= 1.0:
        return X, y
    counts = y.value_counts()
    minority = counts.idxmin()
    minority_idx = y[y == minority].index
    keep_n = max(1, int(len(minority_idx) * ratio))
    rng = np.random.default_rng(random_state)
    keep_idx = set(rng.choice(minority_idx, size=keep_n, replace=False))
    mask = y.index.map(lambda i: (y[i] != minority) or (i in keep_idx))
    return X.loc[mask], y.loc[mask]
