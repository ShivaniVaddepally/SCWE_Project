
import json, os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from data_loader import load_dataset, induce_imbalance
from evaluation import plot_roc, plot_pr, plot_calibration, plot_confusion, threshold_by_f1
from scwe import SCWE, SCWEConfig

THIS_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(THIS_DIR, "config.json"), "r") as f:
    CFG = json.load(f)

RANDOM_STATE = CFG["random_state"]
N_SPLITS = CFG["n_splits"]
VAL_SIZE = CFG["validation_size"]
IMB_RATIO = CFG["imbalance_ratio"]
CAL_METHOD = CFG["calibration_method"]

def scorer_auc(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)

def run():
    X, y, target_names = load_dataset()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    fold_idx = 0

    for train_index, test_index in skf.split(X, y):
        fold_idx += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )

        X_tr, y_tr = induce_imbalance(X_tr, y_tr, IMB_RATIO, random_state=RANDOM_STATE)

        lr = Pipeline([("scaler", StandardScaler()), 
                       ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_STATE))])
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced_subsample")
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

        base_learners = [lr, rf, gb]
        scwe = SCWE(base_learners, SCWEConfig(calibration_method=CAL_METHOD))
        scwe.fit(X_tr, y_tr, X_val, y_val, scorer=scorer_auc)

        val_probs = scwe.predict_proba(X_val)[:, 1]
        thr, best_f1 = threshold_by_f1(y_val, val_probs)

        test_probs = scwe.predict_proba(X_test)[:, 1]
        test_pred = (test_probs >= thr).astype(int)

        metrics = {
            "fold": fold_idx,
            "threshold": thr,
            "roc_auc": roc_auc_score(y_test, test_probs),
            "pr_auc": average_precision_score(y_test, test_probs),
            "f1": f1_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred),
            "recall": recall_score(y_test, test_pred),
            "accuracy": accuracy_score(y_test, test_pred),
        }
        rows.append(metrics)

        plot_roc(y_test, test_probs, os.path.join(OUT_DIR, f"fold{fold_idx}_roc.png"))
        plot_pr(y_test, test_probs, os.path.join(OUT_DIR, f"fold{fold_idx}_pr.png"))
        plot_calibration(y_test, test_probs, os.path.join(OUT_DIR, f"fold{fold_idx}_calibration.png"))
        plot_confusion(y_test, test_pred, os.path.join(OUT_DIR, f"fold{fold_idx}_confusion.png"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "scwe_kfold_results.csv"), index=False)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(df.describe().to_string())

    print("=== SCWE Evaluation Complete ===")
    print(df)

if __name__ == "__main__":
    run()
