import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import HistGradientBoostingClassifier 
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score 
from sklearn.model_selection import train_test_split  
from sklearn.pipeline import Pipeline 

from src.data import load_data, split_data 
from src.train import build_preprocessor 

def best_threshold(y_true, proba):  
    precision, recall, thresholds = precision_recall_curve(y_true, proba) 
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12) 
    i = int(np.argmax(f1))  
    return float(thresholds[i]), float(f1[i])  


def score_at(y_true, proba, threshold): 
    pred = (proba >= threshold).astype(int) 
    return {
        "threshold": round(threshold, 3),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1_yes": f1_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
    }


def main():
    df = load_data()
    Xtr, Xte, ytr, yte = split_data(df)
    Xfit, Xval, yfit, yval = train_test_split(
        Xtr, ytr, test_size=0.25, random_state=42, stratify=ytr
    )

    pipe = Pipeline([
        ("prep", build_preprocessor(Xfit)),
        ("model", HistGradientBoostingClassifier(random_state=42)),
    ])
    pipe.fit(Xfit, yfit)

    proba_val = pipe.predict_proba(Xval)[:, 1]
    thr, f1_val = best_threshold(yval, proba_val)
    print(f"Chosen threshold: {thr:.3f} (validation F1 for yes: {f1_val:.3f})")

    proba_te = pipe.predict_proba(Xte)[:, 1]
    results = pd.DataFrame([
        {"setting": "default_0.5", **score_at(yte, proba_te, 0.5)},
        {"setting": "tuned", **score_at(yte, proba_te, thr)},
    ]).round(4)
    print(results.to_string(index=False))
    results.to_csv("results/threshold_results.csv", index=False)

    precision, recall, thresholds = precision_recall_curve(yval, proba_val)
    plt.figure()
    plt.plot(thresholds, precision[:-1], label="precision")
    plt.plot(thresholds, recall[:-1], label="recall")
    plt.axvline(thr, linestyle="--", color="gray", label="chosen threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision vs recall (validation set)")
    plt.legend()
    plt.savefig("results/threshold_curve.png", dpi=150)
    print("Saved results/threshold_curve.png")


if __name__ == "__main__":
    main()