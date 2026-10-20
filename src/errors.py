import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.analyze import best_threshold
from src.data import load_data, split_data
from src.train import build_preprocessor


def error_table(X, y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    out = X.copy()
    out["actual"] = y.values
    out["predicted"] = pred
    out["error_type"] = "correct"
    out.loc[(out["actual"] == 1) & (out["predicted"] == 0), "error_type"] = "missed_buyer"
    out.loc[(out["actual"] == 0) & (out["predicted"] == 1), "error_type"] = "false_alarm"
    return out


def segment_miss_rate(table, column, min_buyers=30):
    buyers = table[table["actual"] == 1]
    g = buyers.groupby(column)["predicted"].agg(buyers="count", found="sum")
    g["miss_rate"] = 1 - g["found"] / g["buyers"]
    g = g[g["buyers"] >= min_buyers]
    return g.sort_values("miss_rate", ascending=False).round(3)


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
    thr, _ = best_threshold(yval, proba_val)
    print(f"Threshold used: {thr:.3f}")

    imp = permutation_importance(
        pipe, Xval, yval, scoring="roc_auc", n_repeats=3, random_state=42
    )
    importance = pd.DataFrame({
        "feature": Xval.columns,
        "roc_auc_drop": imp.importances_mean,
    }).sort_values("roc_auc_drop", ascending=False).round(4)
    print(importance.head(10).to_string(index=False))
    importance.to_csv("results/feature_importance.csv", index=False)

    top = importance.head(10).iloc[::-1]
    plt.figure(figsize=(7, 4))
    plt.barh(top["feature"], top["roc_auc_drop"])
    plt.xlabel("ROC-AUC drop when column is shuffled")
    plt.title("Feature importance (validation set)")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)

    proba_te = pipe.predict_proba(Xte)[:, 1]
    table = error_table(Xte, yte, proba_te, thr)
    print(table["error_type"].value_counts().to_string())
    for col in ["month", "job"]:
        print(f"\nMissed-buyer rate by {col}:")
        print(segment_miss_rate(table, col).head(5).to_string())
    table[table["error_type"] != "correct"].head(30).to_csv("results/error_samples.csv", index=False)
    print("\nSaved results/feature_importance.png and results/error_samples.csv")


if __name__ == "__main__":
    main()