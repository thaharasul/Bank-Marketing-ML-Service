import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.compose import ColumnTransformer 
from sklearn.dummy import DummyClassifier  
from sklearn.ensemble import HistGradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (  
    ConfusionMatrixDisplay, 
    f1_score, 
    precision_score,  
    recall_score,  
    roc_auc_score, 
)
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import OneHotEncoder, StandardScaler  

from src.data import load_data, split_data  


def build_preprocessor(X):  
    numeric = X.select_dtypes(include="number").columns.tolist()   
    categorical = X.select_dtypes(exclude="number").columns.tolist()  
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def evaluate(name, model, Xte, yte):
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    return {
        "model": name,
        "macro_f1": f1_score(yte, pred, average="macro"),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred),
        "roc_auc": roc_auc_score(yte, proba),
    }


def main():
    df = load_data()
    Xtr, Xte, ytr, yte = split_data(df)
    print("Positive rate (train):", round(ytr.mean(), 3))

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=42),
    }

    rows, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("model", clf)])
        pipe.fit(Xtr, ytr)
        rows.append(evaluate(name, pipe, Xte, yte))
        fitted[name] = pipe

    results = pd.DataFrame(rows).round(4)
    print(results.to_string(index=False))
    results.to_csv("results/metrics.csv", index=False)

    best = results.sort_values("macro_f1").iloc[-1]["model"]
    ConfusionMatrixDisplay.from_predictions(yte, fitted[best].predict(Xte))
    plt.title(f"Confusion matrix: {best}")
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Best by macro-F1:", best)


if __name__ == "__main__":
    main()