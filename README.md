# Bank Marketing: Leakage-Safe ML Benchmark

Predicts whether a customer subscribes to a term deposit (UCI Bank Marketing, CC BY 4.0).

## Key decisions
- `duration` is dropped: it is only known after the call (data leakage).
- Stratified 80/20 split BEFORE any preprocessing.
- Baseline first, then two models, judged on macro-F1 and ROC-AUC (not accuracy: 88% of customers say no).

## Run
    pip install -r requirements.txt
    python -m pytest
    python -m src.train

Download `bank-full.csv` from the UCI link above into `data/`.

## Results (held-out test set)
| Model | Macro-F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Majority baseline | 0.4689 | 0.0000 | 0.0000 | 0.5000 |
| Logistic regression (balanced) | 0.6104 | 0.2662 | 0.6238 | 0.7722 |
| Gradient boosting | 0.6452 | 0.6469 | 0.2372 | 0.8026 |

Gradient boosting leads on macro-F1 and ROC-AUC. The models trade precision against recall differently; threshold tuning is next.

## Limitations
- One train/test split, no hyperparameter tuning.
- Default 0.5 threshold, not tuned to a business cost.
- Single dataset from one Portuguese bank.

## Next
M3: error analysis and threshold tuning.