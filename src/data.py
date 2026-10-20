import pandas as pd 
from sklearn.model_selection import train_test_split  

DATA_PATH = "data/bank-full.csv"
TARGET = "y"
LEAKY_COLUMNS = ["duration"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=LEAKY_COLUMNS)
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def split_data(df, test_size=0.2, seed=42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )