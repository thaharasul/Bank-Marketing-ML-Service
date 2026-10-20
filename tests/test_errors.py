import numpy as np  # build small arrays
import pandas as pd  # build small tables

from src.errors import error_table, segment_miss_rate  # the functions under test


def test_error_table_labels():  # each customer must get the right label
    X = pd.DataFrame({"month": ["a", "a", "b", "b"]})  # four customers
    y = pd.Series([1, 1, 0, 0])  # two buyers, two non-buyers
    proba = np.array([0.9, 0.1, 0.8, 0.1])  # model scores
    t = error_table(X, y, proba, 0.5)  # cutoff 0.5
    assert t["error_type"].tolist() == ["correct", "missed_buyer", "false_alarm", "correct"]  # expected labels


def test_segment_miss_rate():  # miss rate per group must be computed correctly
    X = pd.DataFrame({"month": ["a", "a", "b", "b"]})  # four customers
    y = pd.Series([1, 1, 1, 0])  # three buyers
    proba = np.array([0.9, 0.1, 0.9, 0.1])  # model finds one of two buyers in group a
    t = error_table(X, y, proba, 0.5)  # label them
    r = segment_miss_rate(t, "month", min_buyers=1)  # allow tiny groups for this test
    assert r.loc["a", "miss_rate"] == 0.5  # group a: 1 of 2 buyers missed
    assert r.loc["b", "miss_rate"] == 0.0  # group b: its only buyer was found