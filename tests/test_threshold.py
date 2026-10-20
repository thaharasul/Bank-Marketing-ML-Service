import numpy as np 
from src.analyze import best_threshold  
def test_best_threshold_on_separable_data():
    y = np.array([0, 0, 1, 1]) 
    proba = np.array([0.1, 0.2, 0.8, 0.9]) 
    thr, f1 = best_threshold(y, proba) 
    assert abs(f1 - 1.0) < 1e-6 
    assert 0.2 < thr <= 0.8 