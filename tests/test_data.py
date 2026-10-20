import pandas as pd  
from src.data import load_data, split_data      
def test_load_drops_duration_and_encodes_target(tmp_path):         
    csv = tmp_path / "mini.csv"     
    csv.write_text("age;job;duration;y\n30;admin;100;yes\n40;tech;200;no\n")    
    df = load_data(str(csv))    
    assert "duration" not in df.columns     
    assert df["y"].tolist() == [1, 0]   


def test_split_is_stratified_and_disjoint():    
    df = pd.DataFrame({"x": range(100), "y": [1] * 10 + [0] * 90})  
    Xtr, Xte, ytr, yte = split_data(df)     
    assert abs(ytr.mean() - yte.mean()) < 0.02  
    assert set(Xtr.index).isdisjoint(set(Xte.index))    