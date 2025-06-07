import pandas as pd
from wine_quality_model.processing.data_manager import load_dataset

def test_data_loading():
    """Проверка загрузки данных"""
    data = load_dataset(file_name="winequality-white.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_feature_ranges(raw_data):
    """Проверка диапазонов значений"""
    assert (raw_data["alcohol"] > 7.0).all()
    assert (raw_data["alcohol"] < 15.0).all()
    assert (raw_data["pH"] > 2.5).all()
    assert (raw_data["pH"] < 4.0).all()

def test_missing_values(raw_data):
    """Проверка на отсутствие пропусков"""
    assert not raw_data[raw_data.isnull().any(axis=1)].empty