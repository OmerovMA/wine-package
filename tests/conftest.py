import pytest
import pandas as pd
from wine_quality_model.processing.data_manager import load_dataset

@pytest.fixture
def raw_data():
    """Загружает исходные данные"""
    return load_dataset(file_name="winequality-white.csv")

@pytest.fixture
def sample_input():
    """Корректные входные данные для предсказания"""
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }