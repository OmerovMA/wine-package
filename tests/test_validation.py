import pytest
import numpy as np
import pandas as pd
from wine_quality_model.processing.validation import validate_inputs


def test_validation_with_valid_data(sample_input):
    """Проверка валидации корректных данных"""
    input_df = pd.DataFrame(sample_input, index=[0])
    validated_data, errors = validate_inputs(input_data=input_df)

    assert errors is None
    assert not validated_data.empty


def test_validation_with_missing_values(sample_input):
    """Проверка обработки пропущенных значений"""
    invalid_input = sample_input.copy()
    invalid_input["fixed_acidity"] = np.nan

    input_df = pd.DataFrame(invalid_input, index=[0])
    _, errors = validate_inputs(input_data=input_df)

    assert errors is not None
    assert "fixed_acidity" in errors


def test_validation_with_invalid_types(sample_input):
    """Проверка обработки неверных типов данных"""
    invalid_input = sample_input.copy()
    invalid_input["alcohol"] = "not_a_number"

    input_df = pd.DataFrame(invalid_input, index=[0])
    _, errors = validate_inputs(input_data=input_df)

    assert errors is not None
    assert "alcohol" in errors