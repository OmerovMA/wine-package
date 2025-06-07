import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError
from typing import Optional, Tuple

from wine_quality_model.config.core import config

class WineInputSchema(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    validated_data = input_data.copy()
    validated_data.columns = [col.replace(" ", "_") for col in validated_data.columns]
    errors = None

    try:
        WineInputSchema(**validated_data.replace({np.nan: None}).to_dict(orient="records")[0])
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors