import typing as t
import pandas as pd

from wine_quality_model import __version__ as _version
from wine_quality_model.config.core import config
from wine_quality_model.processing.data_manager import load_pipeline
from wine_quality_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(input_data: t.Union[pd.DataFrame, dict]) -> dict:
    #data = pd.DataFrame(input_data)
    if isinstance(input_data, dict):
        data = pd.DataFrame(input_data, index=[0])  # Указываем индекс [0]
    else:
        data = pd.DataFrame(input_data)
    print("Input data:", data)
    validated_data, errors = validate_inputs(input_data=data)
    print("Validation errors:", errors)

    results = {
        "predictions": None,
        "version": _version,
        "errors": errors
    }

    if not errors:
        preds = _pipe.predict(validated_data)
        results["predictions"] = [round(float(pred), 2) for pred in preds]

    return results

#def get_user_input() -> t.Dict[str, float]:
    #"""Запрашивает у пользователя параметры вина."""
    #print("Введите параметры вина:")
    #input_data = {
        #"fixed acidity": float(input("Fixed Acidity (7.0-15.0): ")),
        #"volatile acidity": float(input("Volatile Acidity (0.1-1.0): ")),
        #"citric acid": float(input("Citric Acid (0.0-1.0): ")),
        #"residual sugar": float(input("Residual Sugar (0.5-15.0): ")),
        #"chlorides": float(input("Chlorides (0.01-0.2): ")),
        #"free sulfur dioxide": float(input("Free Sulfur Dioxide (1-100): ")),
        #"total sulfur dioxide": float(input("Total Sulfur Dioxide (5-200): ")),
        #"density": float(input("Density (0.98-1.04): ")),
        #"pH": float(input("pH (2.5-4.0): ")),
        #"sulphates": float(input("Sulphates (0.3-2.0): ")),
        #"alcohol": float(input("Alcohol (% vol, 8-15): "))
   #}
    #return input_data

#if __name__ == "__main__":
    #user_data = get_user_input()  # Получаем ввод
    #prediction = make_prediction(input_data=user_data)  # Делаем предсказание
    #print(f"Предсказанное качество вина: {prediction['predictions'][0]:.1f}/10")