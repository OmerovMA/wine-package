from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import MeanMedianImputer

from wine_quality_model.config.core import config

wine_pipe = Pipeline([
    # Импутация пропусков (если есть)
    ("imputer", MeanMedianImputer(
        imputation_method="median",
        variables=config.model_config_params.numerical_vars
    )),
    # Скейлинг
    ("scaler", StandardScaler()),
    # Модель регрессии
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=config.model_config_params.random_state
    ))
])