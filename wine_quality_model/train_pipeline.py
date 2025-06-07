import logging
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from wine_quality_model.config.core import LOG_DIR, config
from wine_quality_model.pipeline import wine_pipe
from wine_quality_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    # Загрузка данных
    data = load_dataset(file_name=config.app_config.training_data_file)
    X = data[config.model_config_params.features]
    y = data[config.model_config_params.target]

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.model_config_params.test_size,
        random_state=config.model_config_params.random_state
    )

    # Обучение
    wine_pipe.fit(X_train, y_train)

    # Оценка
    y_pred = wine_pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Test MSE: {mse:.2f}")
    logging.info(f"Test R2: {r2:.2f}")

    # Сохранение модели
    save_pipeline(pipeline_to_persist=wine_pipe)

if __name__ == "__main__":
    run_training()