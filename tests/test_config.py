from wine_quality_model.config.core import config

def test_config_loaded():
    """Проверка загрузки конфигурации"""
    assert hasattr(config, "app_config")
    assert hasattr(config, "model_config")

def test_feature_list():
    """Проверка списка фичей"""
    assert len(config.model_config_params.features) > 0
    assert "alcohol" in config.model_config_params.features

def test_target_variable():
    """Проверка целевой переменной"""
    assert config.model_config_params.target == "quality"