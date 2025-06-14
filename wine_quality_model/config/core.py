from pathlib import Path
from typing import List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import wine_quality_model

PACKAGE_ROOT = Path(wine_quality_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
LOG_DIR = PACKAGE_ROOT / "logs"

class app_config(BaseModel):
    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str

class model_config(BaseModel):
    target: str
    features: List[str]
    numerical_vars: List[str]
    test_size: float
    random_state: int

class Config(BaseModel):
    app_config: app_config
    model_config_params: model_config

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=app_config(**parsed_config.data),
        model_config_params=model_config(**parsed_config.data),
    )

    return _config

config = create_and_validate_config()