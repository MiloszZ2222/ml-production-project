import pytest
import yaml
from pydantic import ValidationError
from pydantic_settings import (
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

from src.config import AWSConfig, Settings, get_settings


def test_config_loaded_successfully():
    config = Settings()

    assert config.aws.region
    assert config.aws.private_bucket
    assert config.mlflow.tracking_uri
    assert config.mlflow.model_name

    assert 0 < config.training.test_size < 1
    assert 1 <= config.api.port <= 65535


def test_yaml_configuration_loading(tmp_path):
    # Create an independent YAML configuration for testing
    config_data = Settings().model_dump()

    config_data["aws"]["region"] = "test-region-1"

    config_path = tmp_path / "test_config.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_data, file)

    class TestSettings(Settings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls,
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ):
            return (
                YamlConfigSettingsSource(
                    settings_cls,
                    yaml_file=config_path,
                ),
            )

    config = TestSettings()

    assert config.aws.region == "test-region-1"


def test_environment_variable_override(monkeypatch):
    monkeypatch.setenv("ML_AWS__REGION", "test-region-2")

    config = Settings()

    assert config.aws.region == "test-region-2"


def test_invalid_api_port():
    with pytest.raises(ValidationError):
        Settings(
            api={
                "host": "127.0.0.1",
                "port": 99999,
            }
        )


def test_invalid_test_size():
    with pytest.raises(ValidationError):
        Settings(
            training={
                "random_state": 42,
                "test_size": 1.5,
                "model_params": {},
            }
        )


def test_missing_required_field():
    config_data = Settings().aws.model_dump()

    del config_data["region"]

    with pytest.raises(ValidationError):
        AWSConfig.model_validate(config_data)


def test_settings_cache():
    assert get_settings() is get_settings()
