from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class AWSConfig(BaseModel):
    profile: str | None = None
    region: str
    private_bucket: str
    public_bucket: str


class DataConfig(BaseModel):
    prefix: str
    train_filename: str
    test_filename: str
    target_column: str
    id_column: str


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    model_name: str
    model_alias: str
    backend_store_path: str
    artifact_prefix: str


class TrainingConfig(BaseModel):
    random_state: int
    test_size: float = Field(gt=0, lt=1)
    model_params: dict[str, int | float]


class APIConfig(BaseModel):
    host: str
    port: int = Field(ge=1, le=65535)


class DeploymentConfig(BaseModel):
    model_url: str = Field(pattern=r"^https://")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    download_timeout: int = Field(gt=0)


class Settings(BaseSettings):
    aws: AWSConfig
    data: DataConfig
    mlflow: MLflowConfig
    training: TrainingConfig
    api: APIConfig
    deployment: DeploymentConfig

    model_config = SettingsConfigDict(
        yaml_file=CONFIG_PATH,
        env_file=ENV_PATH,
        env_prefix="ML_",
        env_nested_delimiter="__",
        extra="forbid",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
