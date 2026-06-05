from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Model endpoints
    production_model_url: str = "http://localhost:8001"
    challenger_model_url: str = "http://localhost:8002"
    production_model_version: str = "v1.2.0"
    challenger_model_version: str = "v1.3.0"

    # Routing
    canary_weight: float = 0.10  # fraction of traffic routed to challenger
    shadow_enabled: bool = True   # async duplicate to challenger

    # Drift thresholds
    ks_drift_threshold: float = 0.05   # KS statistic p-value below which drift fires
    psi_drift_threshold: float = 0.20  # PSI above which drift fires
    rollback_on_drift: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Database
    database_url: str = "postgresql+asyncpg://shadowserve:shadowserve@localhost:5432/shadowserve"

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"

    # Prometheus
    metrics_port: int = 9090

    class Config:
        env_file = ".env"
        env_prefix = "SHADOWSERVE_"


settings = Settings()
