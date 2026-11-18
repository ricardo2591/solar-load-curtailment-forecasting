"""Configuration loader and canonical feature definitions for V2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "pipeline_v2.yaml"

TARGET_SOLAR = "capacity_factor"
TARGET_SOLAR_KW = "solar_power_kw"
TARGET_LOAD = "load_power_kw"

SOLAR_FEATURES_V2: list[str] = [
    # 1. Causal site energy signature (Target Encoding fitted strictly on Train)
    "site_cf_mean",
    "site_cf_std",
    # 2. Solar physics, transposed irradiance, and array mounting
    "poa_global",
    "dni_w_m2",
    "dhi_w_m2",
    "tilt_deg",
    "azimuth_deg",
    "ghi_w_m2_lag_1",
    # 3. Clear-sky index and solar astronomy
    "clearness_index_kt",
    "apparent_zenith_deg",
    "solar_azimuth_deg",
    # 4. Satellite meteorology
    "ambient_temp_c",
    "cloud_cover_pct",
    "precipitation_mm",
    # 5. Battery dynamics (causal lags)
    "battery_soc_pct_lag_4",
    "battery_soc_pct_lag_1",
    "battery_soc_pct_lag_96",
    # 6. Curtailment proxy, relative load, and soiling degradation
    "load_ratio_lag_4",
    "days_since_rain",
    # 7. Telecom load inertia (causal lags)
    "load_power_kw_lag_4",
    "load_power_kw_rolling_mean_1h",
    "load_power_kw_lag_96",
    # 8. Cyclical time and calendar
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
]

LOAD_FEATURES_V2: list[str] = [
    # 1. Causal site baseline demand
    "site_load_mean",
    # 2. Inertial telecom demand history
    "load_power_kw_lag_4",
    "load_power_kw_rolling_mean_1h",
    "load_power_kw_lag_96",
    # 3. Weather / cooling proxy
    "ambient_temp_c",
    "precipitation_mm",
    # 4. Battery dynamics
    "battery_soc_pct_lag_4",
    "battery_soc_pct_lag_1",
    # 5. Cyclical time and calendar
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
]


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved runtime configuration for ETL, training, evaluation, and inference."""

    repo_root: Path
    raw: dict[str, Any]

    def resolve_path(self, key: str) -> Path:
        """Resolve a path configured under `paths.<key>` relative to repo root."""
        rel_path = Path(self.raw["paths"][key])
        return rel_path if rel_path.is_absolute() else (self.repo_root / rel_path)

    @property
    def sample_raw_parquet(self) -> Path:
        return self.resolve_path("sample_raw_parquet")

    @property
    def processed_features_parquet(self) -> Path:
        return self.resolve_path("processed_features_parquet")

    @property
    def models_dir(self) -> Path:
        return self.resolve_path("models_dir")

    @property
    def figures_dir(self) -> Path:
        return self.resolve_path("figures_dir")


def load_config(
    config_path: str | Path | None = None,
    repo_root: Path | None = None,
) -> PipelineConfig:
    """Load YAML pipeline configuration with relative paths bound to `repo_root`."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    cfg_file = Path(config_path) if config_path is not None else (root / "configs" / "pipeline_v2.yaml")
    with cfg_file.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return PipelineConfig(repo_root=root, raw=raw)
