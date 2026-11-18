"""Out-of-core Polars ETL for 15-minute hybrid microgrid telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from microgrid_ml.config import PipelineConfig
from microgrid_ml.features.solar_physics import (
    build_temporal_and_physical_features,
    compute_solar_geometry_and_poa,
)

CANONICAL_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "timezone": pl.Categorical,
    "site_id": pl.Categorical,
    "region_zone": pl.Categorical,
    "topology": pl.Categorical,
    "solar_var_type": pl.Categorical,
    "cardinal_orientation": pl.Categorical,
    "edge_offline_alarm": pl.Int8,
    "solar_power_kw": pl.Float32,
    "inverter_array_power_kw": pl.Float32,
    "solar_charger_power_kw": pl.Float32,
    "solar_capacity_kwp": pl.Float32,
    "battery_soc_pct": pl.Float32,
    "battery_voltage_v": pl.Float32,
    "battery_current_a": pl.Float32,
    "load_power_kw": pl.Float32,
    "ac_load_power_kw": pl.Float32,
    "dc_load_power_kw": pl.Float32,
    "grid_power_kw": pl.Float32,
    "generator_power_kw": pl.Float32,
    "latitude": pl.Float32,
    "longitude": pl.Float32,
    "ambient_temp_c": pl.Float32,
    "cloud_cover_pct": pl.Float32,
    "ghi_w_m2": pl.Float32,
    "dni_w_m2": pl.Float32,
    "dhi_w_m2": pl.Float32,
    "precipitation_mm": pl.Float32,
    "tilt_deg": pl.Int16,
    "azimuth_deg": pl.Int16,
}

SOLAR_POWER_COLS = (
    "solar_power_kw",
    "inverter_array_power_kw",
    "solar_charger_power_kw",
)

NON_SOLAR_TELEMETRY_COLS = (
    "load_power_kw",
    "ac_load_power_kw",
    "dc_load_power_kw",
    "battery_soc_pct",
    "battery_voltage_v",
    "battery_current_a",
    "grid_power_kw",
    "generator_power_kw",
)

NON_NEGATIVE_POWER_COLS = (
    "load_power_kw",
    "grid_power_kw",
    "generator_power_kw",
    "ac_load_power_kw",
    "dc_load_power_kw",
)


@dataclass(frozen=True)
class DataHealthScorecard:
    """Statistical audit metrics before final daytime filtering."""

    total_rows: int
    daytime_rows: int
    nighttime_rows: int
    night_zero_integrity_pct: float
    continuous_15m_pct: float
    time_gap_rows: int
    zero_variance_cols: list[str]
    active_sites: int
    dormant_sites: int


def convert_csv_to_parquet(csv_path: str | Path, parquet_path: str | Path) -> Path:
    """Stream raw telemetry CSV to Snappy-compressed Parquet without materializing in RAM."""
    src = Path(csv_path)
    dst = Path(parquet_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    (
        pl.scan_csv(src, schema_overrides=CANONICAL_SCHEMA_OVERRIDES)
        .with_columns(
            [
                pl.col("time_utc").str.to_datetime(time_zone="UTC").alias("time_utc_dt"),
                pl.col("time_local").str.to_datetime().alias("time_local_dt"),
            ]
        )
        .drop(["time_utc", "time_local"])
        .sink_parquet(dst, compression="snappy")
    )
    return dst


def mask_operational_states(df: pl.DataFrame) -> pl.DataFrame:
    """Mask offline edge intervals to null and nighttime solar generation to zero."""
    expressions: list[pl.Expr] = []

    for col in SOLAR_POWER_COLS:
        if col in df.columns:
            expressions.append(
                pl.when(pl.col("edge_offline_alarm") == 1)
                .then(None)
                .when(pl.col("ghi_w_m2") == 0.0)
                .then(0.0)
                .otherwise(pl.col(col))
                .alias(col)
            )

    for col in NON_SOLAR_TELEMETRY_COLS:
        if col in df.columns:
            expressions.append(
                pl.when(pl.col("edge_offline_alarm") == 1)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    return df.with_columns(expressions) if expressions else df


def apply_physical_bounds(df: pl.DataFrame, overpower_ratio: float = 1.30) -> pl.DataFrame:
    """Clip telemetry to physically admissible bounds for power and battery SoC."""
    expressions: list[pl.Expr] = []

    for col in NON_NEGATIVE_POWER_COLS:
        if col in df.columns:
            expressions.append(pl.col(col).clip(lower_bound=0.0).alias(col))

    if "battery_soc_pct" in df.columns:
        expressions.append(pl.col("battery_soc_pct").clip(0.0, 100.0).alias("battery_soc_pct"))

    if "solar_power_kw" in df.columns and "solar_capacity_kwp" in df.columns:
        pmax_ceiling = pl.col("solar_capacity_kwp") * overpower_ratio
        expressions.append(
            pl.when((pl.col("solar_capacity_kwp") > 0) & (pl.col("solar_power_kw") > pmax_ceiling))
            .then(pmax_ceiling)
            .otherwise(pl.col("solar_power_kw"))
            .clip(lower_bound=0.0)
            .alias("solar_power_kw")
        )

    return df.with_columns(expressions) if expressions else df


def audit_data_health(df: pl.DataFrame, dormant_site_max_cf: float = 0.05) -> DataHealthScorecard:
    """Compute pre-modeling data health scorecard across temporal continuity and hardware."""
    total_rows = df.height
    daytime_rows = df.filter(pl.col("sun_up") == 1).height
    nighttime_rows = df.filter(pl.col("sun_up") == 0).height

    night_zeros = df.filter((pl.col("sun_up") == 0) & (pl.col("solar_power_kw") == 0.0)).height
    night_zero_pct = (night_zeros / nighttime_rows * 100.0) if nighttime_rows > 0 else 100.0

    delta_minutes = (
        (pl.col("time_local_dt") - pl.col("time_local_dt").shift(1))
        .dt.total_minutes()
        .over("site_id")
    )
    time_gap_rows = df.filter(delta_minutes > 15).height
    continuous_pct = ((total_rows - time_gap_rows) / total_rows * 100.0) if total_rows > 0 else 0.0

    zero_var_cols = [col for col in df.columns if df[col].n_unique() <= 1]

    site_cf_max = (
        df.filter(pl.col("solar_capacity_kwp") > 0)
        .group_by("site_id")
        .agg(
            (pl.col("solar_power_kw").max() / pl.col("solar_capacity_kwp").first()).alias(
                "cf_max_hist"
            )
        )
    )
    dormant_sites = site_cf_max.filter(pl.col("cf_max_hist") < dormant_site_max_cf).height

    return DataHealthScorecard(
        total_rows=total_rows,
        daytime_rows=daytime_rows,
        nighttime_rows=nighttime_rows,
        night_zero_integrity_pct=night_zero_pct,
        continuous_15m_pct=continuous_pct,
        time_gap_rows=time_gap_rows,
        zero_variance_cols=zero_var_cols,
        active_sites=df["site_id"].n_unique(),
        dormant_sites=dormant_sites,
    )


def filter_healthy_daytime_sites(
    df: pl.DataFrame,
    physics_cfg: dict[str, Any],
) -> pl.DataFrame:
    """Drop zero-variance flags, remove dormant sensors, and keep daytime ML observations."""
    cleaned = df
    if "edge_offline_alarm" in cleaned.columns and cleaned["edge_offline_alarm"].n_unique() <= 1:
        cleaned = cleaned.drop("edge_offline_alarm")

    dormant_max_cf = float(physics_cfg.get("dormant_site_max_cf", 0.05))
    poa_min = float(physics_cfg.get("daytime_poa_min_w_m2", 5.0))

    healthy_site_ids = (
        cleaned.filter(pl.col("solar_capacity_kwp") > 0)
        .group_by("site_id")
        .agg(
            (pl.col("solar_power_kw").max() / pl.col("solar_capacity_kwp").first()).alias("cf_max")
        )
        .filter(pl.col("cf_max") >= dormant_max_cf)["site_id"]
        .to_list()
    )

    cleaned = cleaned.filter(pl.col("site_id").cast(pl.Utf8).is_in([str(s) for s in healthy_site_ids]))
    return cleaned.filter((pl.col("sun_up") == 1) | (pl.col("poa_global") > poa_min))


def run_etl_pipeline(
    cfg: PipelineConfig,
    input_parquet: Path | None = None,
    output_parquet: Path | None = None,
) -> tuple[pl.DataFrame, DataHealthScorecard]:
    """Execute end-to-end Polars ETL from consolidated Parquet to Gold daytime ML matrix."""
    src_path = input_parquet or cfg.sample_raw_parquet
    dst_path = output_parquet or cfg.processed_features_parquet
    physics_cfg = cfg.raw["physics"]

    df = pl.read_parquet(src_path).sort(["site_id", "time_utc_dt"])
    df = mask_operational_states(df)
    df = apply_physical_bounds(df, overpower_ratio=float(physics_cfg["solar_overpower_clip_ratio"]))
    df = compute_solar_geometry_and_poa(df, physics_cfg=physics_cfg)
    df = build_temporal_and_physical_features(df, physics_cfg=physics_cfg)

    df_valid = df.filter(
        pl.col("solar_power_kw").is_not_null()
        & pl.col("poa_global").is_not_null()
        & pl.col("ambient_temp_c").is_not_null()
        & pl.col("load_power_kw").is_not_null()
    )
    scorecard = audit_data_health(
        df_valid,
        dormant_site_max_cf=float(physics_cfg.get("dormant_site_max_cf", 0.05)),
    )
    df_gold = filter_healthy_daytime_sites(df_valid, physics_cfg=physics_cfg)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df_gold.write_parquet(
        dst_path,
        compression="snappy",
        statistics=True,
        row_group_size=1_000_000,
    )
    return df_gold, scorecard
