"""Solar physics features (pvlib NREL solar position + isotropic POA) and causal lags."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pvlib


def compute_solar_geometry_and_poa(
    df: pl.DataFrame,
    physics_cfg: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Compute astronomical solar angles and Plane-of-Array (POA) irradiance via pvlib."""
    cfg = physics_cfg or {}
    default_tilt = float(cfg.get("default_tilt_deg", 15.0))
    default_azimuth = float(cfg.get("default_azimuth_deg", 180.0))
    sun_up_max_zenith = float(cfg.get("sun_up_zenith_max_deg", 85.0))

    unique_time_series = (
        df.select("time_utc_dt")
        .unique()
        .sort("time_utc_dt")["time_utc_dt"]
    )
    unique_times = pd.DatetimeIndex(unique_time_series.to_list())
    if unique_times.tz is None:
        unique_times = unique_times.tz_localize("UTC")

    coords_unique = (
        df.filter((pl.col("latitude") != 0.0) & (pl.col("longitude") != 0.0))
        .select(["latitude", "longitude"])
        .unique()
    )

    zenith_records: list[pl.DataFrame] = []
    seen_keys: set[tuple[float, float]] = set()
    time_utc_values = unique_times.tz_convert("UTC").tz_localize(None).to_numpy()

    for lat_val, lon_val in coords_unique.iter_rows():
        lat_k = round(float(lat_val), 3)
        lon_k = round(float(lon_val), 3)
        key = (lat_k, lon_k)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        solpos = pvlib.solarposition.get_solarposition(
            unique_times,
            latitude=lat_k,
            longitude=lon_k,
            method="nrel_numpy",
        )
        zenith_records.append(
            pl.DataFrame(
                {
                    "lat_k": np.full(len(unique_times), lat_k, dtype=np.float32),
                    "lon_k": np.full(len(unique_times), lon_k, dtype=np.float32),
                    "time_utc_dt": time_utc_values,
                    "apparent_zenith_deg": solpos["apparent_zenith"].to_numpy(dtype=np.float32),
                    "solar_azimuth_deg": solpos["azimuth"].to_numpy(dtype=np.float32),
                }
            )
        )

    df_solpos = pl.concat(zenith_records).with_columns(
        pl.col("time_utc_dt").dt.replace_time_zone("UTC")
    )

    df_joined = (
        df.with_columns(
            [
                pl.col("latitude").round(3).cast(pl.Float32).alias("lat_k"),
                pl.col("longitude").round(3).cast(pl.Float32).alias("lon_k"),
            ]
        )
        .join(df_solpos, on=["lat_k", "lon_k", "time_utc_dt"], how="left")
        .drop(["lat_k", "lon_k"])
    )

    poa_comp = pvlib.irradiance.get_total_irradiance(
        surface_tilt=df_joined["tilt_deg"].fill_null(default_tilt).cast(pl.Float32).to_numpy(),
        surface_azimuth=df_joined["azimuth_deg"]
        .fill_null(default_azimuth)
        .cast(pl.Float32)
        .to_numpy(),
        solar_zenith=df_joined["apparent_zenith_deg"].fill_null(90.0).cast(pl.Float32).to_numpy(),
        solar_azimuth=df_joined["solar_azimuth_deg"].fill_null(180.0).cast(pl.Float32).to_numpy(),
        dni=df_joined["dni_w_m2"].fill_null(0.0).cast(pl.Float32).to_numpy(),
        ghi=df_joined["ghi_w_m2"].fill_null(0.0).cast(pl.Float32).to_numpy(),
        dhi=df_joined["dhi_w_m2"].fill_null(0.0).cast(pl.Float32).to_numpy(),
        model="isotropic",
    )

    return df_joined.with_columns(
        [
            pl.Series("poa_global", poa_comp["poa_global"].astype(np.float32)).clip(
                lower_bound=0.0
            ),
            (pl.col("apparent_zenith_deg") < sun_up_max_zenith).cast(pl.Int8).alias("sun_up"),
        ]
    )


def build_temporal_and_physical_features(
    df: pl.DataFrame,
    physics_cfg: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Create normalized capacity factor, causal lags, soiling proxy, and curtailment flags."""
    cfg = physics_cfg or {}
    cf_max = float(cfg.get("capacity_factor_clip_max", 1.25))
    rain_wash_mm = float(cfg.get("rain_wash_threshold_mm", 2.0))
    soc_thr = float(cfg.get("curtailment_soc_threshold_pct", 95.0))
    load_thr = float(cfg.get("curtailment_load_threshold_kw", 1.0))
    poa_thr = float(cfg.get("curtailment_poa_threshold_w_m2", 200.0))

    df_sorted = df.sort(["site_id", "time_utc_dt"])

    kpi_expr = (
        pl.when(pl.col("solar_capacity_kwp") > 0)
        .then(pl.col("solar_power_kw") / pl.col("solar_capacity_kwp"))
        .otherwise(None)
        .clip(0.0, cf_max)
        .alias("capacity_factor")
    )

    h_dec = (
        pl.col("time_local_dt").dt.hour().cast(pl.Float32)
        + pl.col("time_local_dt").dt.minute().cast(pl.Float32) / 60.0
        + pl.col("time_local_dt").dt.second().cast(pl.Float32) / 3600.0
    )
    hour_sin_expr = (2.0 * np.pi * h_dec / 24.0).sin().alias("hour_sin")
    hour_cos_expr = (2.0 * np.pi * h_dec / 24.0).cos().alias("hour_cos")
    day_of_week_expr = pl.col("time_local_dt").dt.weekday().cast(pl.Int8).alias("day_of_week")
    is_weekend_expr = (pl.col("time_local_dt").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend")

    lag_targets = ("load_power_kw", "battery_soc_pct", "ghi_w_m2")
    lag_steps = (1, 4, 96)
    lag_exprs = [
        pl.col(col).shift(step).over("site_id").alias(f"{col}_lag_{step}")
        for col in lag_targets
        if col in df_sorted.columns
        for step in lag_steps
    ]

    rolling_exprs: list[pl.Expr] = []
    if "load_power_kw" in df_sorted.columns:
        rolling_exprs.append(
            pl.col("load_power_kw")
            .rolling_mean(window_size=4, min_samples=1)
            .over("site_id")
            .alias("load_power_kw_rolling_mean_1h")
        )

    last_rain_time = (
        pl.when(pl.col("precipitation_mm") > rain_wash_mm)
        .then(pl.col("time_local_dt"))
        .otherwise(None)
        .forward_fill()
        .over("site_id")
    )
    first_site_time = pl.col("time_local_dt").first().over("site_id")
    days_since_rain_expr = (
        ((pl.col("time_local_dt") - pl.coalesce([last_rain_time, first_site_time])).dt.total_seconds())
        / 86400.0
    ).cast(pl.Float32).alias("days_since_rain")

    curtailment_cond = (
        (pl.col("battery_soc_pct") >= soc_thr)
        & (pl.col("load_power_kw") < load_thr)
        & (pl.col("poa_global") > poa_thr)
    )
    curtailment_expr = (
        pl.when(curtailment_cond.fill_null(False))
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("curtailment_state")
    )

    out = df_sorted.with_columns(
        [
            kpi_expr,
            hour_sin_expr,
            hour_cos_expr,
            day_of_week_expr,
            is_weekend_expr,
            days_since_rain_expr,
            curtailment_expr,
            *lag_exprs,
            *rolling_exprs,
        ]
    )

    # Drop raw GHI after POA transposition and lag_1 creation to prevent VIF collinearity
    if "ghi_w_m2" in out.columns:
        out = out.drop("ghi_w_m2")

    return out


def compute_causal_site_profiles(
    train_df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Fit target-encoded site energy signatures exclusively on the training partition."""
    site_profiles = train_df.group_by("site_id").agg(
        [
            pl.col("capacity_factor").mean().alias("site_cf_mean"),
            pl.col("capacity_factor").std().fill_null(0.0).alias("site_cf_std"),
            pl.col("load_power_kw").mean().alias("site_load_mean"),
        ]
    )
    global_fallbacks = {
        "site_cf_mean": float(train_df["capacity_factor"].mean()),
        "site_cf_std": float(train_df["capacity_factor"].std() or 0.0),
        "site_load_mean": float(train_df["load_power_kw"].mean()),
    }
    return site_profiles, global_fallbacks


def enrich_split_with_profiles_and_physics(
    split_df: pl.DataFrame,
    site_profiles: pl.DataFrame,
    global_fallbacks: dict[str, float],
    extraterrestrial_irradiance: float = 1361.0,
) -> pl.DataFrame:
    """Attach causal train-fitted profiles, relative load ratio, and clearness index kt."""
    res = split_df.join(site_profiles, on="site_id", how="left").with_columns(
        [
            pl.col("site_cf_mean").fill_null(global_fallbacks["site_cf_mean"]),
            pl.col("site_cf_std").fill_null(global_fallbacks["site_cf_std"]),
            pl.col("site_load_mean").fill_null(global_fallbacks["site_load_mean"]),
            (
                pl.col("load_power_kw_lag_4")
                / pl.col("solar_capacity_kwp").clip(lower_bound=0.1)
            ).alias("load_ratio_lag_4"),
        ]
    )

    zenith_rad = pl.col("apparent_zenith_deg") * (np.pi / 180.0)
    g_extraterrestrial = (extraterrestrial_irradiance * zenith_rad.cos()).clip(lower_bound=20.0)

    return res.with_columns(
        (pl.col("poa_global") / g_extraterrestrial)
        .clip(lower_bound=0.0, upper_bound=1.5)
        .alias("clearness_index_kt")
    )
