"""Production inference module for 15-min solar, load, and curtailment estimation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from microgrid_ml.config import PipelineConfig
from microgrid_ml.data.etl_polars import run_etl_pipeline
from microgrid_ml.features.solar_physics import enrich_split_with_profiles_and_physics
from microgrid_ml.models.train_v2 import train_v2_pipeline


@dataclass(frozen=True)
class SiteForecastSummary:
    """Energy and curtailment summary for an evaluated forecasting window."""

    site_id: str
    topology: str
    solar_capacity_kwp: float
    intervals: int
    forecast_solar_kwh: float
    actual_solar_kwh: float
    actual_load_kwh: float
    forecast_load_kwh: float
    curtailed_kwh: float
    curtailment_pct_of_potential: float
    avoidable_diesel_liters: float
    avoidable_diesel_usd: float


class MicrogridForecaster:
    """Simultaneous 15-minute solar, load demand, and counterfactual curtailment predictor."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        models_dir = cfg.models_dir
        solar_path = models_dir / "lgbm_solar_generation_v2.joblib"
        load_path = models_dir / "lgbm_load_demand_v2.joblib"
        profiles_path = models_dir / "site_profiles_v2.parquet"

        if not (solar_path.exists() and load_path.exists() and profiles_path.exists()):
            train_v2_pipeline(cfg)

        solar_bundle = joblib.load(solar_path)
        load_bundle = joblib.load(load_path)

        self.solar_model = solar_bundle["model"]
        self.solar_features: list[str] = solar_bundle["features"]
        self.global_fallbacks: dict[str, float] = solar_bundle["global_fallbacks"]

        self.load_model = load_bundle["model"]
        self.load_features: list[str] = load_bundle["features"]
        self.site_profiles = pl.read_parquet(profiles_path)

    def list_sites(self, df_features: pl.DataFrame, top_n: int = 20) -> pl.DataFrame:
        """Return anonymized catalog of available microgrid sites."""
        return (
            df_features.group_by("site_id")
            .agg(
                [
                    pl.col("region_zone").first(),
                    pl.col("topology").first(),
                    pl.col("solar_capacity_kwp").first(),
                    pl.col("latitude").first(),
                    pl.col("longitude").first(),
                    pl.len().alias("daytime_intervals"),
                ]
            )
            .sort("site_id")
            .head(top_n)
        )

    def _enrich_features(self, df: pl.DataFrame) -> pl.DataFrame:
        g0 = float(self.cfg.raw["physics"].get("extraterrestrial_irradiance_w_m2", 1361.0))
        return enrich_split_with_profiles_and_physics(
            df,
            site_profiles=self.site_profiles,
            global_fallbacks=self.global_fallbacks,
            extraterrestrial_irradiance=g0,
        )

    def predict_frame(self, df_features: pl.DataFrame) -> pl.DataFrame:
        """Predict solar power, unconstrained solar counterfactual, load, and curtailment."""
        enriched = self._enrich_features(df_features)
        cf_max = float(self.cfg.raw["physics"].get("capacity_factor_clip_max", 1.25))

        X_solar = enriched.select(self.solar_features).to_numpy()
        cf_pred = np.clip(self.solar_model.predict(X_solar), 0.0, cf_max)
        cap_kwp = enriched["solar_capacity_kwp"].to_numpy()
        solar_pred_kw = (cf_pred * cap_kwp).astype(np.float32)

        # Counterfactual unconstrained solar estimate (neutral unsaturated battery SoC = 50%)
        unconstrained = enriched.with_columns(
            [
                pl.lit(50.0, dtype=pl.Float32).alias("battery_soc_pct_lag_1"),
                pl.lit(50.0, dtype=pl.Float32).alias("battery_soc_pct_lag_4"),
                pl.lit(50.0, dtype=pl.Float32).alias("battery_soc_pct_lag_96"),
                pl.max_horizontal(pl.col("load_ratio_lag_4"), pl.lit(0.35)).alias(
                    "load_ratio_lag_4"
                ),
            ]
        )
        cf_potential = np.clip(
            self.solar_model.predict(unconstrained.select(self.solar_features).to_numpy()),
            0.0,
            cf_max,
        )
        potential_kw = np.maximum(cf_potential * cap_kwp, solar_pred_kw).astype(np.float32)

        is_curtailed = enriched["curtailment_state"].to_numpy() == 1
        actual_or_pred_kw = enriched["solar_power_kw"].to_numpy()
        curtailment_kw = np.where(
            is_curtailed,
            np.maximum(potential_kw - np.minimum(actual_or_pred_kw, solar_pred_kw), 0.0),
            0.0,
        ).astype(np.float32)

        X_load = enriched.select(self.load_features).to_numpy()
        load_pred_kw = np.clip(self.load_model.predict(X_load), 0.0, None).astype(np.float32)
        net_balance_kw = (solar_pred_kw - load_pred_kw).astype(np.float32)

        return enriched.with_columns(
            [
                pl.Series("solar_pred_kw", solar_pred_kw),
                pl.Series("solar_potential_kw", potential_kw),
                pl.Series("load_pred_kw", load_pred_kw),
                pl.Series("curtailment_kw", curtailment_kw),
                pl.Series("net_balance_pred_kw", net_balance_kw),
            ]
        )

    def forecast_site(
        self,
        df_features: pl.DataFrame,
        site_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_rows: int | None = None,
    ) -> pl.DataFrame:
        """Run simultaneous solar, load, and curtailment inference for a single site."""
        subset = df_features.filter(pl.col("site_id").cast(pl.Utf8) == site_id).sort(
            "time_local_dt"
        )
        if start_date:
            subset = subset.filter(pl.col("time_local_dt") >= datetime.fromisoformat(start_date))
        if end_date:
            subset = subset.filter(pl.col("time_local_dt") <= datetime.fromisoformat(end_date))
        if max_rows:
            subset = subset.head(max_rows)
        if subset.is_empty():
            raise ValueError(f"No observations found for site_id='{site_id}' in requested window.")

        return self.predict_frame(subset)

    def summarize_forecast(self, df_forecast: pl.DataFrame) -> SiteForecastSummary:
        """Aggregate energy KPIs (kWh) and avoidable diesel impact over a forecasted site."""
        dt_hours = 0.25
        inf_cfg = self.cfg.raw["inference"]
        diesel_price = float(inf_cfg.get("diesel_price_usd_per_liter", 1.10))
        liters_per_kwh = float(inf_cfg.get("diesel_liters_per_kwh", 0.30))

        forecast_solar_kwh = float(df_forecast["solar_pred_kw"].sum()) * dt_hours
        actual_solar_kwh = float(df_forecast["solar_power_kw"].sum()) * dt_hours
        actual_load_kwh = float(df_forecast["load_power_kw"].sum()) * dt_hours
        forecast_load_kwh = float(df_forecast["load_pred_kw"].sum()) * dt_hours
        curtailed_kwh = float(df_forecast["curtailment_kw"].sum()) * dt_hours
        potential_kwh = actual_solar_kwh + curtailed_kwh
        curtailment_pct = (curtailed_kwh / potential_kwh * 100.0) if potential_kwh > 0 else 0.0

        avoidable_liters = curtailed_kwh * liters_per_kwh
        avoidable_usd = avoidable_liters * diesel_price

        return SiteForecastSummary(
            site_id=str(df_forecast["site_id"][0]),
            topology=str(df_forecast["topology"][0]),
            solar_capacity_kwp=float(df_forecast["solar_capacity_kwp"][0]),
            intervals=df_forecast.height,
            forecast_solar_kwh=forecast_solar_kwh,
            actual_solar_kwh=actual_solar_kwh,
            actual_load_kwh=actual_load_kwh,
            forecast_load_kwh=forecast_load_kwh,
            curtailed_kwh=curtailed_kwh,
            curtailment_pct_of_potential=curtailment_pct,
            avoidable_diesel_liters=avoidable_liters,
            avoidable_diesel_usd=avoidable_usd,
        )

    def rank_fleet_curtailment(
        self,
        df_features: pl.DataFrame,
        top_n: int = 15,
    ) -> pl.DataFrame:
        """Rank anonymized fleet sites by cumulative curtailed solar energy and diesel impact."""
        inf_cfg = self.cfg.raw["inference"]
        diesel_price = float(inf_cfg.get("diesel_price_usd_per_liter", 1.10))
        liters_per_kwh = float(inf_cfg.get("diesel_liters_per_kwh", 0.30))

        predicted = self.predict_frame(df_features)
        return (
            predicted.group_by("site_id")
            .agg(
                [
                    pl.col("region_zone").first(),
                    pl.col("topology").first(),
                    pl.col("solar_capacity_kwp").first(),
                    (pl.col("curtailment_kw").sum() * 0.25).alias("curtailed_kwh"),
                    (pl.col("solar_power_kw").sum() * 0.25).alias("actual_solar_kwh"),
                    pl.col("curtailment_state").sum().alias("curtailment_intervals"),
                ]
            )
            .with_columns(
                [
                    (pl.col("curtailed_kwh") * liters_per_kwh).alias("avoidable_diesel_liters"),
                    (pl.col("curtailed_kwh") * liters_per_kwh * diesel_price).alias(
                        "avoidable_diesel_usd"
                    ),
                ]
            )
            .sort("curtailed_kwh", descending=True)
            .head(top_n)
        )


def load_features_parquet(cfg: PipelineConfig, path: Path | None = None) -> pl.DataFrame:
    """Load processed daytime feature matrix, running ETL automatically if absent."""
    target_path = path or cfg.processed_features_parquet
    if path is None and not target_path.exists():
        df_gold, _ = run_etl_pipeline(cfg)
        return df_gold
    return pl.read_parquet(target_path)
