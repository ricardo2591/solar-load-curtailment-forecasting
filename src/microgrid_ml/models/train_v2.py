"""V2 training pipeline: causal split, target encoding, physical weighting, LightGBM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl

from microgrid_ml.config import (
    LOAD_FEATURES_V2,
    SOLAR_FEATURES_V2,
    TARGET_LOAD,
    TARGET_SOLAR,
    TARGET_SOLAR_KW,
    PipelineConfig,
)
from microgrid_ml.data.etl_polars import run_etl_pipeline
from microgrid_ml.features.solar_physics import (
    compute_causal_site_profiles,
    enrich_split_with_profiles_and_physics,
)
from microgrid_ml.models.evaluate import RegressionMetrics, compute_regression_metrics


@dataclass
class TrainingArtifacts:
    """Trained models, blind-test evaluation metrics, and persisted paths."""

    solar_model_path: Path
    load_model_path: Path
    site_profiles_path: Path
    metrics_path: Path
    solar_metrics: RegressionMetrics
    load_metrics: RegressionMetrics
    solar_features: list[str]
    load_features: list[str]
    y_test_kw: np.ndarray
    y_pred_kw: np.ndarray
    y_test_load_kw: np.ndarray
    y_pred_load_kw: np.ndarray
    solar_importance_gain: pl.DataFrame


def temporal_split(
    df: pl.DataFrame,
    val_start: str,
    test_start: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Partition dataset chronologically into Train / Validation / Blind Test."""
    val_dt = datetime.fromisoformat(val_start)
    test_dt = datetime.fromisoformat(test_start)

    train_df = df.filter(pl.col("time_local_dt") < val_dt)
    val_df = df.filter(
        (pl.col("time_local_dt") >= val_dt) & (pl.col("time_local_dt") < test_dt)
    )
    test_df = df.filter(pl.col("time_local_dt") >= test_dt)
    return train_df, val_df, test_df


def compute_solar_sample_weights(
    df: pl.DataFrame,
    weights_cfg: dict[str, Any],
) -> np.ndarray:
    """Compute multiplicative sample weights emphasizing curtailment and extreme irradiance."""
    w_curt_val = float(weights_cfg.get("curtailment_multiplier", 2.5))
    high_poa = float(weights_cfg.get("high_poa_threshold_w_m2", 650.0))
    w_high_val = float(weights_cfg.get("high_poa_multiplier", 1.5))
    low_poa = float(weights_cfg.get("low_poa_threshold_w_m2", 150.0))
    w_low_val = float(weights_cfg.get("low_poa_multiplier", 1.3))

    w_curtailment = pl.when(pl.col("curtailment_state") == 1).then(w_curt_val).otherwise(1.0)
    w_peak = pl.when(pl.col("poa_global") >= high_poa).then(w_high_val).otherwise(1.0)
    w_low = pl.when(pl.col("poa_global") <= low_poa).then(w_low_val).otherwise(1.0)

    expr = (w_curtailment * w_peak * w_low).alias("sample_weight")
    return df.select(expr).to_numpy().flatten().astype(np.float32)


def _train_lgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    params: dict[str, Any],
    train_weights: np.ndarray | None = None,
    val_weights: np.ndarray | None = None,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, weight=train_weights, feature_name=feature_names)
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        weight=val_weights,
        reference=train_set,
        feature_name=feature_names,
    )

    lgb_params = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": "gbdt",
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params.get("max_depth", -1)),
        "min_child_samples": int(params.get("min_child_samples", 20)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.7)),
        "n_jobs": -1,
        "verbose": -1,
        "random_state": int(params.get("random_state", 42)),
    }

    return lgb.train(
        lgb_params,
        train_set,
        num_boost_round=int(params.get("num_boost_round", 2000)),
        valid_sets=[train_set, val_set],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=int(params.get("early_stopping_rounds", 50)),
                verbose=False,
            )
        ],
    )


def train_v2_pipeline(
    cfg: PipelineConfig,
    input_parquet: Path | None = None,
) -> TrainingArtifacts:
    """Run full V2 causal training, blind evaluation, and model serialization."""
    features_path = input_parquet or cfg.processed_features_parquet
    if input_parquet is None and not features_path.exists():
        df, _ = run_etl_pipeline(cfg)
    else:
        df = pl.read_parquet(features_path)

    split_cfg = cfg.raw["split"]
    physics_cfg = cfg.raw["physics"]
    weights_cfg = cfg.raw["sample_weights"]
    models_cfg = cfg.raw["models"]

    # 1. Strict chronological split BEFORE target encoding to eliminate data leakage
    train_raw, val_raw, test_raw = temporal_split(
        df,
        val_start=split_cfg["val_start"],
        test_start=split_cfg["test_start"],
    )

    # 2. Causal site profiles computed exclusively from training data
    site_profiles, global_fallbacks = compute_causal_site_profiles(train_raw)
    g0 = float(physics_cfg.get("extraterrestrial_irradiance_w_m2", 1361.0))

    train_df = enrich_split_with_profiles_and_physics(train_raw, site_profiles, global_fallbacks, g0)
    val_df = enrich_split_with_profiles_and_physics(val_raw, site_profiles, global_fallbacks, g0)
    test_df = enrich_split_with_profiles_and_physics(test_raw, site_profiles, global_fallbacks, g0)

    solar_features = [c for c in SOLAR_FEATURES_V2 if c in train_df.columns]
    load_features = [c for c in LOAD_FEATURES_V2 if c in train_df.columns]

    # 3. Train Model 1: Solar Capacity Factor with regime sample weights
    w_train = compute_solar_sample_weights(train_df, weights_cfg)
    w_val = compute_solar_sample_weights(val_df, weights_cfg)

    model_solar = _train_lgbm_regressor(
        X_train=train_df.select(solar_features).to_numpy(),
        y_train=train_df[TARGET_SOLAR].to_numpy(),
        X_val=val_df.select(solar_features).to_numpy(),
        y_val=val_df[TARGET_SOLAR].to_numpy(),
        feature_names=solar_features,
        params=models_cfg["solar_v2"],
        train_weights=w_train,
        val_weights=w_val,
    )

    cf_max = float(physics_cfg.get("capacity_factor_clip_max", 1.25))
    y_test_cf = test_df[TARGET_SOLAR].to_numpy()
    y_pred_cf = np.clip(
        model_solar.predict(
            test_df.select(solar_features).to_numpy(),
            num_iteration=model_solar.best_iteration,
        ),
        0.0,
        cf_max,
    )
    cap_test_kwp = test_df["solar_capacity_kwp"].to_numpy()
    y_test_kw = test_df[TARGET_SOLAR_KW].to_numpy()
    y_pred_kw = y_pred_cf * cap_test_kwp

    solar_metrics = compute_regression_metrics(
        y_true_kw=y_test_kw,
        y_pred_kw=y_pred_kw,
        y_true_cf=y_test_cf,
        y_pred_cf=y_pred_cf,
    )

    # 4. Train Model 2: Telecom Load Demand
    model_load = _train_lgbm_regressor(
        X_train=train_df.select(load_features).to_numpy(),
        y_train=train_df[TARGET_LOAD].to_numpy(),
        X_val=val_df.select(load_features).to_numpy(),
        y_val=val_df[TARGET_LOAD].to_numpy(),
        feature_names=load_features,
        params=models_cfg["load_v2"],
    )

    y_test_load_kw = test_df[TARGET_LOAD].to_numpy()
    y_pred_load_kw = np.clip(
        model_load.predict(
            test_df.select(load_features).to_numpy(),
            num_iteration=model_load.best_iteration,
        ),
        0.0,
        None,
    )
    load_metrics = compute_regression_metrics(y_true_kw=y_test_load_kw, y_pred_kw=y_pred_load_kw)

    # 5. Persist models and anonymized site profiles
    models_dir = cfg.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    solar_model_path = models_dir / "lgbm_solar_generation_v2.joblib"
    load_model_path = models_dir / "lgbm_load_demand_v2.joblib"
    site_profiles_path = models_dir / "site_profiles_v2.parquet"
    metrics_path = models_dir / "metrics_v2.json"

    joblib.dump(
        {
            "model": model_solar,
            "features": solar_features,
            "metrics": solar_metrics.to_dict(),
            "global_fallbacks": global_fallbacks,
        },
        solar_model_path,
    )
    joblib.dump(
        {
            "model": model_load,
            "features": load_features,
            "metrics": load_metrics.to_dict(),
            "global_fallbacks": global_fallbacks,
        },
        load_model_path,
    )
    site_profiles.write_parquet(site_profiles_path)

    metrics_payload = {
        "solar_v2": solar_metrics.to_dict(),
        "load_v2": load_metrics.to_dict(),
        "split_rows": {
            "train": train_df.height,
            "val": val_df.height,
            "test": test_df.height,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    importance_df = pl.DataFrame(
        {
            "feature": solar_features,
            "gain": model_solar.feature_importance(importance_type="gain"),
        }
    ).sort("gain", descending=True)

    return TrainingArtifacts(
        solar_model_path=solar_model_path,
        load_model_path=load_model_path,
        site_profiles_path=site_profiles_path,
        metrics_path=metrics_path,
        solar_metrics=solar_metrics,
        load_metrics=load_metrics,
        solar_features=solar_features,
        load_features=load_features,
        y_test_kw=y_test_kw,
        y_pred_kw=y_pred_kw,
        y_test_load_kw=y_test_load_kw,
        y_pred_load_kw=y_pred_load_kw,
        solar_importance_gain=importance_df,
    )
