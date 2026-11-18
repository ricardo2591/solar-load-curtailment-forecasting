"""End-to-end smoke tests for ETL -> LightGBM V2 Training -> Inference Forecaster."""

from pathlib import Path

import pytest

from microgrid_ml.config import load_config
from microgrid_ml.data.etl_polars import run_etl_pipeline
from microgrid_ml.inference.predictor import MicrogridForecaster
from microgrid_ml.models.train_v2 import compute_solar_sample_weights, train_v2_pipeline


def test_end_to_end_pipeline_on_sample(tmp_path: Path) -> None:
    cfg = load_config()
    assert cfg.sample_raw_parquet.exists(), "Synthetic sample parquet must exist in repo."

    raw = {**cfg.raw}
    raw["paths"] = {
        **cfg.raw["paths"],
        "processed_features_parquet": str(tmp_path / "features.parquet"),
        "models_dir": str(tmp_path / "models"),
        "figures_dir": str(tmp_path / "figures"),
    }
    raw["models"] = {
        "solar_v2": {**cfg.raw["models"]["solar_v2"], "num_boost_round": 40, "min_child_samples": 20},
        "load_v2": {**cfg.raw["models"]["load_v2"], "num_boost_round": 40},
    }
    test_cfg = type(cfg)(repo_root=cfg.repo_root, raw=raw)

    df_gold, scorecard = run_etl_pipeline(test_cfg)
    assert df_gold.height > 1000
    assert scorecard.dormant_sites == 0

    weights = compute_solar_sample_weights(df_gold, test_cfg.raw["sample_weights"])
    assert len(weights) == df_gold.height
    assert float(weights.min()) >= 1.0

    artifacts = train_v2_pipeline(test_cfg)
    assert artifacts.solar_model_path.exists()
    assert artifacts.load_model_path.exists()
    assert artifacts.solar_metrics.r2 > 0.50
    assert artifacts.load_metrics.r2 > 0.50

    forecaster = MicrogridForecaster(test_cfg)
    catalog = forecaster.list_sites(df_gold, top_n=10)
    assert catalog.height == 10

    fc_df = forecaster.forecast_site(df_gold, site_id="MG-SITE-001", max_rows=200)
    assert (fc_df["solar_potential_kw"] >= fc_df["solar_pred_kw"]).all()
    summary = forecaster.summarize_forecast(fc_df)
    assert summary.site_id == "MG-SITE-001"
    assert summary.forecast_solar_kwh >= 0.0

    ranking = forecaster.rank_fleet_curtailment(df_gold, top_n=5)
    assert ranking.height == 5
    assert ranking["curtailed_kwh"][0] >= ranking["curtailed_kwh"][-1]

    with pytest.raises(ValueError, match="No observations found"):
        forecaster.forecast_site(df_gold, site_id="MG-SITE-DOES-NOT-EXIST")

