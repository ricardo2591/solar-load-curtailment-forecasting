"""CLI entrypoints for ETL, V2 training, figure generation, and inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from microgrid_ml.config import load_config
from microgrid_ml.data.etl_polars import run_etl_pipeline
from microgrid_ml.inference.predictor import MicrogridForecaster, load_features_parquet
from microgrid_ml.models.train_v2 import train_v2_pipeline
from microgrid_ml.visualization.plots import (
    plot_capacity_factor_distribution,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_fleet_curtailment_ranking,
    plot_site_forecast_dashboard,
    plot_solar_blind_test_series,
)


def prepare_features_cmd(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Polars + pvlib ETL to build V2 Gold features.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df_gold, scorecard = run_etl_pipeline(cfg)
    plot_capacity_factor_distribution(df_gold, cfg.figures_dir / "capacity_factor_distribution.png")
    plot_correlation_matrix(df_gold, cfg.figures_dir / "correlation_matrix.png")

    print(
        f"ETL complete: {df_gold.height:,} daytime rows across {df_gold['site_id'].n_unique()} sites "
        f"(Scorecard valid rows before day filter: {scorecard.total_rows:,})."
    )


def train_cmd(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Solar V2 & Load V2 LightGBM models.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    artifacts = train_v2_pipeline(cfg)
    plot_solar_blind_test_series(artifacts, cfg.figures_dir / "solar_blind_test_series.png")
    plot_feature_importance(artifacts, cfg.figures_dir / "feature_importance_solar_v2.png")

    print("V2 Blind-Test Results:")
    print(
        f"  Solar -> R2={artifacts.solar_metrics.r2:.4f}, nRMSE={artifacts.solar_metrics.nrmse_pu:.4f}, "
        f"RMSE={artifacts.solar_metrics.rmse_kw:.3f} kW, MAE={artifacts.solar_metrics.mae_kw:.3f} kW, "
        f"WAPE={artifacts.solar_metrics.wape_pct:.2f}%"
    )
    print(
        f"  Load  -> R2={artifacts.load_metrics.r2:.4f}, RMSE={artifacts.load_metrics.rmse_kw:.3f} kW, "
        f"WAPE={artifacts.load_metrics.wape_pct:.2f}%"
    )


def predict_cmd(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run site forecast and fleet curtailment ranking.")
    parser.add_argument("--site-id", default="MG-SITE-001", help="Anonymized site identifier.")
    parser.add_argument("--start-date", default="2026-07-15", help="Start timestamp (ISO).")
    parser.add_argument("--end-date", default="2026-07-23", help="End timestamp (ISO).")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df_features = load_features_parquet(cfg)
    forecaster = MicrogridForecaster(cfg)

    df_fc = forecaster.forecast_site(
        df_features,
        site_id=args.site_id,
        start_date=args.start_date,
        end_date=args.end_date,
        max_rows=672,
    )
    summary = forecaster.summarize_forecast(df_fc)
    plot_site_forecast_dashboard(df_fc, cfg.figures_dir / "site_forecast_dashboard.png")

    ranking = forecaster.rank_fleet_curtailment(df_features, top_n=15)
    plot_fleet_curtailment_ranking(ranking, cfg.figures_dir / "fleet_curtailment_ranking.png")

    print(
        f"Forecast {summary.site_id} ({summary.topology}, {summary.solar_capacity_kwp:.2f} kWp): "
        f"Pred Solar={summary.forecast_solar_kwh:.1f} kWh, Curtailed={summary.curtailed_kwh:.1f} kWh "
        f"({summary.curtailment_pct_of_potential:.1f}%), Avoidable Diesel=${summary.avoidable_diesel_usd:.2f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="microgrid_ml", description="V2 Microgrid ML CLI.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-features")
    sub.add_parser("train")
    sub.add_parser("predict")

    args, remainder = parser.parse_known_args(argv)
    if args.command == "prepare-features":
        prepare_features_cmd(remainder)
    elif args.command == "train":
        train_cmd(remainder)
    elif args.command == "predict":
        predict_cmd(remainder)


if __name__ == "__main__":
    main()
