"""Clean visualization helpers for EDA, V2 model audit, and curtailment dashboards."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.patches import Patch

from microgrid_ml.models.train_v2 import TrainingArtifacts

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.sans-serif": "DejaVu Sans",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "figure.titlesize": 13,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def _save_or_close(fig: plt.Figure, out_path: Path | None) -> Path | None:
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def plot_capacity_factor_distribution(
    df: pl.DataFrame,
    out_path: Path | None = None,
    max_samples: int = 250_000,
) -> Path | None:
    """Plot daytime capacity factor histogram and KDE with mean/median markers."""
    valid = df.filter((pl.col("sun_up") == 1) & pl.col("capacity_factor").is_not_null())
    n_sample = min(max_samples, valid.height)
    series = valid.select("capacity_factor").sample(n=n_sample, seed=42).to_pandas()["capacity_factor"]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=130)
    sns.histplot(series, bins=40, kde=True, color="#2D6A4F", edgecolor="#1B4332", alpha=0.68, ax=ax)

    median_val = float(series.median())
    mean_val = float(series.mean())
    ax.axvline(median_val, color="#D90429", linestyle="--", linewidth=1.9, label=f"Median CF: {median_val:.3f}")
    ax.axvline(mean_val, color="#1D3557", linestyle=":", linewidth=1.9, label=f"Mean CF: {mean_val:.3f}")
    ax.set_title("Daytime Photovoltaic Capacity Factor Distribution (Synthetic Sample)", fontweight="bold")
    ax.set_xlabel("Capacity Factor (solar_power_kw / solar_capacity_kwp)")
    ax.set_ylabel("Count")
    ax.legend(frameon=True, facecolor="white")
    return _save_or_close(fig, out_path)


def plot_correlation_matrix(
    df: pl.DataFrame,
    out_path: Path | None = None,
    max_samples: int = 200_000,
) -> Path | None:
    """Plot Pearson correlation heatmap across electrical, battery, and solar variables."""
    cols = [
        "solar_power_kw",
        "capacity_factor",
        "poa_global",
        "load_power_kw",
        "battery_soc_pct",
        "battery_voltage_v",
        "battery_current_a",
        "grid_power_kw",
        "generator_power_kw",
        "ambient_temp_c",
    ]
    present = [c for c in cols if c in df.columns]
    sample_df = df.select(present).sample(n=min(max_samples, df.height), seed=42).to_pandas()
    non_const = [col for col in sample_df.columns if float(sample_df[col].std() or 0.0) > 1e-8]
    corr = sample_df[non_const].corr()

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=130)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 130, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
        annot_kws={"size": 8.5},
    )
    ax.set_title("Electrical, Battery & Irradiance Correlation Matrix", fontweight="bold")
    return _save_or_close(fig, out_path)


def plot_solar_blind_test_series(
    artifacts: TrainingArtifacts,
    out_path: Path | None = None,
    n_points: int = 200,
) -> Path | None:
    """Plot actual vs V2 predicted solar power (kW) on blind holdout intervals."""
    n = min(n_points, len(artifacts.y_test_kw))
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=130)
    ax.plot(artifacts.y_test_kw[:n], label="Actual Solar Power (kW)", color="#1B4332", linewidth=2.0)
    ax.plot(
        artifacts.y_pred_kw[:n],
        label="V2 Prediction (Causal Site Profiling + Sample Weights)",
        color="#D90429",
        linewidth=1.7,
        linestyle="--",
    )
    ax.set_title(
        f"Blind Test Validation: Solar Generation V2 (R2={artifacts.solar_metrics.r2:.3f}, "
        f"WAPE={artifacts.solar_metrics.wape_pct:.2f}%)",
        fontweight="bold",
    )
    ax.set_xlabel("Consecutive 15-Minute Daytime Intervals")
    ax.set_ylabel("Power (kW)")
    ax.legend(frameon=True, facecolor="white")
    return _save_or_close(fig, out_path)


def plot_feature_importance(
    artifacts: TrainingArtifacts,
    out_path: Path | None = None,
    top_n: int = 15,
) -> Path | None:
    """Plot top split-gain predictors for the Solar V2 LightGBM model."""
    imp_pd = artifacts.solar_importance_gain.head(top_n).to_pandas()
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=130)
    sns.barplot(
        data=imp_pd,
        x="gain",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        edgecolor="#1B4332",
        ax=ax,
    )
    ax.set_title(f"Top {top_n} Predictors in Solar V2 LightGBM (Split Gain)", fontweight="bold")
    ax.set_xlabel("Information Gain")
    ax.set_ylabel("Feature")
    return _save_or_close(fig, out_path)


def plot_site_forecast_dashboard(
    df_forecast: pl.DataFrame,
    out_path: Path | None = None,
) -> Path | None:
    """Plot 4-panel site forecast dashboard: solar vs potential, load, SoC, curtailment."""
    pdf = df_forecast.sort("time_local_dt").head(384).to_pandas()
    site_id = str(pdf["site_id"].iloc[0])
    topology = str(pdf["topology"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), dpi=130, sharex=True)
    ax_solar, ax_load, ax_soc, ax_curt = axes.flatten()

    ax_solar.plot(pdf["time_local_dt"], pdf["solar_potential_kw"], label="Unconstrained Potential (kW)", color="#F4A261", linestyle=":", linewidth=1.6)
    ax_solar.plot(pdf["time_local_dt"], pdf["solar_power_kw"], label="Actual Solar (kW)", color="#1B4332", linewidth=1.9)
    ax_solar.plot(pdf["time_local_dt"], pdf["solar_pred_kw"], label="Forecast V2 (kW)", color="#D90429", linestyle="--", linewidth=1.5)
    ax_solar.set_title("Solar Generation vs. Potential", fontweight="bold")
    ax_solar.set_ylabel("Power (kW)")
    ax_solar.legend(fontsize=8, frameon=True, facecolor="white")

    ax_load.plot(pdf["time_local_dt"], pdf["load_power_kw"], label="Actual Load (kW)", color="#264653", linewidth=1.9)
    ax_load.plot(pdf["time_local_dt"], pdf["load_pred_kw"], label="Forecast Load V2 (kW)", color="#2A9D8F", linestyle="--", linewidth=1.6)
    ax_load.set_title("Telecom Site Load Demand", fontweight="bold")
    ax_load.set_ylabel("Power (kW)")
    ax_load.legend(fontsize=8, frameon=True, facecolor="white")

    ax_soc.plot(pdf["time_local_dt"], pdf["battery_soc_pct"], color="#457B9D", linewidth=1.8, label="Battery SoC (%)")
    ax_soc.axhline(95.0, color="#E76F51", linestyle="--", linewidth=1.2, label="Saturation Threshold (95%)")
    ax_soc.set_title("Battery State of Charge Dynamics", fontweight="bold")
    ax_soc.set_ylabel("SoC (%)")
    ax_soc.set_ylim(0, 105)
    ax_soc.legend(fontsize=8, frameon=True, facecolor="white")

    ax_curt.fill_between(pdf["time_local_dt"], pdf["curtailment_kw"], color="#E63946", alpha=0.45, label="Estimated Curtailment (kW)")
    ax_curt.plot(pdf["time_local_dt"], pdf["curtailment_kw"], color="#9B2226", linewidth=1.4)
    ax_curt.set_title("Identified Solar Curtailment Losses", fontweight="bold")
    ax_curt.set_ylabel("Curtailed Power (kW)")
    ax_curt.legend(fontsize=8, frameon=True, facecolor="white")

    for ax in (ax_soc, ax_curt):
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(f"Executive Microgrid Forecast Dashboard — {site_id} ({topology})", fontweight="bold", y=1.01)
    return _save_or_close(fig, out_path)


def plot_fleet_curtailment_ranking(
    ranking_df: pl.DataFrame,
    out_path: Path | None = None,
) -> Path | None:
    """Plot horizontal bar ranking of synthetic sites with highest estimated curtailment."""
    pdf = ranking_df.to_pandas()
    colors = [
        "#E76F51" if "Genset" in str(topo) else "#2D6A4F" for topo in pdf["topology"]
    ]
    labels = pdf["site_id"].astype(str) + " [" + pdf["region_zone"].astype(str) + "]"

    fig, ax = plt.subplots(figsize=(12.5, 6.5), dpi=130)
    bars = ax.barh(labels, pdf["curtailed_kwh"], color=colors, edgecolor="#1B1B1B", linewidth=0.7, alpha=0.9)
    ax.invert_yaxis()
    ax.set_title("Top Microgrid Sites by Estimated Solar Curtailment (Synthetic Fleet)", fontweight="bold")
    ax.set_xlabel("Cumulative Curtailed Solar Energy (kWh)")
    ax.set_ylabel("Anonymized Site ID")

    max_val = float(pdf["curtailed_kwh"].max()) if len(pdf) else 1.0
    offset = max_val * 0.015
    for bar, (_, row) in zip(bars, pdf.iterrows(), strict=True):
        w = bar.get_width()
        if "Genset" in str(row["topology"]):
            lbl = f"{w:,.1f} kWh (Avoidable Diesel: ${row['avoidable_diesel_usd']:,.0f})"
        else:
            lbl = f"{w:,.1f} kWh (Idle Battery Capacity)"
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2.0, lbl, va="center", fontsize=8.5, fontweight="bold")

    legend_handles = [
        Patch(facecolor="#E76F51", edgecolor="black", label="Hybrid-Solar-Battery-Genset: Avoidable Diesel Impact"),
        Patch(facecolor="#2D6A4F", edgecolor="black", label="OffGrid-Solar-Battery: Surplus Capacity Opportunity"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", fontsize=8.5)
    ax.set_xlim(right=max_val * 1.48 if max_val > 0 else 10.0)
    return _save_or_close(fig, out_path)
