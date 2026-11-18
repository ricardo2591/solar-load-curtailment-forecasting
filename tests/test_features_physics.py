"""Unit tests for physical bounds, operational masking, and feature engineering."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from microgrid_ml.data.etl_polars import (
    apply_physical_bounds,
    convert_csv_to_parquet,
    mask_operational_states,
)
from microgrid_ml.features.solar_physics import build_temporal_and_physical_features


def test_operational_masking_and_physical_clipping() -> None:
    df = pl.DataFrame(
        {
            "edge_offline_alarm": [1, 0, 0],
            "ghi_w_m2": [500.0, 0.0, 800.0],
            "solar_capacity_kwp": [10.0, 10.0, 10.0],
            "solar_power_kw": [4.0, 1.5, 25.0],
            "load_power_kw": [2.0, -0.5, 3.0],
            "battery_soc_pct": [80.0, 115.0, -5.0],
        }
    )
    masked = mask_operational_states(df)
    bounded = apply_physical_bounds(masked, overpower_ratio=1.30)

    assert bounded["solar_power_kw"][0] is None  # Offline masked
    assert bounded["load_power_kw"][0] is None  # Offline masked
    assert bounded["solar_power_kw"][1] == 0.0  # Night forced to 0
    assert bounded["load_power_kw"][1] == 0.0  # Negative load clipped
    assert bounded["battery_soc_pct"][1] == 100.0  # SoC upper bound
    assert bounded["solar_power_kw"][2] == 13.0  # 1.30 * 10.0 kWp ceiling
    assert bounded["battery_soc_pct"][2] == 0.0  # SoC lower bound


def test_soiling_curtailment_and_causal_lags() -> None:
    t0 = datetime(2026, 7, 1, 12, 0, 0)
    timestamps = [t0 + timedelta(minutes=15 * i) for i in range(6)]
    df = pl.DataFrame(
        {
            "site_id": ["MG-SITE-001"] * 6,
            "time_utc_dt": timestamps,
            "time_local_dt": timestamps,
            "solar_capacity_kwp": [10.0] * 6,
            "solar_power_kw": [1.0, 2.0, 3.0, 4.0, 5.0, 0.8],
            "load_power_kw": [2.0, 2.0, 2.0, 2.0, 2.0, 0.6],
            "battery_soc_pct": [60.0, 70.0, 80.0, 90.0, 94.0, 97.0],
            "ghi_w_m2": [700.0] * 6,
            "poa_global": [720.0] * 6,
            "precipitation_mm": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).with_columns(pl.col("time_utc_dt").dt.replace_time_zone("UTC"))

    feat = build_temporal_and_physical_features(df)

    # GHI must be dropped to avoid collinearity while ghi_w_m2_lag_1 is retained
    assert "ghi_w_m2" not in feat.columns
    assert "ghi_w_m2_lag_1" in feat.columns
    # Last interval triggers curtailment (SoC >= 95, load < 1.0, POA > 200)
    assert feat["curtailment_state"].to_list() == [0, 0, 0, 0, 0, 1]
    # Days since rain should increase monotonically after t=0 rain event
    assert feat["days_since_rain"][-1] > feat["days_since_rain"][0]
    # Causal lag-1 must equal prior step's value
    assert feat["battery_soc_pct_lag_1"][5] == 94.0


def test_convert_csv_to_parquet_streaming(tmp_path: Path) -> None:
    csv_file = tmp_path / "telemetry.csv"
    parquet_file = tmp_path / "out.parquet"
    csv_file.write_text(
        "timezone,site_id,region_zone,topology,solar_var_type,cardinal_orientation,"
        "edge_offline_alarm,solar_power_kw,inverter_array_power_kw,solar_charger_power_kw,"
        "solar_capacity_kwp,battery_soc_pct,battery_voltage_v,battery_current_a,load_power_kw,"
        "ac_load_power_kw,dc_load_power_kw,grid_power_kw,generator_power_kw,latitude,"
        "longitude,ambient_temp_c,cloud_cover_pct,ghi_w_m2,dni_w_m2,dhi_w_m2,"
        "precipitation_mm,tilt_deg,azimuth_deg,time_utc,time_local\n"
        "Etc/GMT+3,MG-SITE-001,Zone-Equatorial-Grid,Hybrid-Solar-Battery-Genset,"
        "hybrid_pv_total,South,0,5.2,3.4,1.8,10.0,88.0,53.2,20.1,0.8,0.2,0.6,0.0,0.0,"
        "6.5,-45.0,26.0,20.0,650.0,500.0,150.0,0.0,15,180,2026-07-01T15:00:00,2026-07-01T12:00:00\n",
        encoding="utf-8",
    )

    out_path = convert_csv_to_parquet(csv_file, parquet_file)
    assert out_path.exists()
    loaded = pl.read_parquet(out_path)
    assert loaded.height == 1
    assert "time_utc_dt" in loaded.columns
    assert "time_local_dt" in loaded.columns
    assert loaded["site_id"][0] == "MG-SITE-001"

