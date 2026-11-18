"""Unit tests ensuring strict chronological separation and zero target leakage."""

from datetime import datetime, timedelta

import polars as pl

from microgrid_ml.features.solar_physics import (
    compute_causal_site_profiles,
    enrich_split_with_profiles_and_physics,
)
from microgrid_ml.models.train_v2 import temporal_split


def _make_mock_features_df() -> pl.DataFrame:
    rows = []
    start = datetime(2026, 5, 31, 10, 0, 0)
    for i in range(6):
        dt = start + timedelta(days=i * 15)
        rows.append(
            {
                "site_id": "MG-SITE-001",
                "time_local_dt": dt,
                "capacity_factor": 0.2 if dt < datetime(2026, 6, 1) else 0.95,
                "load_power_kw": 1.0 if dt < datetime(2026, 6, 1) else 10.0,
                "solar_capacity_kwp": 10.0,
                "load_power_kw_lag_4": 1.0,
                "apparent_zenith_deg": 25.0,
                "poa_global": 750.0,
            }
        )
    # Unseen site appearing only in blind test
    rows.append(
        {
            "site_id": "MG-SITE-999",
            "time_local_dt": datetime(2026, 7, 15, 12, 0, 0),
            "capacity_factor": 0.6,
            "load_power_kw": 2.5,
            "solar_capacity_kwp": 10.0,
            "load_power_kw_lag_4": 2.0,
            "apparent_zenith_deg": 20.0,
            "poa_global": 820.0,
        }
    )
    return pl.DataFrame(rows)


def test_temporal_split_boundaries() -> None:
    df = _make_mock_features_df()
    train_df, val_df, test_df = temporal_split(df, "2026-06-01T00:00:00", "2026-07-01T00:00:00")

    assert train_df["time_local_dt"].max() < datetime(2026, 6, 1)
    assert val_df["time_local_dt"].min() >= datetime(2026, 6, 1)
    assert val_df["time_local_dt"].max() < datetime(2026, 7, 1)
    assert test_df["time_local_dt"].min() >= datetime(2026, 7, 1)


def test_causal_site_profiles_ignore_future_partitions() -> None:
    df = _make_mock_features_df()
    train_df, _, test_df = temporal_split(df, "2026-06-01T00:00:00", "2026-07-01T00:00:00")

    profiles, fallbacks = compute_causal_site_profiles(train_df)
    enriched_test = enrich_split_with_profiles_and_physics(test_df, profiles, fallbacks)

    known_site = enriched_test.filter(pl.col("site_id") == "MG-SITE-001")
    unseen_site = enriched_test.filter(pl.col("site_id") == "MG-SITE-999")

    # Known site must retain Train mean (0.2), completely blind to Test (0.95)
    assert abs(float(known_site["site_cf_mean"][0]) - 0.2) < 1e-6
    assert abs(float(known_site["site_load_mean"][0]) - 1.0) < 1e-6
    # Unseen site falls back to global Train average without nulls
    assert abs(float(unseen_site["site_cf_mean"][0]) - fallbacks["site_cf_mean"]) < 1e-6
