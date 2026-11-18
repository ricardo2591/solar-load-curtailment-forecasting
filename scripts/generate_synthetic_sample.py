"""Generate a deterministic, physics-calibrated synthetic microgrid telemetry sample.

Produces strictly synthetic open-ocean coordinates (mid-Atlantic and mid-Pacific)
and anonymized `MG-SITE-###` identifiers so zero proprietary customer, operator,
or infrastructure data ever touches the repository.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pvlib

from microgrid_ml.config import load_config

# Open-ocean synthetic regional profiles: solar noon aligns with local 12:00
SYNTHETIC_ZONES = [
    ("Zone-Equatorial-Grid", "Etc/GMT+3", 6.5, -45.0, "Hybrid-Solar-Battery-Genset"),
    ("Zone-Tropical-Grid", "Etc/GMT+3", 15.0, -45.0, "Hybrid-Solar-Battery-Genset"),
    ("Zone-Subtropical-Grid", "Etc/GMT+9", -16.0, -135.0, "OffGrid-Solar-Battery"),
]


def _synthetic_weather_and_solar(
    utc_index: pd.DatetimeIndex,
    lat: float,
    lon: float,
    tilt_deg: int,
    azimuth_deg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    location = pvlib.location.Location(latitude=lat, longitude=lon)
    clearsky = location.get_clearsky(utc_index, model="ineichen")
    solpos = pvlib.solarposition.get_solarposition(utc_index, latitude=lat, longitude=lon, method="nrel_numpy")

    n = len(utc_index)
    # AR(1) cloud regime for temporal autocorrelation
    cloud = np.empty(n, dtype=np.float32)
    cloud[0] = rng.uniform(10.0, 55.0)
    noise = rng.normal(0.0, 8.5, size=n).astype(np.float32)
    base_cloud = rng.uniform(22.0, 48.0)
    for idx in range(1, n):
        cloud[idx] = np.clip(0.92 * cloud[idx - 1] + 0.08 * base_cloud + noise[idx], 0.0, 100.0)

    cloud_factor = np.clip(1.0 - 0.72 * (cloud / 100.0) ** 1.35, 0.12, 1.0).astype(np.float32)
    ghi = (clearsky["ghi"].to_numpy(dtype=np.float32) * cloud_factor).clip(min=0.0)
    dni = (clearsky["dni"].to_numpy(dtype=np.float32) * (cloud_factor ** 1.45)).clip(min=0.0)
    dhi = np.maximum(ghi - dni * np.maximum(np.cos(np.radians(solpos["apparent_zenith"].to_numpy(dtype=np.float32))), 0.0), 0.0)

    # Temperature diurnal cycle + slight cloud cooling
    utc_hours = utc_index.hour.to_numpy(dtype=np.float32) + utc_index.minute.to_numpy(dtype=np.float32) / 60.0
    solar_noon_offset_h = lon / 15.0
    local_h = (utc_hours + solar_noon_offset_h) % 24.0
    temp_c = (
        24.5
        + 4.8 * np.sin(2.0 * np.pi * (local_h - 9.0) / 24.0)
        - 2.2 * (cloud / 100.0)
        + rng.normal(0.0, 0.6, size=n)
    ).astype(np.float32)

    # Convective rain bursts when cloud cover is high
    rain_prob = np.where(cloud > 72.0, 0.085, 0.002)
    rain_events = rng.random(n) < rain_prob
    precip_mm = np.where(rain_events, rng.exponential(3.2, size=n), 0.0).astype(np.float32)

    # Effective tilted irradiance proxy for synthetic DC generation simulation
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solpos["apparent_zenith"].to_numpy(dtype=np.float32),
        solar_azimuth=solpos["azimuth"].to_numpy(dtype=np.float32),
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model="isotropic",
    )["poa_global"]
    return ghi, dni, dhi, cloud, temp_c, precip_mm, np.nan_to_num(poa, nan=0.0).astype(np.float32)


def generate_synthetic_dataset(output_path: Path | None = None) -> Path:
    """Create the anonymized synthetic 15-minute sample Parquet matching production schema."""
    cfg = load_config()
    sample_cfg = cfg.raw["synthetic_sample"]
    rng = np.random.default_rng(int(sample_cfg["seed"]))

    n_sites = int(sample_cfg["n_sites"])
    freq_min = int(sample_cfg["freq_minutes"])
    start_dt = datetime.fromisoformat(sample_cfg["start_date"])
    end_dt = datetime.fromisoformat(sample_cfg["end_date"])
    offline_rate = float(sample_cfg.get("offline_rate", 0.08))

    utc_index = pd.date_range(start_dt, end_dt, freq=f"{freq_min}min", tz="UTC")
    n_steps = len(utc_index)
    frames: list[pl.DataFrame] = []

    for site_idx in range(n_sites):
        zone_name, tz_name, base_lat, base_lon, topology = SYNTHETIC_ZONES[site_idx % len(SYNTHETIC_ZONES)]
        site_id = f"MG-SITE-{site_idx + 1:03d}"
        lat = round(base_lat + float(rng.uniform(-1.5, 1.5)), 3)
        lon = round(base_lon + float(rng.uniform(-2.0, 2.0)), 3)
        offset_hours = -3 if tz_name == "Etc/GMT+3" else -9
        local_dt = (utc_index + timedelta(hours=offset_hours)).tz_localize(None)

        capacity_kwp = round(float(rng.choice([12.6, 15.2, 18.0, 24.0, 30.0])), 2)
        tilt_deg = int(rng.choice([10, 15, 20]))
        azimuth_deg = int(180 if lat >= 0 else 0)
        cardinal = "South" if azimuth_deg == 180 else "North"
        # First site and ~half of fleet model high-curtailment rural low-load towers
        if site_idx % 2 == 0:
            base_load_kw = float(rng.uniform(0.45, 0.85))
        else:
            base_load_kw = float(rng.uniform(1.10, 2.40))
        battery_kwh = capacity_kwp * float(rng.uniform(1.2, 2.0))

        ghi, dni, dhi, cloud, temp_c, precip_mm, poa = _synthetic_weather_and_solar(
            utc_index, lat, lon, tilt_deg, azimuth_deg, rng
        )

        # Soiling accumulation reset by rain > 2.0 mm
        soiling_loss = np.empty(n_steps, dtype=np.float32)
        dry_days = 0.0
        for t in range(n_steps):
            if precip_mm[t] > 2.0:
                dry_days = 0.0
            else:
                dry_days += freq_min / 1440.0
            soiling_loss[t] = min(0.18, dry_days * 0.0022)

        # Telecom site load profile (24h diurnal traffic + ambient temp cooling)
        local_hours = local_dt.hour.to_numpy(dtype=np.float32) + local_dt.minute.to_numpy(dtype=np.float32) / 60.0
        traffic_curve = 0.18 * np.sin(2.0 * np.pi * (local_hours - 10.0) / 24.0)
        cooling_component = 0.03 * np.maximum(temp_c - 22.0, 0.0)
        load_kw = np.clip(
            base_load_kw * (1.0 + traffic_curve) + cooling_component + rng.normal(0.0, 0.08, size=n_steps),
            0.25,
            None,
        ).astype(np.float32)

        # Unconstrained PV generation with temperature derating + soiling
        temp_cell = temp_c + 0.03 * poa
        temp_derate = np.clip(1.0 - 0.004 * (temp_cell - 25.0), 0.72, 1.05)
        unconstrained_solar_kw = (
            capacity_kwp * (poa / 1000.0) * 0.86 * temp_derate * (1.0 - soiling_loss)
        ).clip(min=0.0).astype(np.float32)

        # Battery SoC and curtailment physics simulation
        soc = np.empty(n_steps, dtype=np.float32)
        solar_kw = np.empty(n_steps, dtype=np.float32)
        gen_kw = np.zeros(n_steps, dtype=np.float32)
        grid_kw = np.zeros(n_steps, dtype=np.float32)
        soc_curr = float(rng.uniform(75.0, 96.0)) if site_idx % 2 == 0 else float(rng.uniform(45.0, 85.0))
        dt_h = freq_min / 60.0

        for t in range(n_steps):
            pv = float(unconstrained_solar_kw[t])
            ld = float(load_kw[t])

            if soc_curr >= 95.0 and ld < 1.0 and poa[t] > 200.0:
                # Inverter curtails PV down to local load + small float charge
                pv_actual = min(pv, ld + 0.05 * capacity_kwp)
            else:
                pv_actual = pv

            net_kwh = (pv_actual - ld) * dt_h
            soc_next = soc_curr + (net_kwh / battery_kwh) * 100.0

            if soc_next < 22.0:
                if "Genset" in topology:
                    gen_kw[t] = max(ld * 1.4, 2.5)
                    soc_next += ((gen_kw[t] - ld) * dt_h / battery_kwh) * 100.0
                else:
                    grid_kw[t] = ld
                    soc_next = 25.0

            soc_curr = float(np.clip(soc_next, 15.0, 100.0))
            soc[t] = soc_curr
            noise_scale = 0.07 * max(pv_actual, 0.08)
            solar_kw[t] = max(0.0, pv_actual + float(rng.normal(0.0, noise_scale)))

        # Edge offline telemetry dropouts (mask realistic short intervals)
        offline_alarm = (rng.random(n_steps) < offline_rate).astype(np.int8)

        site_df = pl.DataFrame(
            {
                "timezone": [tz_name] * n_steps,
                "site_id": [site_id] * n_steps,
                "region_zone": [zone_name] * n_steps,
                "topology": [topology] * n_steps,
                "solar_var_type": ["hybrid_pv_total"] * n_steps,
                "cardinal_orientation": [cardinal] * n_steps,
                "latitude": np.full(n_steps, lat, dtype=np.float32),
                "longitude": np.full(n_steps, lon, dtype=np.float32),
                "tilt_deg": np.full(n_steps, tilt_deg, dtype=np.int16),
                "azimuth_deg": np.full(n_steps, azimuth_deg, dtype=np.int16),
                "solar_capacity_kwp": np.full(n_steps, capacity_kwp, dtype=np.float32),
                "solar_power_kw": solar_kw,
                "inverter_array_power_kw": (solar_kw * 0.65).astype(np.float32),
                "solar_charger_power_kw": (solar_kw * 0.35).astype(np.float32),
                "battery_soc_pct": soc,
                "battery_voltage_v": (48.0 + 0.06 * soc + rng.normal(0.0, 0.15, n_steps)).astype(np.float32),
                "battery_current_a": (((solar_kw - load_kw) * 1000.0) / 51.0).astype(np.float32),
                "load_power_kw": load_kw,
                "ac_load_power_kw": (load_kw * 0.25).astype(np.float32),
                "dc_load_power_kw": (load_kw * 0.75).astype(np.float32),
                "grid_power_kw": grid_kw,
                "generator_power_kw": gen_kw,
                "edge_offline_alarm": offline_alarm,
                "ambient_temp_c": temp_c,
                "cloud_cover_pct": cloud,
                "ghi_w_m2": ghi,
                "dni_w_m2": dni,
                "dhi_w_m2": dhi,
                "precipitation_mm": precip_mm,
                "time_utc_dt": utc_index.tz_convert("UTC").tz_localize(None).to_numpy(),
                "time_local_dt": local_dt.to_numpy(),
            }
        ).with_columns(
            [
                pl.col("timezone").cast(pl.Categorical),
                pl.col("site_id").cast(pl.Categorical),
                pl.col("region_zone").cast(pl.Categorical),
                pl.col("topology").cast(pl.Categorical),
                pl.col("solar_var_type").cast(pl.Categorical),
                pl.col("cardinal_orientation").cast(pl.Categorical),
                pl.col("time_utc_dt").dt.replace_time_zone("UTC"),
            ]
        )
        frames.append(site_df)

    full_df = pl.concat(frames)
    dst = output_path or cfg.sample_raw_parquet
    dst.parent.mkdir(parents=True, exist_ok=True)
    full_df.write_parquet(dst, compression="zstd")
    return dst


if __name__ == "__main__":
    out_file = generate_synthetic_dataset()
    size_mb = out_file.stat().st_size / (1024 ** 2)
    print(f"Generated synthetic sample dataset at {out_file} ({size_mb:.2f} MB)")
