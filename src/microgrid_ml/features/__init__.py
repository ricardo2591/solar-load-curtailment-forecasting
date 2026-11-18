"""Solar astronomy, irradiance transposition, and causal feature engineering."""

from microgrid_ml.features.solar_physics import (
    build_temporal_and_physical_features,
    compute_causal_site_profiles,
    compute_solar_geometry_and_poa,
    enrich_split_with_profiles_and_physics,
)

__all__ = [
    "build_temporal_and_physical_features",
    "compute_causal_site_profiles",
    "compute_solar_geometry_and_poa",
    "enrich_split_with_profiles_and_physics",
]
