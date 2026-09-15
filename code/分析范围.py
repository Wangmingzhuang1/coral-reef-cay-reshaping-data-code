"""统一管理科研分析中的样本排除，不删除原始数据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_EXCLUSION_CSV = Path(__file__).resolve().parent / "分析排除清单.csv"


ANALYSIS_CONTRACT_VERSION = "2026-09-11-s2only-interval-gate-2"
ANALYSIS_MODULES = {
    "directional_background": {
        "primary_sensor": "sentinel2",
        "interval_days": (90, 365),
        "sensitivity_interval_days": ((60, 365), (90, 365), (120, 365)),
        "primary_outcomes": ("along_shift", "cross_shift"),
        "secondary_outcomes": ("log_gross_mobility",),
        "inference_units": {
            "primary": "interval nested within sand cay",
            "robustness": "interval nested within reef",
        },
        "fdr_family": "outcome x specification",
    },
    "typhoon_event": {
        "primary_sensor": "sentinel2",
        "event_window_days": (-60, -5, 5, 60),
        "sensitivity_window_days": ((-90, -5, 5, 90), (-120, -5, 5, 120)),
        "primary_outcome": "gross_mobility_fraction",
        "inference_unit": "independent storm after cay-level matched differences",
        "fdr_family": "specification",
        "r34_definition": (
            "IBTrACS USA_R34 radius in the quadrant from the storm center "
            "toward the sand cay at the nearest track point"
        ),
    },
    "vegetation_state": {
        "primary_sensor": "sentinel2",
        "interval_days": (90, 730),
        "sensitivity_interval_days": ((90, 540), (90, 730), (120, 730)),
        "primary_outcome": "gross_mobility_fraction_per_year",
        "surface_state_variable": "green_vegetation_fraction",
        "transition_surface_variable": "dark_olive_anomaly_fraction",
        "transition_surface_rule": "spectral contrast with bright substrate; not crust, algae, wet sand, shadow, or a confirmed biological stage",
        "inference_units": {
            "between": "sand cay",
            "within": "interval within sand cay",
        },
        "fdr_family": "outcome x specification",
    },
}
TIDE_CONTRACT = {
    "model_name": "FES2022b",
    "variable": "ocean_tide_elevation",
    "grid": "ocean_tide_extrapolated",
    "prediction_library": "pyfes",
    "citation": (
        "The FES2022 Tide product was funded by CNES, produced by LEGOS, NOVELTIS "
        "and CLS and made freely available by AVISO. CNES, 2024. FES2022 (Finite "
        "Element Solution) Ocean Tide (Version 2024) [Data set]. CNES. "
        "https://doi.org/10.24400/527896/A01-2024.004"
    ),
    "url": "https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes/release-fes22.html",
    "use": "observation-condition sensitivity proxy, not a geomorphic mechanism variable",
    "required_fields": (
        "image_id",
        "sensor",
        "acquisition_datetime_utc",
        "sand_cay_center_lon",
        "sand_cay_center_lat",
        "tide_elevation_m",
        "tide_model",
        "tide_grid",
        "prediction_library_version",
        "prediction_status",
    ),
}
WIND_CONTRACT = {
    "model_name": "ERA5",
    "variables": ("u10", "v10"),
    "current_use": (
        "Partial ERA5 coverage may be used now in coverage-restricted exploratory "
        "and sensitivity analyses. Coverage must be reported by interval, cay, reef, "
        "sensor, and calendar period."
    ),
    "activation_rule": (
        "Inferential wind models require complete GLORYS, WAVERYS, and ERA5 coverage "
        "over the identical (time_t, time_t1] analytic window, with sufficient "
        "independent cays and reefs and no informative missing-coverage pattern. "
        "Tide remains paused and must not gate the current wind-wave-current analysis."
    ),
    "comparison": "flow_wave versus flow_wave_wind on the wind-complete subset",
    "likelihood_method": "maximum_likelihood",
}


def load_excluded_reefs(
    exclusion_csv: Path = DEFAULT_EXCLUSION_CSV,
    scope: str = "all",
) -> set[str]:
    if not exclusion_csv.is_file():
        return set()
    exclusions = pd.read_csv(exclusion_csv, dtype=str).fillna("")
    selected = exclusions["scope"].isin(["all", scope])
    return set(exclusions.loc[selected, "reef_id"].str.strip()) - {""}


def exclude_reefs(
    data: pd.DataFrame,
    scope: str = "all",
    reef_column: str = "reef_id",
) -> pd.DataFrame:
    excluded = load_excluded_reefs(scope=scope)
    if not excluded or reef_column not in data.columns:
        return data.copy()
    return data.loc[~data[reef_column].astype("string").isin(excluded)].copy()


def module_contract(module: str) -> dict:
    if module not in ANALYSIS_MODULES:
        raise KeyError(module)
    return ANALYSIS_MODULES[module]


def analysis_contract_digest() -> str:
    payload = {
        "contract_version": ANALYSIS_CONTRACT_VERSION,
        "modules": ANALYSIS_MODULES,
        "tide": TIDE_CONTRACT,
        "wind": WIND_CONTRACT,
        "excluded_reefs": sorted(load_excluded_reefs()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_contract_audit(output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "分析合同核查.json"
    payload = {
        "contract_version": ANALYSIS_CONTRACT_VERSION,
        "contract_sha256": analysis_contract_digest(),
        "excluded_reefs": sorted(load_excluded_reefs()),
        "modules": ANALYSIS_MODULES,
        "tide": TIDE_CONTRACT,
        "wind": WIND_CONTRACT,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
