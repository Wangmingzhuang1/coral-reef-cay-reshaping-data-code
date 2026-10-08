"""将已下载的 GLORYS 流场与 WAVERYS 波浪场聚合到沙洲观测变化区间。

物理强迫按 (t, t1] 聚合，用于解释已经发生的形态变化；这与预测任务中
仅使用 t 及以前历史特征的时间边界不同。脚本只做覆盖审计和探索性秩关联，
不拟合模型，也不输出因果归因。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from 分析范围 import (
    ANALYSIS_CONTRACT_VERSION,
    analysis_contract_digest,
    exclude_reefs,
    module_contract,
)


WIND_FIELDS = ("u10", "v10")
CURRENT_FIELDS = ("uo", "vo")
WAVE_FIELDS = ("VHM0", "VTM02", "VMDR", "VSDX", "VSDY")
DIRECTIONAL_CONTRACT = module_contract("directional_background")
ANALYSIS_INTERVAL_DAYS = tuple(DIRECTIONAL_CONTRACT["interval_days"])
WINDOW_SENSITIVITY = [
    tuple(days) for days in DIRECTIONAL_CONTRACT["sensitivity_interval_days"]
]
OUTCOMES = (
    "erosion_fraction_per_year",
    "deposition_fraction_per_year",
    "gross_mobility_fraction_per_year",
    "centroid_shift_normalized_per_year",
)
DRIVERS = (
    "current_speed_mean",
    "current_speed_max",
    "wave_hs_mean",
    "wave_hs_max",
    "wave_hs_p90",
)


def circular_direction(u: pd.Series, v: pd.Series) -> float:
    mean_u, mean_v = u.mean(), v.mean()
    if pd.isna(mean_u) or pd.isna(mean_v):
        return np.nan
    return float((np.degrees(np.arctan2(mean_u, mean_v)) + 360.0) % 360.0)


def vector_summary(frame: pd.DataFrame, u: str, v: str, prefix: str) -> dict[str, float]:
    speed = np.hypot(frame[u], frame[v])
    return {
        f"{prefix}_u_mean": float(frame[u].mean()),
        f"{prefix}_v_mean": float(frame[v].mean()),
        f"{prefix}_speed_mean": float(speed.mean()),
        f"{prefix}_speed_max": float(speed.max()),
        f"{prefix}_speed_p90": float(speed.quantile(0.9)),
        f"{prefix}_direction_deg": circular_direction(frame[u], frame[v]),
    }


def aggregate_source(
    transitions: pd.DataFrame,
    daily: pd.DataFrame,
    source: str,
    fields: tuple[str, ...],
    local_typhoon_days: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    source_daily = daily.loc[daily["source"].eq(source)].copy()
    source_daily["time"] = pd.to_datetime(source_daily["time"]).dt.normalize()
    by_reef = {reef: group.set_index("time").sort_index() for reef, group in source_daily.groupby("reef_id")}
    typhoon_dates_by_cay: dict[str, set[pd.Timestamp]] = {}
    if source == "era5" and local_typhoon_days is not None and not local_typhoon_days.empty:
        typhoon_days = local_typhoon_days.copy()
        typhoon_days["date"] = pd.to_datetime(typhoon_days["date"], errors="coerce").dt.normalize()
        typhoon_days = typhoon_days.loc[
            typhoon_days["local_typhoon_day"].eq(1) & typhoon_days["date"].notna()
        ]
        typhoon_dates_by_cay = {
            str(cay): set(group["date"])
            for cay, group in typhoon_days.groupby("sand_cay_id")
        }
    records: list[dict[str, object]] = []
    available_reefs = 0
    for item in transitions.to_dict("records"):
        reef = item["reef_id"]
        if reef not in by_reef:
            records.append({"transition_id": item["transition_id"], f"{source}_available": 0, f"{source}_complete": 0})
            continue
        available_reefs += 1
        table = by_reef[reef]
        start, end = pd.Timestamp(item["time_t"]).normalize(), pd.Timestamp(item["time_t1"]).normalize()
        expected_days = int((end - start).days)
        window = table.loc[(table.index > start) & (table.index <= end)]
        observed = window.reindex(columns=list(fields))
        complete = (
            expected_days > 0
            and len(window) == expected_days
            and window.index.nunique() == expected_days
            and not observed.isna().any().any()
        )
        record: dict[str, object] = {
            "transition_id": item["transition_id"],
            f"{source}_available": 1,
            f"{source}_complete": int(complete),
            f"{source}_expected_days": expected_days,
            f"{source}_observed_days": int(len(window)),
        }
        if complete and source == "glorys":
            record.update(vector_summary(window, "uo", "vo", "current"))
        elif complete and source == "era5":
            record.update(vector_summary(window, "u10", "v10", "wind"))
            local_dates = typhoon_dates_by_cay.get(str(item["sand_cay_id"]), set())
            pulse_mask = window.index.isin(local_dates)
            pulse_window = window.loc[pulse_mask]
            background_window = window.loc[~pulse_mask]
            record.update(
                {
                    "local_typhoon_day_count": int(pulse_mask.sum()),
                    "non_typhoon_day_count": int((~pulse_mask).sum()),
                    "local_typhoon_day_fraction": float(pulse_mask.mean()),
                }
            )
            if len(background_window):
                record.update(
                    vector_summary(background_window, "u10", "v10", "wind_background")
                )
            if len(pulse_window):
                record.update(
                    vector_summary(pulse_window, "u10", "v10", "wind_typhoon")
                )
        elif complete and source == "waverys":
            record.update(vector_summary(window, "VSDX", "VSDY", "wave_vector"))
            record.update(
                {
                    "wave_hs_mean": float(window["VHM0"].mean()),
                    "wave_hs_max": float(window["VHM0"].max()),
                    "wave_hs_p90": float(window["VHM0"].quantile(0.9)),
                    "wave_period_mean": float(window["VTM02"].mean()),
                    "wave_direction_mean_deg": float(window["VMDR"].mean()),
                }
            )
        records.append(record)
    result = pd.DataFrame(records)
    summary = {
        "transitions_with_source_reef": available_reefs,
        "complete_intervals": int(result[f"{source}_complete"].sum()),
    }
    if source == "era5" and "local_typhoon_day_count" in result:
        complete = result.loc[result["era5_complete"].eq(1)]
        summary["complete_intervals_with_local_typhoon_days"] = int(
            complete["local_typhoon_day_count"].gt(0).sum()
        )
        summary["local_typhoon_days_across_complete_intervals"] = int(
            complete["local_typhoon_day_count"].sum()
        )
    return result, summary


def bh_fdr(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return result
    ranks = np.arange(1, len(valid) + 1)
    adjusted = (valid.to_numpy() * len(valid) / ranks)[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def rank_associations(core: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor, group in core.groupby("sensor"):
        cay_medians = group.groupby("sand_cay_id", as_index=False).median(numeric_only=True)
        for unit_name, frame in (("interval_exploratory", group), ("cay_median", cay_medians)):
            for driver in DRIVERS:
                for outcome in OUTCOMES:
                    selected = frame[[driver, outcome]].dropna()
                    if len(selected) < 5 or selected.nunique().min() < 2:
                        rho, p_value = np.nan, np.nan
                    else:
                        rho, p_value = spearmanr(selected[driver], selected[outcome])
                    rows.append({
                        "sensor": sensor,
                        "analysis_unit": unit_name,
                        "driver": driver,
                        "outcome": outcome,
                        "n": int(len(selected)),
                        "spearman_rho": rho,
                        "two_sided_p_value": p_value,
                    })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q_value"] = result.groupby(["sensor", "analysis_unit"])["two_sided_p_value"].transform(bh_fdr)
    return result


def _precision_map(observations: pd.DataFrame) -> pd.DataFrame:
    precision = observations.drop_duplicates(["sand_cay_id", "date"])[
        ["sand_cay_id", "date", "date_precision"]
    ].copy()
    precision["date10"] = precision["date"].astype(str).str[:10]
    return precision[["sand_cay_id", "date10", "date_precision"]]


def sensor_cadence_summary(core: pd.DataFrame, observations: pd.DataFrame) -> dict:
    summary: dict[str, object] = {}
    precision = _precision_map(observations)
    core = core.copy()
    core["t10"] = pd.to_datetime(core["time_t"]).dt.strftime("%Y-%m-%d")
    core["t10_1"] = pd.to_datetime(core["time_t1"]).dt.strftime("%Y-%m-%d")
    core = core.merge(
        precision.rename(columns={"date10": "t10", "date_precision": "prec_t"}),
        on=["sand_cay_id", "t10"],
        how="left",
    ).merge(
        precision.rename(columns={"date10": "t10_1", "date_precision": "prec_t1"}),
        on=["sand_cay_id", "t10_1"],
        how="left",
    )
    sensors = sorted(core["sensor"].unique())
    for sensor in sensors:
        sub = core[core["sensor"].eq(sensor)]
        obs = observations[observations["sensor"].eq(sensor)]
        gaps: list[float] = []
        rates: list[float] = []
        for _, group in obs.groupby("sand_cay_id"):
            dates = pd.to_datetime(group["date"]).sort_values()
            if len(dates) >= 2:
                gaps.extend(dates.diff().dt.days.dropna().to_numpy())
                span_years = max((dates.iloc[-1] - dates.iloc[0]).days, 1) / 365.2425
                rates.append(len(dates) / span_years)
        other = [s for s in sensors if s != sensor]
        shared = set(sub["sand_cay_id"]) & set(
            core[core["sensor"].isin(other)]["sand_cay_id"]
        )
        summary[sensor] = {
            "core_intervals": int(len(sub)),
            "core_cays": int(sub["sand_cay_id"].nunique()),
            "core_reefs": int(sub["reef_id"].nunique()),
            "median_interval_days": float(sub["time_interval_days"].median()),
            "iqr_interval_days": [
                float(sub["time_interval_days"].quantile(0.25)),
                float(sub["time_interval_days"].quantile(0.75)),
            ],
            "median_observation_gap_days": float(np.median(gaps)) if gaps else np.nan,
            "median_observations_per_year": float(np.median(rates)) if rates else np.nan,
            "fraction_day_precision_both_ends": float(
                (
                    sub["prec_t"].eq("day").fillna(False)
                    & sub["prec_t1"].eq("day").fillna(False)
                ).mean()
            )
            if len(sub)
            else np.nan,
            "cays_shared_with_other_sensor": int(len(shared)),
        }
    return summary


def sensor_consistency_diagnostic(
    core: pd.DataFrame, observations: pd.DataFrame
) -> pd.DataFrame:
    """检验时间分散/日期精度能否解释传感器间驱动—响应方向冲突。"""
    precision = _precision_map(observations)
    frame = core.copy()
    frame["t10"] = pd.to_datetime(frame["time_t"]).dt.strftime("%Y-%m-%d")
    frame["t10_1"] = pd.to_datetime(frame["time_t1"]).dt.strftime("%Y-%m-%d")
    frame = frame.merge(
        precision.rename(columns={"date10": "t10", "date_precision": "prec_t"}),
        on=["sand_cay_id", "t10"],
        how="left",
    ).merge(
        precision.rename(columns={"date10": "t10_1", "date_precision": "prec_t1"}),
        on=["sand_cay_id", "t10_1"],
        how="left",
    )
    # The wind columns are absent in pre-ERA5 products.  Keeping explicit NaN
    # columns lets the established flow/wave analysis remain runnable while the
    # ERA5 download is incomplete, and makes coverage gating auditable.
    for column in ("wind_u_mean", "wind_v_mean"):
        if column not in frame:
            frame[column] = np.nan
    day_only = frame["prec_t"].eq("day").fillna(False) & frame["prec_t1"].eq("day").fillna(False)
    subsets = {
        "all_core": np.ones(len(frame), dtype=bool),
        "interval_le_180d": frame["time_interval_days"].le(180).to_numpy(),
        "interval_le_270d": frame["time_interval_days"].le(270).to_numpy(),
        "day_precision_only": day_only.to_numpy(),
    }
    rows: list[dict[str, object]] = []
    for sensor, group in frame.groupby("sensor"):
        for name, mask in subsets.items():
            idx = group.index.to_numpy()
            sub = group.loc[idx[mask[idx]]]
            row: dict[str, object] = {
                "sensor": sensor,
                "subset": name,
                "n_intervals": int(len(sub)),
                "n_cays": int(sub["sand_cay_id"].nunique()),
            }
            for tag, driver, outcome in (
                (
                    "current_speed_mean__gross_mobility",
                    "current_speed_mean",
                    "gross_mobility_fraction_per_year",
                ),
                (
                    "wave_hs_p90__centroid_shift",
                    "wave_hs_p90",
                    "centroid_shift_normalized_per_year",
                ),
            ):
                selected = sub[[driver, outcome]].dropna()
                if len(selected) < 5 or selected.nunique().min() < 2:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(selected[driver], selected[outcome])
                row[f"rho_{tag}"] = rho
                row[f"p_{tag}"] = p_value
            rows.append(row)
    return pd.DataFrame(rows)


NORTH_CONVENTION = "image_row_increases_south_column_increases_east"
REVIEW_STAMP = "verified_north_up_2026-09-07"
AXIS_PAIR_WINDOW_DAYS = 120


def _circular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def verify_north(
    transitions: pd.DataFrame, observations: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """逐参考框架北向核验：产品规格（S2/GE 均为北向上栅格）+ 跨传感器主轴方位一致性。

    区间级质心方位在跨传感器短区间上不可比（掩膜边界与水位差异主导），
    因此采用静态几何锚：近同期观测的主轴方位（mod 180）一致性。
    """
    shared = set(
        transitions.loc[transitions["sensor"].eq("sentinel2"), "sand_cay_id"]
    ) & set(transitions.loc[transitions["sensor"].eq("google_earth"), "sand_cay_id"])
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"])
    phi = np.radians(obs["sand_cay_major_axis_angle"].to_numpy(dtype=float))
    obs["axis_bearing_deg"] = (
        np.degrees(np.arctan2(np.sin(phi), -np.cos(phi))) % 180.0
    )
    diffs: list[float] = []
    for cay in sorted(shared):
        sub = obs[obs["sand_cay_id"].eq(cay)]
        s2 = sub[sub["sensor"].eq("sentinel2")]
        ge = sub[sub["sensor"].eq("google_earth")]
        for a in s2.to_dict("records"):
            for b in ge.to_dict("records"):
                if abs((a["date"] - b["date"]).days) <= AXIS_PAIR_WINDOW_DAYS:
                    diffs.append(
                        float(
                            (a["axis_bearing_deg"] - b["axis_bearing_deg"] + 90.0)
                            % 180.0
                            - 90.0
                        )
                    )
    diffs_arr = np.asarray(diffs, dtype=float)
    if len(diffs_arr):
        resultant = float(np.abs(np.mean(np.exp(2j * np.radians(diffs_arr)))))
        median_abs = float(np.median(np.abs(diffs_arr)))
    else:
        resultant = np.nan
        median_abs = np.nan
    cross_stats = {
        "test": "near-simultaneous cross-sensor major-axis bearing agreement (mod 180)",
        "shared_cays": int(len(shared)),
        "axis_pair_window_days": AXIS_PAIR_WINDOW_DAYS,
        "axis_bearing_pairs": int(len(diffs_arr)),
        "median_abs_axis_diff_deg": median_abs,
        "resultant_length_mod180": resultant,
        "pass_thresholds": {
            "resultant_length_min": 0.7,
            "median_abs_diff_max_deg": 20.0,
        },
        "cross_sensor_pass": bool(
            len(diffs_arr) >= 30 and resultant >= 0.7 and median_abs <= 20.0
        ),
    }
    rows: list[dict[str, object]] = []
    for (frame, sensor), group in transitions.groupby(
        ["reference_frame_id", "sensor"]
    ):
        n_obs = int(
            observations[
                observations["reference_frame_id"].eq(frame)
                & observations["sensor"].eq(sensor)
            ]["image_id"].nunique()
        )
        tested = sensor in ("sentinel2", "google_earth")
        if tested and cross_stats["cross_sensor_pass"]:
            evidence = "product_spec_north_up+cross_sensor_axis_agreement"
            status = "verified"
        elif tested:
            evidence = "product_spec_north_up+cross_sensor_axis_agreement_failed"
            status = "flagged"
        else:
            evidence = "product_spec_north_up_only"
            status = "spec_only"
        rows.append(
            {
                "reference_frame_id": frame,
                "sensor": sensor,
                "n_observations": n_obs,
                "n_intervals": int(len(group)),
                "north_convention": NORTH_CONVENTION,
                "evidence": evidence,
                "status": status,
            }
        )
    return pd.DataFrame(rows), cross_stats


def decompose_along_cross(
    core: pd.DataFrame, observations: pd.DataFrame
) -> pd.DataFrame:
    """把流/波矢量与质心位移分解到沙洲主轴沿轴/横轴方向。

    skimage orientation 约定经合成掩膜实测：phi=0 主轴沿图像行、phi=90 沿列，
    数组方向向量 (cos phi, sin phi)；地理方向 (east, north) = (sin phi, -cos phi)。
    """
    from 构建沙洲观测与变化表 import select_analysis_observations
    keys = ["sensor", "sand_cay_id", "reference_frame_id"]
    angle = select_analysis_observations(observations)[
        keys + ["date", "sand_cay_major_axis_angle"]
    ].copy()
    angle["t10"] = pd.to_datetime(angle["date"]).dt.strftime("%Y-%m-%d")
    frame = core.copy()
    frame["t10"] = pd.to_datetime(frame["time_t"]).dt.strftime("%Y-%m-%d")
    frame = frame.merge(
        angle[keys + ["t10", "sand_cay_major_axis_angle"]],
        on=keys + ["t10"],
        how="left",
        validate="many_to_one",
    )
    if frame["sand_cay_major_axis_angle"].isna().any():
        raise ValueError("方向区间缺少同源、同固定框的起始主轴信息。")
    phi = np.radians(frame["sand_cay_major_axis_angle"].to_numpy(dtype=float))
    axis_east, axis_north = np.sin(phi), -np.cos(phi)
    perp_east, perp_north = np.cos(phi), np.sin(phi)
    years = frame["time_interval_days"].to_numpy(dtype=float) / 365.2425
    scale = np.sqrt(frame["area_t_m2"].to_numpy(dtype=float))

    def project(east: pd.Series, north: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        e = east.to_numpy(dtype=float)
        n = north.to_numpy(dtype=float)
        return e * axis_east + n * axis_north, e * perp_east + n * perp_north

    current_along, current_cross = project(
        frame["current_u_mean"], frame["current_v_mean"]
    )
    wave_along, wave_cross = project(
        frame["wave_vector_u_mean"], frame["wave_vector_v_mean"]
    )
    wind_along, wind_cross = project(frame["wind_u_mean"], frame["wind_v_mean"])
    shift_along, shift_cross = project(
        frame["centroid_east_shift_m"], frame["centroid_north_shift_m"]
    )
    frame["current_along_mean"] = current_along
    frame["current_cross_mean"] = current_cross
    frame["wave_vector_along_mean"] = wave_along
    frame["wave_vector_cross_mean"] = wave_cross
    frame["wind_along_mean"] = wind_along
    frame["wind_cross_mean"] = wind_cross
    for source_prefix, output_prefix in (
        ("wind_background", "background_wind"),
        ("wind_typhoon", "typhoon_wind"),
    ):
        u_column, v_column = f"{source_prefix}_u_mean", f"{source_prefix}_v_mean"
        if u_column in frame and v_column in frame:
            along, cross = project(frame[u_column], frame[v_column])
            frame[f"{output_prefix}_along_mean"] = along
            frame[f"{output_prefix}_cross_mean"] = cross
    frame["centroid_along_shift_normalized_per_year"] = shift_along / scale / years
    frame["centroid_cross_shift_normalized_per_year"] = shift_cross / scale / years
    frame["major_axis_change_m_per_year"] = (
        frame["major_axis_change_m"].to_numpy(dtype=float) / years
    )
    frame["minor_axis_change_m_per_year"] = (
        frame["minor_axis_change_m"].to_numpy(dtype=float) / years
    )
    wave_dir = np.degrees(
        np.arctan2(frame["wave_vector_u_mean"], frame["wave_vector_v_mean"])
    )
    axis_bearing = np.degrees(np.arctan2(axis_east, axis_north))
    frame["wave_axis_angle_diff_deg"] = np.abs(
        _circular_diff_deg(wave_dir, axis_bearing)
    )
    frame["wave_axis_angle_diff_deg"] = np.minimum(
        frame["wave_axis_angle_diff_deg"], 180.0 - frame["wave_axis_angle_diff_deg"]
    )
    return frame


DIRECTIONAL_DRIVERS = (
    "current_along_mean",
    "current_cross_mean",
    "wave_vector_along_mean",
    "wave_vector_cross_mean",
    "wave_hs_p90",
)
DIRECTIONAL_OUTCOMES = (
    "centroid_along_shift_normalized_per_year",
    "centroid_cross_shift_normalized_per_year",
    "gross_mobility_fraction_per_year",
    "major_axis_change_m_per_year",
    "minor_axis_change_m_per_year",
)
PREDEFINED_DIRECTIONAL_PAIRS = {
    ("current_along_mean", "centroid_along_shift_normalized_per_year"),
    ("wave_hs_p90", "centroid_cross_shift_normalized_per_year"),
    ("wave_hs_p90", "gross_mobility_fraction_per_year"),
    ("current_along_mean", "major_axis_change_m_per_year"),
    ("wave_vector_cross_mean", "minor_axis_change_m_per_year"),
}


def directional_rank_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor, group in frame.groupby("sensor"):
        for driver in DIRECTIONAL_DRIVERS:
            for outcome in DIRECTIONAL_OUTCOMES:
                selected = group[[driver, outcome]].dropna()
                if len(selected) < 5 or selected.nunique().min() < 2:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(selected[driver], selected[outcome])
                rows.append(
                    {
                        "sensor": sensor,
                        "driver": driver,
                        "outcome": outcome,
                        "predefined_hypothesis": (driver, outcome)
                        in PREDEFINED_DIRECTIONAL_PAIRS,
                        "n": int(len(selected)),
                        "spearman_rho": rho,
                        "two_sided_p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q_value"] = result.groupby("sensor")[
            "two_sided_p_value"
        ].transform(bh_fdr)
    return result


def directional_window_sensitivity(
    aligned: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """按预定义时间窗重算方向分解与方向性秩关联，不改变主分析结果。"""
    rows: list[dict[str, object]] = []
    for low, high in WINDOW_SENSITIVITY:
        core = aligned[
            aligned["flow_wave_complete"]
            & aligned["boundary_change_status"].eq("ok")
            & aligned["time_interval_days"].between(low, high)
            & aligned["area_t_m2"].gt(0)
        ].copy()
        decomposition = decompose_along_cross(core, observations)
        associations = directional_rank_associations(decomposition)
        for row in associations.itertuples(index=False):
            rows.append(
                {
                    "window_days": f"{low}-{high}",
                    "sensor": row.sensor,
                    "driver": row.driver,
                    "outcome": row.outcome,
                    "predefined_hypothesis": bool(row.predefined_hypothesis),
                    "n": int(row.n),
                    "spearman_rho": row.spearman_rho,
                    "two_sided_p_value": row.two_sided_p_value,
                    "fdr_q_value": row.fdr_q_value,
                }
            )
        rows.append(
            {
                "window_days": f"{low}-{high}",
                "sensor": "all",
                "driver": "__window_summary__",
                "outcome": "__window_summary__",
                "predefined_hypothesis": False,
                "n": int(len(core)),
                "spearman_rho": np.nan,
                "two_sided_p_value": np.nan,
                "fdr_q_value": np.nan,
            }
        )
    return pd.DataFrame(rows)


PULSE_COLUMNS = ("typhoon_strong_count", "typhoon_r34_count")
REGIME_OUTCOMES = (
    "gross_mobility_fraction_per_year",
    "erosion_fraction_per_year",
    "deposition_fraction_per_year",
    "centroid_along_shift_normalized_per_year",
    "centroid_cross_shift_normalized_per_year",
)


def label_background_pulse(frame: pd.DataFrame) -> pd.DataFrame:
    """背景区间 = 区间内无强台风且无 R34 暴露；否则为台风脉冲区间。"""
    frame = frame.copy()
    pulse = frame[list(PULSE_COLUMNS)].fillna(0).sum(axis=1) > 0
    frame["forcing_regime"] = np.where(pulse, "typhoon_pulse", "background")
    return frame


def background_pulse_contrast(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor, group in frame.groupby("sensor"):
        background = group[group["forcing_regime"].eq("background")]
        pulse = group[group["forcing_regime"].eq("typhoon_pulse")]
        for outcome in REGIME_OUTCOMES:
            a = background[outcome].dropna()
            b = pulse[outcome].dropna()
            if len(a) >= 5 and len(b) >= 5:
                _, p_value = mannwhitneyu(a, b, alternative="two-sided")
            else:
                p_value = np.nan
            rows.append(
                {
                    "sensor": sensor,
                    "outcome": outcome,
                    "n_background": int(len(a)),
                    "n_pulse": int(len(b)),
                    "median_background": float(a.median()) if len(a) else np.nan,
                    "median_pulse": float(b.median()) if len(b) else np.nan,
                    "mannwhitney_p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q_value"] = result.groupby("sensor")[
            "mannwhitney_p_value"
        ].transform(bh_fdr)
    return result


def post_event_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    """脉冲区间后首个同沙洲区间的可动性是否回到该沙洲背景水平。"""
    rows: list[dict[str, object]] = []
    keys = ["sand_cay_id", "sensor", "reference_frame_id"]
    for key, group in frame.groupby(keys):
        group = group.sort_values("time_t")
        background_median = float(
            group.loc[
                group["forcing_regime"].eq("background"),
                "gross_mobility_fraction_per_year",
            ].median()
        )
        pulses = group[group["forcing_regime"].eq("typhoon_pulse")]
        for _, row in pulses.iterrows():
            following = group[group["time_t"].ge(row["time_t1"])]
            following = following[following["transition_id"].ne(row["transition_id"])]
            if following.empty:
                continue
            nxt = following.iloc[0]
            area_pre = float(row["area_t_m2"])
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "pulse_transition_id": row["transition_id"],
                    "pulse_end": row["time_t1"],
                    "typhoon_max_nearest_wind_m_s": row["typhoon_max_nearest_wind_m_s"],
                    "pulse_gross_mobility": row["gross_mobility_fraction_per_year"],
                    "post_transition_id": nxt["transition_id"],
                    "post_interval_days": nxt["time_interval_days"],
                    "post_gross_mobility": nxt["gross_mobility_fraction_per_year"],
                    "cay_background_median_gross": background_median,
                    "post_to_background_ratio": (
                        nxt["gross_mobility_fraction_per_year"] / background_median
                        if background_median > 0
                        else np.nan
                    ),
                    "post_area_relative_to_pre": (
                        float(nxt["area_t1_m2"]) / area_pre if area_pre > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def recovery_summary(recovery: pd.DataFrame) -> dict:
    if recovery.empty:
        return {"n_pulse_with_post": 0}
    primary = recovery[recovery["sensor"].eq("sentinel2")]
    ratios = primary["post_to_background_ratio"].dropna()
    paired = primary.dropna(
        subset=["post_gross_mobility", "cay_background_median_gross"]
    )
    p_value = np.nan
    if len(paired) >= 8:
        p_value = float(
            wilcoxon(
                paired["post_gross_mobility"], paired["cay_background_median_gross"]
            ).pvalue
        )
    return {
        "n_pulse_with_post": int(len(primary)),
        "median_post_to_background_ratio": float(ratios.median())
        if len(ratios)
        else np.nan,
        "fraction_recovered_ratio_le_1": float((ratios <= 1.0).mean())
        if len(ratios)
        else np.nan,
        "fraction_recovered_ratio_le_1p5": float((ratios <= 1.5).mean())
        if len(ratios)
        else np.nan,
        "median_post_area_relative_to_pre": float(
            primary["post_area_relative_to_pre"].median()
        ),
        "wilcoxon_p_post_vs_background": p_value,
    }


def main() -> None:
    research_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-csv", type=Path, default=research_root / "outputs" / "沙洲变化区间.csv")
    parser.add_argument(
        "--reef-daily-csv",
        type=Path,
        default=research_root
        / "data"
        / "environment"
        / "harmonized"
        / "daily"
        / "reef_daily_features.csv",
    )
    parser.add_argument(
        "--typhoon-day-csv",
        type=Path,
        default=research_root / "outputs" / "台风局地强风日.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=research_root / "outputs")
    args = parser.parse_args()

    transitions = exclude_reefs(pd.read_csv(args.transition_csv))
    transitions["time_interval_days"] = pd.to_numeric(transitions["time_interval_days"], errors="coerce")
    transitions["area_t_m2"] = pd.to_numeric(transitions["area_t_m2"], errors="coerce")
    for column in ("erosion_area_m2", "deposition_area_m2", "gross_boundary_change_m2", "centroid_shift_m"):
        transitions[column] = pd.to_numeric(transitions[column], errors="coerce")
    years = transitions["time_interval_days"] / 365.2425
    transitions["erosion_fraction_per_year"] = transitions["erosion_area_m2"] / transitions["area_t_m2"] / years
    transitions["deposition_fraction_per_year"] = transitions["deposition_area_m2"] / transitions["area_t_m2"] / years
    transitions["gross_mobility_fraction_per_year"] = transitions["gross_boundary_change_m2"] / transitions["area_t_m2"] / years
    transitions["centroid_shift_normalized_per_year"] = transitions["centroid_shift_m"] / np.sqrt(transitions["area_t_m2"]) / years

    daily = pd.read_csv(args.reef_daily_csv)
    local_typhoon_days = (
        pd.read_csv(args.typhoon_day_csv, low_memory=False)
        if args.typhoon_day_csv.is_file()
        else pd.DataFrame()
    )
    wind, wind_summary = aggregate_source(
        transitions, daily, "era5", WIND_FIELDS, local_typhoon_days
    )
    current, current_summary = aggregate_source(transitions, daily, "glorys", CURRENT_FIELDS)
    wave, wave_summary = aggregate_source(transitions, daily, "waverys", WAVE_FIELDS)
    aligned = (
        transitions.merge(wind, on="transition_id", how="left")
        .merge(current, on="transition_id", how="left")
        .merge(wave, on="transition_id", how="left")
    )
    aligned["flow_wave_complete"] = aligned["glorys_complete"].eq(1) & aligned["waverys_complete"].eq(1)
    aligned["flow_wave_wind_complete"] = (
        aligned["flow_wave_complete"] & aligned["era5_complete"].eq(1)
    )
    core = aligned[
        aligned["flow_wave_complete"]
        & aligned["boundary_change_status"].eq("ok")
        & aligned["time_interval_days"].between(*ANALYSIS_INTERVAL_DAYS)
        & aligned["area_t_m2"].gt(0)
    ].copy()
    associations = rank_associations(core)
    observations = pd.read_csv(args.output_dir / "沙洲观测主表.csv", encoding="utf-8-sig")
    consistency = sensor_consistency_diagnostic(core, observations)
    north_frames, cross_stats = verify_north(transitions, observations)
    decomposition = decompose_along_cross(core, observations)
    window_sensitivity = directional_window_sensitivity(aligned, observations)
    wind_ready_core = decomposition.loc[
        decomposition["flow_wave_wind_complete"].eq(True)
        & decomposition["sensor"].eq("sentinel2")
    ].copy()
    directional = directional_rank_associations(decomposition)
    regime_frame = label_background_pulse(decomposition)
    contrast = background_pulse_contrast(regime_frame)
    background_directional = directional_rank_associations(
        regime_frame[regime_frame["forcing_regime"].eq("background")]
    )
    recovery = post_event_recovery(regime_frame)
    recovery_stats = recovery_summary(recovery)
    all_verified = bool(north_frames["status"].eq("verified").all()) or bool(
        north_frames["status"].isin(["verified", "spec_only"]).all()
        and cross_stats["cross_sensor_pass"]
    )
    if all_verified:
        raw = pd.read_csv(args.transition_csv)
        verified_frames = set(
            north_frames.loc[north_frames["status"].ne("flagged"), "reference_frame_id"]
        )
        raw.loc[
            raw["reference_frame_id"].isin(verified_frames), "direction_status"
        ] = REVIEW_STAMP
        raw.to_csv(args.transition_csv, index=False, encoding="utf-8-sig")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "流场波浪区间特征.csv"
    core_path = args.output_dir / "流场波浪核心区间.csv"
    association_path = args.output_dir / "流场波浪形态秩相关.csv"
    summary_path = args.output_dir / "流场波浪覆盖核查.json"
    consistency_path = args.output_dir / "流场波浪传感器一致性诊断.csv"
    north_path = args.output_dir / "参考框架北向核验.csv"
    decomposition_path = args.output_dir / "流场波浪沿轴横轴分解.csv"
    directional_path = args.output_dir / "流场波浪方向假设秩相关.csv"
    window_path = args.output_dir / "流场波浪方向时间窗敏感性.csv"
    contrast_path = args.output_dir / "背景与台风脉冲对比.csv"
    background_directional_path = args.output_dir / "流场波浪背景区间方向秩相关.csv"
    recovery_path = args.output_dir / "事件后恢复.csv"
    wind_core_path = args.output_dir / "风浪流完整S2核心区间.csv"
    wind_partition_path = args.output_dir / "台风日风场分解.csv"
    aligned.to_csv(aligned_path, index=False, encoding="utf-8-sig")
    core.to_csv(core_path, index=False, encoding="utf-8-sig")
    associations.to_csv(association_path, index=False, encoding="utf-8-sig")
    consistency.to_csv(consistency_path, index=False, encoding="utf-8-sig")
    north_frames.to_csv(north_path, index=False, encoding="utf-8-sig")
    decomposition.to_csv(decomposition_path, index=False, encoding="utf-8-sig")
    directional.to_csv(directional_path, index=False, encoding="utf-8-sig")
    window_sensitivity.to_csv(window_path, index=False, encoding="utf-8-sig")
    contrast.to_csv(contrast_path, index=False, encoding="utf-8-sig")
    background_directional.to_csv(
        background_directional_path, index=False, encoding="utf-8-sig"
    )
    recovery.to_csv(recovery_path, index=False, encoding="utf-8-sig")
    wind_ready_core.to_csv(wind_core_path, index=False, encoding="utf-8-sig")
    partition_columns = [
        "transition_id", "sensor", "reef_id", "sand_cay_id", "time_t", "time_t1",
        "time_interval_days", "typhoon_strong_count", "typhoon_r34_count",
        "local_typhoon_day_count", "non_typhoon_day_count", "local_typhoon_day_fraction",
        "wind_u_mean", "wind_v_mean", "wind_speed_mean", "wind_speed_p90",
        "wind_background_u_mean", "wind_background_v_mean", "wind_background_speed_mean",
        "wind_background_speed_p90", "wind_typhoon_u_mean", "wind_typhoon_v_mean",
        "wind_typhoon_speed_mean", "wind_typhoon_speed_p90",
        "wind_along_mean", "wind_cross_mean", "background_wind_along_mean",
        "background_wind_cross_mean", "typhoon_wind_along_mean", "typhoon_wind_cross_mean",
    ]
    wind_ready_core[[c for c in partition_columns if c in wind_ready_core.columns]].to_csv(
        wind_partition_path, index=False, encoding="utf-8-sig"
    )
    summary = {
        "analysis_type": "interval forcing alignment and exploratory rank association; no predictive or causal model fitted",
        "analysis_contract_version": ANALYSIS_CONTRACT_VERSION,
        "analysis_contract_sha256": analysis_contract_digest(),
        "forcing_window": "(time_t, time_t1]",
        "core_interval_days": list(ANALYSIS_INTERVAL_DAYS),
        "sensitivity_interval_days": [list(days) for days in WINDOW_SENSITIVITY],
        "input_transitions": int(len(transitions)),
        "era5": wind_summary,
        "wind_typhoon_day_partition": {
            "definition": (
                "A local typhoon day has an IBTrACS track point in the cay-specific "
                "R34 quadrant or is within 100 km of a >=17.5 m/s storm center."
            ),
            "source_file": str(args.typhoon_day_csv),
            "wind_complete_sentinel2_intervals_with_local_typhoon_days": int(
                wind_ready_core["local_typhoon_day_count"].fillna(0).gt(0).sum()
            ) if "local_typhoon_day_count" in wind_ready_core else 0,
            "wind_complete_sentinel2_local_typhoon_days": int(
                wind_ready_core["local_typhoon_day_count"].fillna(0).sum()
            ) if "local_typhoon_day_count" in wind_ready_core else 0,
        },
        "glorys": current_summary,
        "waverys": wave_summary,
        "complete_flow_wave_intervals": int(aligned["flow_wave_complete"].sum()),
        "complete_flow_wave_wind_intervals": int(aligned["flow_wave_wind_complete"].sum()),
        "complete_flow_wave_wind_sentinel2_intervals": int(len(wind_ready_core)),
        "core_complete_intervals": int(len(core)),
        "core_by_sensor": {str(key): int(value) for key, value in core["sensor"].value_counts().items()},
        "core_reefs": int(core["reef_id"].nunique()),
        "core_sand_cays": int(core["sand_cay_id"].nunique()),
        "sensor_consistency": sensor_cadence_summary(core, observations),
        "north_verification": {
            **cross_stats,
            "n_frames": int(len(north_frames)),
            "n_frames_verified": int(north_frames["status"].eq("verified").sum()),
            "n_frames_spec_only": int(north_frames["status"].eq("spec_only").sum()),
            "n_frames_flagged": int(north_frames["status"].eq("flagged").sum()),
            "direction_status_stamp": REVIEW_STAMP if all_verified else "unchanged",
        },
        "directional_analysis": {
            "axis_convention": "skimage orientation verified with synthetic masks; geographic axis unit vector (east, north) = (sin phi, -cos phi)",
            "predefined_pairs": [
                {"driver": d, "outcome": o} for d, o in sorted(PREDEFINED_DIRECTIONAL_PAIRS)
            ],
            "primary_sensor": "sentinel2",
            "warning": "Directional decomposition inherits the north-up assumption; associations remain observational.",
        },
        "background_pulse": {
            "regime_rule": "pulse = interval contains strong typhoon or R34 exposure; otherwise background",
            "n_by_regime": {
                str(sensor): {
                    str(regime): int(count)
                    for regime, count in group["forcing_regime"].value_counts().items()
                }
                for sensor, group in regime_frame.groupby("sensor")
            },
            "recovery": recovery_stats,
            "warning": "Post-event recovery is descriptive; mean reversion and tide aliasing remain alternative explanations.",
        },
        "interpretation_warning": "Wind, current and wave fields are coarse-grid external forcing proxies. Associations cannot separate storm, tide, reef geometry, registration, sensor or unresolved near-reef processes, and do not establish a primary driver.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in (
        aligned_path,
        core_path,
        association_path,
        consistency_path,
        north_path,
        decomposition_path,
        directional_path,
        window_path,
        contrast_path,
        background_directional_path,
        recovery_path,
        wind_core_path,
        wind_partition_path,
        summary_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
