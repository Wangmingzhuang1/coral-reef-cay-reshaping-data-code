"""合并颜色、形态、质心与台风数据，重建可用于后续统计分析的变化区间表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops


QUALITY_RANK = {"A": 3, "B": 2, "C": 1, "": 0}
PERTURBATIONS = (-1, 0, 1)
MASK_STRUCTURING_ELEMENT = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)


def read_mask(path_text: object) -> np.ndarray | None:
    path = Path(str(path_text))
    if not path.is_file():
        return None
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return mask > 0 if mask is not None else None


def load_observations(dataset_root: Path, color_csv: Path) -> pd.DataFrame:
    color = pd.read_csv(color_csv)
    color["color_status"] = color["status"]
    color_columns = [
        "image_id",
        "sensor",
        "manual_vegetation_state",
        "development_stage",
        "surface_cover",
        "quality_grade",
        "mask_path",
        "color_status",
        "pixel_count",
        "vegetation_fraction",
        "light_substrate_fraction",
        "dark_olive_anomaly_fraction",
        "unclassified_fraction",
        "spectral_composition_sum",
        "automatic_vegetation_state",
        "mean_red",
        "mean_green",
        "mean_blue",
        "median_value",
        "median_saturation",
        "median_green_excess",
    ]
    color = color[color_columns].rename(columns={"quality_grade": "annotation_quality_grade"})

    feature_specs = [
        (
            dataset_root / "outputs" / "derived_features" / "mask_features.csv",
            "google_earth",
        ),
        (
            dataset_root
            / "outputs"
            / "sentinel2_annotations"
            / "derived_features"
            / "mask_features.csv",
            "sentinel2",
        ),
    ]
    features = []
    for path, sensor in feature_specs:
        frame = pd.read_csv(path)
        frame["sensor"] = sensor
        if sensor == "sentinel2":
            frame.loc[frame["image_id"].str.startswith("qsat__", na=False), "sensor"] = "qsat_2017"
        features.append(frame)
    morphology = pd.concat(features, ignore_index=True)
    morphology = morphology.rename(columns={"quality_grade": "feature_quality_grade"})

    metadata_specs = [
        (
            dataset_root / "data" / "metadata" / "image_metadata_template.csv",
            "google_earth",
        ),
        (
            dataset_root / "data" / "metadata" / "sentinel2_annotation_metadata.csv",
            "sentinel2",
        ),
        (
            dataset_root / "data" / "metadata" / "qsat_annotation_metadata.csv",
            "qsat_2017",
        ),
    ]
    metadata_frames = []
    for path, sensor in metadata_specs:
        frame = pd.read_csv(path, dtype=str).fillna("")
        frame["sensor"] = sensor
        selected = [
            column
            for column in [
                "image_id",
                "sensor",
                "reference_frame_id",
                "pixel_size_m",
                "date_precision",
                "crs",
                "source",
                "scale_qc",
                "geo_scale_qc",
            ]
            if column in frame.columns
        ]
        metadata_frames.append(frame[selected])
    metadata = pd.concat(metadata_frames, ignore_index=True)
    metadata = metadata.drop_duplicates(["image_id", "sensor"], keep="last")

    observations = morphology.merge(color, on=["image_id", "sensor"], how="left")
    observations = observations.merge(metadata, on=["image_id", "sensor"], how="left")
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    observations["pixel_size_m"] = pd.to_numeric(observations["pixel_size_m"], errors="coerce")
    inferred_pixel_size = np.sqrt(
        pd.to_numeric(observations["sand_cay_area_m2"], errors="coerce")
        / pd.to_numeric(observations["sand_cay_area_pixels"], errors="coerce")
    )
    observations["pixel_size_m"] = observations["pixel_size_m"].fillna(inferred_pixel_size)
    observations["quality_grade"] = observations["annotation_quality_grade"].fillna("")
    missing_quality = observations["quality_grade"].eq("")
    observations.loc[missing_quality, "quality_grade"] = (
        observations.loc[missing_quality, "feature_quality_grade"].fillna("")
    )
    observations["reference_frame_id"] = observations["reference_frame_id"].fillna("")
    observations["quality_rank"] = observations["quality_grade"].map(QUALITY_RANK).fillna(0)
    observations["pixel_count"] = pd.to_numeric(observations["pixel_count"], errors="coerce")
    observations = observations.sort_values(
        ["sensor", "reef_id", "sand_cay_id", "reference_frame_id", "date", "quality_rank", "pixel_count"],
        ascending=[True, True, True, True, True, False, False],
    )
    return observations


def load_typhoons(event_csv: Path) -> dict[str, pd.DataFrame]:
    path = event_csv
    events = pd.read_csv(path)
    events["nearest_time_utc"] = pd.to_datetime(events["nearest_time_utc"], errors="coerce")
    for column in [
        "nearest_dist_km",
        "wind_at_nearest_ms",
        "storm_max_wind_ms",
        "inside_r34_quadrant_at_nearest",
        "event_relevant_quadrant",
    ]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    return {
        str(cay_id): group.sort_values("nearest_time_utc").reset_index(drop=True)
        for cay_id, group in events.groupby("sand_cay_id")
    }


def aggregate_typhoons(
    events_by_cay: dict[str, pd.DataFrame],
    sand_cay_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    events = events_by_cay.get(sand_cay_id)
    if events is None:
        return {
            "typhoon_data_available": False,
            "typhoon_event_count": 0,
            "typhoon_moderate_count": 0,
            "typhoon_strong_count": 0,
            "typhoon_r34_count": 0,
            "typhoon_min_distance_km": np.nan,
            "typhoon_max_nearest_wind_m_s": np.nan,
            "typhoon_max_storm_wind_m_s": np.nan,
        }
    selected = events[
        (events["nearest_time_utc"] > start) & (events["nearest_time_utc"] <= end)
    ]
    return {
        "typhoon_data_available": True,
        "typhoon_event_count": int(len(selected)),
        "typhoon_moderate_count": int((selected["impact_class"] == "moderate").sum()),
        "typhoon_strong_count": int(
            selected["event_relevant_quadrant"].fillna(0).eq(1).sum()
        ),
        "typhoon_r34_count": int(
            selected["inside_r34_quadrant_at_nearest"].fillna(0).eq(1).sum()
        ),
        "typhoon_min_distance_km": (
            float(selected["nearest_dist_km"].min()) if not selected.empty else np.nan
        ),
        "typhoon_max_nearest_wind_m_s": (
            float(selected["wind_at_nearest_ms"].max()) if not selected.empty else np.nan
        ),
        "typhoon_max_storm_wind_m_s": (
            float(selected["storm_max_wind_ms"].max()) if not selected.empty else np.nan
        ),
    }


def perturb_mask(mask: np.ndarray | None, pixels: int) -> np.ndarray | None:
    if mask is None:
        return None
    if pixels == 0:
        return mask.copy()
    perturbed = mask.astype(np.uint8).copy()
    for _ in range(abs(pixels)):
        kernel = MASK_STRUCTURING_ELEMENT if pixels > 0 else MASK_STRUCTURING_ELEMENT.T
        perturbed = cv2.dilate(perturbed, kernel) if pixels > 0 else cv2.erode(
            perturbed, kernel
        )
    return perturbed.astype(bool)


def perturbed_mask_metrics(mask: np.ndarray, pixel_size: float) -> dict[str, float]:
    """Compute area, centroid and major-axis orientation for a perturbed mask."""
    if not mask.any():
        return {
            "area_m2": np.nan,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "major_axis_angle": np.nan,
            "status": "empty_mask",
        }
    properties = regionprops(mask.astype(np.uint8))
    largest = max(properties, key=lambda item: item.area)
    area_m2 = float(largest.area) * pixel_size**2
    centroid_y, centroid_x = largest.centroid
    return {
        "area_m2": area_m2,
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
        "major_axis_angle": float(np.degrees(largest.orientation)),
        "status": "ok",
    }


def perturbed_transition_metrics(
    current: pd.Series,
    following: pd.Series,
    perturbation: int,
) -> dict[str, float]:
    pixel_size = float(np.nanmean([current["pixel_size_m"], following["pixel_size_m"]]))
    mask_t = perturb_mask(read_mask(current.get("mask_path", "")), perturbation)
    mask_t1 = perturb_mask(read_mask(following.get("mask_path", "")), perturbation)
    if mask_t is None or mask_t1 is None or mask_t.shape != mask_t1.shape:
        return {
            "perturbation_pixels": perturbation,
            "status": "mask_unavailable_or_shape_mismatch",
        }
    metrics_t = perturbed_mask_metrics(mask_t, pixel_size)
    metrics_t1 = perturbed_mask_metrics(mask_t1, pixel_size)
    if metrics_t["status"] != "ok" or metrics_t1["status"] != "ok":
        return {
            "perturbation_pixels": perturbation,
            "status": "empty_perturbed_mask",
        }
    east_shift_m = (metrics_t1["centroid_x"] - metrics_t["centroid_x"]) * pixel_size
    north_shift_m = -(metrics_t1["centroid_y"] - metrics_t["centroid_y"]) * pixel_size
    centroid_shift_m = float(np.hypot(east_shift_m, north_shift_m))
    erosion_pixels = int(np.logical_and(mask_t, ~mask_t1).sum())
    deposition_pixels = int(np.logical_and(~mask_t, mask_t1).sum())
    erosion_area_m2 = erosion_pixels * pixel_size**2
    deposition_area_m2 = deposition_pixels * pixel_size**2
    gross_boundary_change_m2 = erosion_area_m2 + deposition_area_m2
    return {
        "perturbation_pixels": perturbation,
        "status": "ok",
        "area_t_m2": metrics_t["area_m2"],
        "area_t1_m2": metrics_t1["area_m2"],
        "area_change_m2": metrics_t1["area_m2"] - metrics_t["area_m2"],
        "centroid_east_shift_m": east_shift_m,
        "centroid_north_shift_m": north_shift_m,
        "centroid_shift_m": centroid_shift_m,
        "erosion_area_m2": erosion_area_m2,
        "deposition_area_m2": deposition_area_m2,
        "gross_boundary_change_m2": gross_boundary_change_m2,
        "major_axis_angle_t": metrics_t["major_axis_angle"],
        "major_axis_angle_t1": metrics_t1["major_axis_angle"],
    }


def finite_delta(current: pd.Series, following: pd.Series, column: str) -> float:
    first = pd.to_numeric(pd.Series([current.get(column)]), errors="coerce").iloc[0]
    second = pd.to_numeric(pd.Series([following.get(column)]), errors="coerce").iloc[0]
    return float(second - first) if pd.notna(first) and pd.notna(second) else np.nan


def build_transition(
    current: pd.Series,
    following: pd.Series,
    events_by_cay: dict[str, pd.DataFrame],
) -> dict[str, object]:
    start = current["date"]
    end = following["date"]
    interval_days = int((end - start).days)
    years = interval_days / 365.2425
    pixel_size = float(np.nanmean([current["pixel_size_m"], following["pixel_size_m"]]))

    dx_pixels = finite_delta(current, following, "sand_cay_centroid_x")
    dy_pixels = finite_delta(current, following, "sand_cay_centroid_y")
    east_m = dx_pixels * pixel_size
    north_m = -dy_pixels * pixel_size
    shift_m = float(np.hypot(east_m, north_m))
    bearing_deg = float(np.degrees(np.arctan2(east_m, north_m)) % 360.0)

    area_change_m2 = finite_delta(current, following, "sand_cay_area_m2")
    vegetation_change = finite_delta(current, following, "vegetation_fraction")
    substrate_change = finite_delta(current, following, "light_substrate_fraction")
    dark_olive_anomaly_change = finite_delta(
        current, following, "dark_olive_anomaly_fraction"
    )
    unclassified_color_change = finite_delta(
        current, following, "unclassified_fraction"
    )

    mask_t = read_mask(current.get("mask_path", ""))
    mask_t1 = read_mask(following.get("mask_path", ""))
    if mask_t is not None and mask_t1 is not None and mask_t.shape == mask_t1.shape:
        erosion_pixels = int(np.logical_and(mask_t, ~mask_t1).sum())
        deposition_pixels = int(np.logical_and(~mask_t, mask_t1).sum())
        erosion_area_m2 = erosion_pixels * pixel_size**2
        deposition_area_m2 = deposition_pixels * pixel_size**2
        boundary_change_status = "ok"
    else:
        erosion_pixels = deposition_pixels = np.nan
        erosion_area_m2 = deposition_area_m2 = np.nan
        boundary_change_status = "mask_unavailable_or_shape_mismatch"

    transition = {
        "transition_id": (
            f"{current['sensor']}__{current['sand_cay_id']}__"
            f"{start.date()}__to__{end.date()}"
        ),
        "sensor": current["sensor"],
        "reef_id": current["reef_id"],
        "sand_cay_id": current["sand_cay_id"],
        "reference_frame_id": current["reference_frame_id"],
        "time_t": start.date().isoformat(),
        "time_t1": end.date().isoformat(),
        "time_interval_days": interval_days,
        "quality_grade_t": current["quality_grade"],
        "quality_grade_t1": following["quality_grade"],
        "area_t_m2": current["sand_cay_area_m2"],
        "area_t1_m2": following["sand_cay_area_m2"],
        "area_change_m2": area_change_m2,
        "area_change_m2_per_year": area_change_m2 / years,
        "centroid_east_shift_m": east_m,
        "centroid_north_shift_m": north_m,
        "centroid_shift_m": shift_m,
        "centroid_shift_m_per_year": shift_m / years,
        "centroid_bearing_deg": bearing_deg,
        "direction_status": "provisional_north_up_assumption",
        "perimeter_change_m": finite_delta(current, following, "sand_cay_perimeter_m"),
        "major_axis_change_m": finite_delta(current, following, "major_axis_length_m"),
        "minor_axis_change_m": finite_delta(current, following, "minor_axis_length_m"),
        "major_axis_angle_change_deg": finite_delta(
            current, following, "sand_cay_major_axis_angle"
        ),
        "vegetation_fraction_t": current.get("vegetation_fraction", np.nan),
        "vegetation_fraction_t1": following.get("vegetation_fraction", np.nan),
        "vegetation_fraction_change": vegetation_change,
        "light_substrate_fraction_change": substrate_change,
        "dark_olive_anomaly_fraction_change": dark_olive_anomaly_change,
        "unclassified_fraction_change": unclassified_color_change,
        "manual_vegetation_state_t": current.get("manual_vegetation_state", ""),
        "manual_vegetation_state_t1": following.get("manual_vegetation_state", ""),
        "erosion_pixels": erosion_pixels,
        "deposition_pixels": deposition_pixels,
        "erosion_area_m2": erosion_area_m2,
        "deposition_area_m2": deposition_area_m2,
        "gross_boundary_change_m2": erosion_area_m2 + deposition_area_m2,
        "boundary_change_status": boundary_change_status,
    }
    transition.update(
        aggregate_typhoons(events_by_cay, str(current["sand_cay_id"]), start, end)
    )
    return transition


def build_transitions(
    observations: pd.DataFrame,
    events_by_cay: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    unique = observations.drop_duplicates(
        ["sensor", "sand_cay_id", "reference_frame_id", "date"], keep="first"
    )
    rows: list[dict[str, object]] = []
    group_columns = ["sensor", "reef_id", "sand_cay_id", "reference_frame_id"]
    for _, group in unique.groupby(group_columns, dropna=False):
        group = group.sort_values("date")
        records = [row for _, row in group.iterrows()]
        for current, following in zip(records, records[1:]):
            if pd.isna(current["date"]) or pd.isna(following["date"]):
                continue
            if following["date"] <= current["date"]:
                continue
            rows.append(build_transition(current, following, events_by_cay))
    return pd.DataFrame(rows)


def build_mask_perturbation_table(
    observations: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    unique = observations.drop_duplicates(
        ["sensor", "sand_cay_id", "reference_frame_id", "date"], keep="first"
    )
    lookup = unique.set_index(
        ["sensor", "sand_cay_id", "reference_frame_id", "date"]
    )
    rows: list[dict[str, object]] = []
    for transition in transitions.itertuples(index=False):
        keys = [
            ["sensor", "sand_cay_id", "reference_frame_id", "time_t"],
            ["sensor", "sand_cay_id", "reference_frame_id", "time_t1"],
        ]
        records = []
        for key_columns in keys:
            key = tuple(getattr(transition, column) for column in key_columns)
            if key not in lookup.index:
                break
            records.append(lookup.loc[key])
        if len(records) != 2:
            continue
        current, following = records
        for perturbation in PERTURBATIONS:
            metrics = perturbed_transition_metrics(current, following, perturbation)
            rows.append(
                {
                    "transition_id": transition.transition_id,
                    "sensor": transition.sensor,
                    "reef_id": transition.reef_id,
                    "sand_cay_id": transition.sand_cay_id,
                    "reference_frame_id": transition.reference_frame_id,
                    "time_t": transition.time_t,
                    "time_t1": transition.time_t1,
                    "time_interval_days": transition.time_interval_days,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    research_root = Path(__file__).resolve().parent
    default_dataset = research_root / "data" / "source_dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument(
        "--typhoon-event-csv",
        type=Path,
        default=research_root / "outputs" / "台风事件正式暴露.csv",
    )
    parser.add_argument("--color-csv", type=Path, default=research_root / "outputs" / "沙洲颜色组成.csv")
    parser.add_argument("--output-dir", type=Path, default=research_root / "outputs")
    args = parser.parse_args()

    observations = load_observations(args.dataset_root, args.color_csv)
    if not args.typhoon_event_csv.is_file():
        raise FileNotFoundError(
            f"缺少象限匹配 R34 事件表：{args.typhoon_event_csv}；请先运行构建台风事件暴露特征.py"
        )
    events_by_cay = load_typhoons(args.typhoon_event_csv)
    transitions = build_transitions(observations, events_by_cay)
    perturbation = build_mask_perturbation_table(observations, transitions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observation_path = args.output_dir / "沙洲观测主表.csv"
    transition_path = args.output_dir / "沙洲变化区间.csv"
    perturbation_path = args.output_dir / "掩膜边界扰动稳健性.csv"
    summary_path = args.output_dir / "观测变化表核查.json"
    observations.drop(columns=["quality_rank"]).to_csv(
        observation_path, index=False, encoding="utf-8-sig"
    )
    transitions.to_csv(transition_path, index=False, encoding="utf-8-sig")
    perturbation.to_csv(perturbation_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_observations": int(len(observations)),
        "observation_counts_by_sensor": {
            str(key): int(value) for key, value in observations["sensor"].value_counts().items()
        },
        "n_observations_with_color": int(observations["vegetation_fraction"].notna().sum()),
        "n_transitions": int(len(transitions)),
        "transition_counts_by_sensor": {
            str(key): int(value)
            for key, value in transitions["sensor"].value_counts().items()
        },
        "n_transitions_with_boundary_change": int(
            (transitions["boundary_change_status"] == "ok").sum()
        ),
        "n_transitions_with_typhoon_data": int(
            transitions["typhoon_data_available"].fillna(False).sum()
        ),
        "n_intervals_with_strong_typhoon": int(
            (transitions["typhoon_strong_count"] > 0).sum()
        ),
        "n_intervals_with_r34": int((transitions["typhoon_r34_count"] > 0).sum()),
        "typhoon_exposure_definition": (
            "R34 uses the cay-bearing-specific IBTrACS quadrant at the nearest "
            "track point; strong exposure is quadrant-R34 or <=100 km with "
            "nearest center wind >=17.5 m/s."
        ),
        "mask_perturbation": {
            "perturbation_pixels": list(PERTURBATIONS),
            "rows": int(len(perturbation)),
            "transitions": int(perturbation["transition_id"].nunique())
            if not perturbation.empty
            else 0,
            "status_counts": {
                str(key): int(value)
                for key, value in perturbation["status"].value_counts().items()
            }
            if not perturbation.empty
            else {},
        },
        "direction_warning": (
            "质心方向已按图像上方暂作北向换算；正式解释方位前必须逐参考框架核验北向。"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"观测主表: {observation_path}")
    print(f"变化区间: {transition_path}")
    print(f"边界扰动: {perturbation_path}")
    print(f"核查结果: {summary_path}")


if __name__ == "__main__":
    main()
