"""构建FES2022b潮位敏感性所需的精确影像时刻、坐标和预测输入。

潮位是观测条件代理，用于检验边界提取是否受潮位影响；不是水动力机制变量。
脚本不会伪造没有精确时刻的观测，也不会在缺少AVISO数据时生成预测值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from 分析范围 import TIDE_CONTRACT, exclude_reefs


SCENE_PATTERN = r"__(S2[ABC]_.+)$"
REQUIRED_LATLON = ("sand_cay_center_lon", "sand_cay_center_lat")
FRAME_PATTERN = (
    r"s2frame_w(?P<west>[-0-9.]+)_s(?P<south>[-0-9.]+)"
    r"_e(?P<east>[-0-9.]+)_n(?P<north>[-0-9.]+)"
)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    standard_deviation = float(numeric.std(ddof=0))
    if not pd.notna(standard_deviation) or standard_deviation == 0:
        return numeric * 0
    return (numeric - float(numeric.mean())) / standard_deviation


def load_scene_times(metadata_dir: Path) -> pd.DataFrame:
    paths = [
        metadata_dir / "sentinel2_l2a_manifest.csv",
        metadata_dir / "sentinel2_ssl_manifest.csv",
    ]
    frames = []
    for path in paths:
        if not path.is_file():
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "scene_id" in frame.columns and "datetime_utc" in frame.columns:
            frames.append(frame[["scene_id", "datetime_utc"]])
    if not frames:
        return pd.DataFrame(columns=["scene_id", "acquisition_datetime_utc"])
    manifest = pd.concat(frames, ignore_index=True)
    manifest["acquisition_datetime_utc"] = pd.to_datetime(
        manifest["datetime_utc"], errors="coerce", utc=True
    )
    return (
        manifest.loc[manifest["acquisition_datetime_utc"].notna()]
        .drop_duplicates("scene_id")
        [["scene_id", "acquisition_datetime_utc"]]
    )


def load_cay_coordinates(metadata_dir: Path) -> pd.DataFrame:
    path = metadata_dir / "sentinel2_annotation_metadata.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["image_id", *REQUIRED_LATLON])
    metadata = pd.read_csv(path, low_memory=False)
    columns = ["image_id", *REQUIRED_LATLON]
    available = [column for column in columns if column in metadata.columns]
    frame = metadata[available].copy()
    for column in REQUIRED_LATLON:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates("image_id")[["image_id", *REQUIRED_LATLON]]


def project_frame_coordinates(
    sentinel: pd.DataFrame,
) -> pd.DataFrame:
    """Convert pixel centroids to WGS84 using each Sentinel frame and CRS."""
    frame = sentinel.copy()
    bounds = frame["reference_frame_id"].str.extract(FRAME_PATTERN)
    for column in ("west", "south", "east", "north"):
        frame[column] = pd.to_numeric(bounds[column], errors="coerce")
    numeric_columns = [
        "image_width",
        "image_height",
        "sand_cay_centroid_x",
        "sand_cay_centroid_y",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows = []
    for crs, group in frame.groupby("crs", dropna=False):
        if pd.isna(crs) or str(crs).strip() == "":
            rows.append(
                group.assign(
                    sand_cay_center_lon=np.nan,
                    sand_cay_center_lat=np.nan,
                    coordinate_status="missing_crs",
                )
            )
            continue
        frame_to_geographic = Transformer.from_crs(
            str(crs), "EPSG:4326", always_xy=True
        )
        geographic_to_frame = Transformer.from_crs(
            "EPSG:4326", str(crs), always_xy=True
        )
        west = group["west"].to_numpy(dtype=float)
        south = group["south"].to_numpy(dtype=float)
        east = group["east"].to_numpy(dtype=float)
        north = group["north"].to_numpy(dtype=float)
        upper_left_x, upper_left_y = geographic_to_frame.transform(west, north)
        lower_right_x, lower_right_y = geographic_to_frame.transform(east, south)
        width = np.maximum(group["image_width"].to_numpy(dtype=float), 1)
        height = np.maximum(group["image_height"].to_numpy(dtype=float), 1)
        center_x = upper_left_x + (
            group["sand_cay_centroid_x"].to_numpy(dtype=float) + 0.5
        ) * (lower_right_x - upper_left_x) / width
        center_y = upper_left_y - (
            group["sand_cay_centroid_y"].to_numpy(dtype=float) + 0.5
        ) * (upper_left_y - lower_right_y) / height
        lon, lat = frame_to_geographic.transform(center_x, center_y)
        rows.append(
            group.assign(
                projected_frame_x=center_x,
                projected_frame_y=center_y,
                sand_cay_center_lon=lon,
                sand_cay_center_lat=lat,
                coordinate_status="ok",
            )
        )
    if not rows:
        return frame.assign(
            projected_frame_x=np.nan,
            projected_frame_y=np.nan,
            sand_cay_center_lon=np.nan,
            sand_cay_center_lat=np.nan,
            coordinate_status="no_rows",
        )
    result = pd.concat(rows, ignore_index=True)
    invalid = result[list(REQUIRED_LATLON)].isna().any(axis=1)
    result.loc[invalid & result["coordinate_status"].eq("ok"), "coordinate_status"] = (
        "projection_failed"
    )
    return result


def build_observation_tide_inputs(
    observation_csv: Path,
    metadata_dir: Path,
) -> pd.DataFrame:
    observations = exclude_reefs(pd.read_csv(observation_csv, low_memory=False))
    sentinel = observations.loc[observations["sensor"].eq("sentinel2")].copy()
    sentinel["scene_id"] = sentinel["image_id"].str.extract(SCENE_PATTERN)[0]
    scene_times = load_scene_times(metadata_dir)
    sentinel = sentinel.merge(scene_times, on="scene_id", how="left")
    coordinates = load_cay_coordinates(metadata_dir)
    sentinel = sentinel.merge(coordinates, on="image_id", how="left")
    sentinel = project_frame_coordinates(sentinel)
    sentinel["date"] = pd.to_datetime(sentinel["date"], errors="coerce")
    for column in REQUIRED_LATLON:
        sentinel[column] = pd.to_numeric(sentinel[column], errors="coerce")
    sentinel["time_source"] = "sentinel_scene_manifest"
    missing_time = sentinel["acquisition_datetime_utc"].isna()
    sentinel.loc[missing_time, "time_source"] = "missing_exact_acquisition_time"
    sentinel["tide_prediction_status"] = "ready_for_prediction"
    sentinel.loc[
        missing_time
        | sentinel[list(REQUIRED_LATLON)].isna().any(axis=1),
        "tide_prediction_status",
    ] = "missing_time_or_coordinates"
    return sentinel[
        [
            "image_id",
            "sensor",
            "reef_id",
            "sand_cay_id",
            "reference_frame_id",
            "scene_id",
            "date",
            "acquisition_datetime_utc",
            "time_source",
            "crs",
            "coordinate_status",
            "sand_cay_center_lon",
            "sand_cay_center_lat",
            "tide_prediction_status",
        ]
    ].copy()


def predict_tides(
    inputs: pd.DataFrame,
    fes_config: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ready = inputs.loc[inputs["tide_prediction_status"].eq("ready_for_prediction")].copy()
    status: dict[str, object] = {
        "prediction_attempted": bool(len(ready)),
        "ready_observations": int(len(ready)),
        "fes_config": str(fes_config),
        "data_present": bool(fes_config.is_file()),
    }
    if ready.empty:
        status["status"] = "no_exact_timestamps"
        return pd.DataFrame(), status
    if not status["data_present"]:
        status["status"] = "blocked_missing_aviso_fes2022b_data"
        status["message"] = (
            "AVISO FES2022b access is required. Download the extrapolated ocean tide "
            "elevation grids, then provide a PyFES ocean_tide.yaml configuration before rerunning this script."
        )
        return pd.DataFrame(), status
    try:
        import pyfes
    except ImportError as exc:
        status["status"] = "blocked_missing_pyfes"
        status["error"] = str(exc)
        return pd.DataFrame(), status

    try:
        library_version = getattr(pyfes, "__version__", "unknown")
        configuration = pyfes.config.load(fes_config)
        model = configuration.models["tide"]
        settings = configuration.settings
        longitudes = ready["sand_cay_center_lon"].to_numpy(dtype=np.float64)
        latitudes = ready["sand_cay_center_lat"].to_numpy(dtype=np.float64)
        timestamps = ready["acquisition_datetime_utc"].to_numpy(dtype="datetime64[us]")
        diurnal_semidurnal, long_period, quality_flags = pyfes.evaluate_tide(
            model,
            timestamps,
            longitudes,
            latitudes,
            settings=settings,
        )
        total_cm = np.asarray(diurnal_semidurnal, dtype=float) + np.asarray(
            long_period, dtype=float
        )
        ready["tide_elevation_cm"] = total_cm
        ready["tide_elevation_m"] = total_cm / 100.0
        ready["tide_quality_flag"] = np.asarray(quality_flags, dtype=int)
        ready["tide_model"] = TIDE_CONTRACT["model_name"]
        ready["tide_grid"] = TIDE_CONTRACT["grid"]
        ready["prediction_library_version"] = str(library_version)
        ready["prediction_status"] = np.where(
            np.asarray(quality_flags, dtype=int).eq(0),
            "no_model_data_at_position",
            "ok",
        )
        status["status"] = "ok"
        status["prediction_library_version"] = str(library_version)
        status["quality_flag_counts"] = {
            int(flag): int(count)
            for flag, count in pd.Series(ready["tide_quality_flag"]).value_counts().items()
        }
        return ready, status
    except Exception as exc:
        status["status"] = f"prediction_failed:{type(exc).__name__}"
        status["error"] = str(exc)
        return pd.DataFrame(), status


def build_interval_tide(
    observation_tides: pd.DataFrame,
    transition_csv: Path,
    observation_csv: Path,
) -> pd.DataFrame:
    if observation_tides.empty:
        return pd.DataFrame()
    transitions = exclude_reefs(pd.read_csv(transition_csv, low_memory=False))
    observations = exclude_reefs(pd.read_csv(observation_csv, low_memory=False))
    sentinel = transitions.loc[transitions["sensor"].eq("sentinel2")].copy()
    observation_lookup = observations.loc[
        observations["sensor"].eq("sentinel2")
    ].set_index(["sand_cay_id", "date"])["image_id"]
    tide_lookup = observation_tides.set_index("image_id")
    rows = []
    for transition in sentinel.itertuples(index=False):
        keys = [
            (transition.sand_cay_id, pd.to_datetime(transition.time_t)),
            (transition.sand_cay_id, pd.to_datetime(transition.time_t1)),
        ]
        image_ids = []
        for key in keys:
            if key not in observation_lookup.index:
                break
            image_ids.append(observation_lookup.loc[key])
        if len(image_ids) != 2:
            continue
        pre_id, post_id = image_ids
        if pre_id not in tide_lookup.index or post_id not in tide_lookup.index:
            continue
        pre = tide_lookup.loc[pre_id]
        post = tide_lookup.loc[post_id]
        if (
            pd.isna(pre["tide_elevation_m"])
            or pd.isna(post["tide_elevation_m"])
        ):
            continue
        tide_mean = float(pre["tide_elevation_m"] + post["tide_elevation_m"]) / 2
        tide_delta = float(post["tide_elevation_m"] - pre["tide_elevation_m"])
        rows.append(
            {
                "transition_id": transition.transition_id,
                "sensor": transition.sensor,
                "reef_id": transition.reef_id,
                "sand_cay_id": transition.sand_cay_id,
                "pre_image_id": pre_id,
                "post_image_id": post_id,
                "pre_acquisition_datetime_utc": pre["acquisition_datetime_utc"],
                "post_acquisition_datetime_utc": post["acquisition_datetime_utc"],
                "pre_tide_elevation_m": float(pre["tide_elevation_m"]),
                "post_tide_elevation_m": float(post["tide_elevation_m"]),
                "tide_mean_m": tide_mean,
                "tide_delta_m": tide_delta,
                "tide_delta_abs_m": abs(tide_delta),
                "tide_model": TIDE_CONTRACT["model_name"],
                "tide_grid": TIDE_CONTRACT["grid"],
                "prediction_status": "ok",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["tide_mean_z"] = zscore(frame["tide_mean_m"])
    frame["tide_delta_z"] = zscore(frame["tide_delta_m"])
    frame["tide_delta_abs_z"] = zscore(frame["tide_delta_abs_m"])
    frame["tide_adjusted_status"] = "ok"
    return frame


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation-csv",
        type=Path,
        default=root / "outputs" / "沙洲观测主表.csv",
    )
    parser.add_argument(
        "--transition-csv",
        type=Path,
        default=root / "outputs" / "沙洲变化区间.csv",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=root / "data" / "source_dataset" / "data" / "metadata",
    )
    parser.add_argument(
        "--fes-data-dir",
        type=Path,
        default=root / "data" / "environment" / "tide" / "fes2022b" / "ocean_tide_extrapolated",
    )
    parser.add_argument(
        "--fes-config",
        type=Path,
        default=root / "data" / "environment" / "tide" / "fes2022b" / "ocean_tide.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observation_inputs = build_observation_tide_inputs(
        args.observation_csv, args.metadata_dir
    )
    observation_path = args.output_dir / "潮位影像时刻与坐标.csv"
    observation_inputs.to_csv(observation_path, index=False, encoding="utf-8-sig")

    predicted, prediction_status = predict_tides(
        observation_inputs, args.fes_config
    )
    predicted_path = args.output_dir / "潮位影像预测.csv"
    predicted.to_csv(predicted_path, index=False, encoding="utf-8-sig")

    interval_tide = build_interval_tide(
        predicted, args.transition_csv, args.observation_csv
    )
    interval_path = args.output_dir / "潮位敏感性区间.csv"
    interval_tide.to_csv(interval_path, index=False, encoding="utf-8-sig")

    summary = {
        "tide_contract": TIDE_CONTRACT,
        "observation_inputs": {
            "rows": int(len(observation_inputs)),
            "ready_for_prediction": int(
                observation_inputs["tide_prediction_status"]
                .eq("ready_for_prediction")
                .sum()
            ),
            "missing_time_or_coordinates": int(
                observation_inputs["tide_prediction_status"]
                .eq("missing_time_or_coordinates")
                .sum()
            ),
        },
        "prediction": prediction_status,
        "interval_sensitivity": {
            "rows": int(len(interval_tide)),
            "transitions": int(interval_tide["transition_id"].nunique())
            if not interval_tide.empty
            else 0,
        },
    }
    check_path = args.output_dir / "潮位敏感性核查.json"
    check_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for path in (observation_path, predicted_path, interval_path, check_path):
        print(path)


if __name__ == "__main__":
    main()
