"""从 IBTrACS 路径计算台风—沙洲事件的强度、距离、方向和局地持续时间。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from 分析范围 import exclude_reefs


EARTH_RADIUS_KM = 6371.0
MAX_CANDIDATE_DISTANCE_KM = 250.0
LOCAL_WIND_THRESHOLD_M_S = 17.5
R34_COLUMNS = [
    "USA_R34_NE_km",
    "USA_R34_SE_km",
    "USA_R34_SW_km",
    "USA_R34_NW_km",
]


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(value))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    first_latitude, second_latitude = np.radians([lat1, lat2])
    longitude_delta = np.radians(lon2 - lon1)
    y = np.sin(longitude_delta) * np.cos(second_latitude)
    x = np.cos(first_latitude) * np.sin(second_latitude) - np.sin(
        first_latitude
    ) * np.cos(second_latitude) * np.cos(longitude_delta)
    return float(np.degrees(np.arctan2(y, x)) % 360)


def integrated_duration_hours(
    times: pd.Series,
    condition: np.ndarray,
) -> float:
    if len(times) < 2:
        return 0.0
    time_hours = times.astype("int64").to_numpy(dtype=float) / 3.6e12
    delta = np.diff(time_hours)
    valid_delta = np.isfinite(delta) & (delta > 0) & (delta <= 24)
    weights = 0.5 * (condition[:-1].astype(float) + condition[1:].astype(float))
    return float(np.sum(delta[valid_delta] * weights[valid_delta]))


def r34_radius_toward_cay(
    track: pd.DataFrame,
    cay_lat: float,
    cay_lon: float,
) -> np.ndarray:
    """按沙洲相对台风中心的方位选取对应的 IBTrACS R34 象限半径。"""
    bearings = np.array(
        [
            bearing_deg(float(lat), float(lon), cay_lat, cay_lon)
            for lat, lon in zip(track["LAT"], track["LON"], strict=True)
        ]
    )
    quadrant_index = np.floor(bearings / 90).astype(int)
    quadrant_radii = track[R34_COLUMNS].fillna(0).to_numpy(dtype=float)
    return quadrant_radii[np.arange(len(track)), quadrant_index]


def build_quadrant_event_table(events: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """保留原始事件表，同时用沙洲方位对应的 R34 象限给出正式暴露判定。"""
    data = exclude_reefs(events.copy())
    data["nearest_time_utc"] = pd.to_datetime(
        data["nearest_time_utc"], errors="coerce"
    )
    for column in ["nearest_dist_km", "wind_at_nearest_ms", "inside_r34"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    track_data = tracks.copy()
    track_data["ISO_TIME"] = pd.to_datetime(track_data["ISO_TIME"], errors="coerce")
    for column in ["LAT", "LON", "wind_ms", *R34_COLUMNS]:
        track_data[column] = pd.to_numeric(track_data[column], errors="coerce")
    track_data = track_data.dropna(
        subset=["SID", "ISO_TIME", "LAT", "LON"]
    ).sort_values(["SID", "ISO_TIME"])
    tracks_by_sid = {
        str(sid): frame.reset_index(drop=True)
        for sid, frame in track_data.groupby("SID")
    }

    r34_flags: list[float] = []
    r34_radii: list[float] = []
    for event in data.itertuples(index=False):
        track = tracks_by_sid.get(str(event.sid))
        if track is None or track.empty or pd.isna(event.nearest_time_utc):
            r34_flags.append(np.nan)
            r34_radii.append(np.nan)
            continue
        index = int(
            np.argmin(
                np.abs(track["ISO_TIME"] - pd.Timestamp(event.nearest_time_utc))
            )
        )
        point = track.iloc[[index]]
        distance = float(
            haversine_km(
                np.array([float(event.cay_lat)]),
                np.array([float(event.cay_lon)]),
                point["LAT"].to_numpy(dtype=float),
                point["LON"].to_numpy(dtype=float),
            )[0]
        )
        radius = float(
            r34_radius_toward_cay(point, float(event.cay_lat), float(event.cay_lon))[0]
        )
        r34_flags.append(float(radius > 0 and distance <= radius))
        r34_radii.append(radius)
    data["inside_r34_quadrant_at_nearest"] = r34_flags
    data["r34_quadrant_radius_at_nearest_km"] = r34_radii
    data["event_relevant_quadrant"] = (
        data["inside_r34_quadrant_at_nearest"].fillna(0).eq(1)
        | (
            data["nearest_dist_km"].le(100)
            & data["wind_at_nearest_ms"].ge(LOCAL_WIND_THRESHOLD_M_S)
        )
    )
    data["r34_definition"] = (
        "cay-bearing-specific IBTrACS USA_R34 quadrant at nearest track point"
    )
    return data


def build_local_typhoon_days(events: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """构建沙洲级局地强台风日，供 ERA5 风场与台风信号拆分使用。

    日标记仅在对应 IBTrACS 轨迹点满足 R34 象限覆盖，或距中心不超过
    100 km 且中心风速至少 17.5 m/s 时赋值。它不是把整场台风生命期都
    计为局地台风日。
    """
    events = exclude_reefs(events.copy())
    events["nearest_time_utc"] = pd.to_datetime(
        events["nearest_time_utc"], errors="coerce"
    )
    events["nearest_dist_km"] = pd.to_numeric(
        events["nearest_dist_km"], errors="coerce"
    )
    events["wind_at_nearest_ms"] = pd.to_numeric(
        events["wind_at_nearest_ms"], errors="coerce"
    )
    strict = events["event_relevant_quadrant"].fillna(False).astype(bool)
    events = events.loc[strict].drop_duplicates(["sand_cay_id", "sid"]).copy()

    tracks = tracks.copy()
    tracks["ISO_TIME"] = pd.to_datetime(tracks["ISO_TIME"], errors="coerce")
    for column in ["LAT", "LON", "wind_ms", *R34_COLUMNS]:
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    tracks = tracks.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"]).sort_values(
        ["SID", "ISO_TIME"]
    )
    tracks_by_sid = {
        str(sid): frame.reset_index(drop=True) for sid, frame in tracks.groupby("SID")
    }

    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        track = tracks_by_sid.get(str(event.sid))
        if track is None or track.empty:
            continue
        cay_lat, cay_lon = float(event.cay_lat), float(event.cay_lon)
        distance = haversine_km(
            np.full(len(track), cay_lat),
            np.full(len(track), cay_lon),
            track["LAT"].to_numpy(dtype=float),
            track["LON"].to_numpy(dtype=float),
        )
        r34_radius = r34_radius_toward_cay(track, cay_lat, cay_lon)
        r34 = (r34_radius > 0) & (distance <= r34_radius)
        near_strong = (distance <= 100) & (
            track["wind_ms"].to_numpy(dtype=float) >= LOCAL_WIND_THRESHOLD_M_S
        )
        local = r34 | near_strong
        if not local.any():
            continue
        impacted = track.loc[local].copy()
        impacted["date"] = impacted["ISO_TIME"].dt.normalize()
        impacted["r34_at_track_point"] = r34[local]
        impacted["near_strong_at_track_point"] = near_strong[local]
        impacted["distance_km"] = distance[local]
        for date, daily in impacted.groupby("date"):
            rows.append(
                {
                    "reef_id": event.reef_id,
                    "sand_cay_id": event.sand_cay_id,
                    "sid": event.sid,
                    "storm_name": event.name,
                    "date": date,
                    "local_typhoon_day": 1,
                    "r34_track_point_count": int(daily["r34_at_track_point"].sum()),
                    "near_strong_track_point_count": int(
                        daily["near_strong_at_track_point"].sum()
                    ),
                    "min_track_distance_km": float(daily["distance_km"].min()),
                    "max_track_wind_ms": float(daily["wind_ms"].max()),
                    "day_status": "IBTrACS local R34 quadrant or <=100 km strong-center-wind",
                }
            )
    columns = [
        "reef_id", "sand_cay_id", "sid", "storm_name", "date", "local_typhoon_day",
        "r34_track_point_count", "near_strong_track_point_count",
        "min_track_distance_km", "max_track_wind_ms", "day_status",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    daily = pd.DataFrame(rows)
    return (
        daily.groupby(["reef_id", "sand_cay_id", "date"], as_index=False)
        .agg(
            local_typhoon_day=("local_typhoon_day", "max"),
            storm_count=("sid", "nunique"),
            r34_track_point_count=("r34_track_point_count", "sum"),
            near_strong_track_point_count=("near_strong_track_point_count", "sum"),
            min_track_distance_km=("min_track_distance_km", "min"),
            max_track_wind_ms=("max_track_wind_ms", "max"),
            day_status=("day_status", "first"),
        )
        .sort_values(["sand_cay_id", "date"])
    )


def build_features(events: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    events = exclude_reefs(events)
    events["nearest_time_utc"] = pd.to_datetime(events["nearest_time_utc"], errors="coerce")
    events["nearest_dist_km"] = pd.to_numeric(events["nearest_dist_km"], errors="coerce")
    events = events[
        events["nearest_dist_km"].le(MAX_CANDIDATE_DISTANCE_KM)
        & events["nearest_time_utc"].notna()
    ].copy()

    tracks["ISO_TIME"] = pd.to_datetime(tracks["ISO_TIME"], errors="coerce")
    for column in ["LAT", "LON", "wind_ms", *R34_COLUMNS]:
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    tracks = tracks.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"]).sort_values(
        ["SID", "ISO_TIME"]
    )

    rows: list[dict[str, object]] = []
    tracks_by_sid = {str(sid): frame.reset_index(drop=True) for sid, frame in tracks.groupby("SID")}
    for event in events.itertuples(index=False):
        track = tracks_by_sid.get(str(event.sid))
        if track is None or track.empty:
            continue
        cay_lat = float(event.cay_lat)
        cay_lon = float(event.cay_lon)
        distance = haversine_km(
            np.full(len(track), cay_lat),
            np.full(len(track), cay_lon),
            track["LAT"].to_numpy(dtype=float),
            track["LON"].to_numpy(dtype=float),
        )
        nearest_index = int(np.argmin(np.abs(track["ISO_TIME"] - event.nearest_time_utc)))
        previous_index = max(0, nearest_index - 1)
        following_index = min(len(track) - 1, nearest_index + 1)
        motion_bearing = (
            bearing_deg(
                float(track.loc[previous_index, "LAT"]),
                float(track.loc[previous_index, "LON"]),
                float(track.loc[following_index, "LAT"]),
                float(track.loc[following_index, "LON"]),
            )
            if previous_index != following_index
            else np.nan
        )
        center_bearing = bearing_deg(
            cay_lat,
            cay_lon,
            float(track.loc[nearest_index, "LAT"]),
            float(track.loc[nearest_index, "LON"]),
        )
        r34_toward_cay = r34_radius_toward_cay(track, cay_lat, cay_lon)
        center_wind = track["wind_ms"].to_numpy(dtype=float)
        duration_250 = integrated_duration_hours(track["ISO_TIME"], distance <= 250)
        duration_100 = integrated_duration_hours(track["ISO_TIME"], distance <= 100)
        duration_r34 = integrated_duration_hours(
            track["ISO_TIME"],
            (r34_toward_cay > 0) & (distance <= r34_toward_cay),
        )
        duration_wind_proximity = integrated_duration_hours(
            track["ISO_TIME"],
            (distance <= 250) & (center_wind >= LOCAL_WIND_THRESHOLD_M_S),
        )
        rows.append(
            {
                "reef_id": event.reef_id,
                "sand_cay_id": event.sand_cay_id,
                "sid": event.sid,
                "storm_name": event.name,
                "nearest_time_utc": event.nearest_time_utc,
                "nearest_dist_km": float(event.nearest_dist_km),
                "wind_at_nearest_ms": event.wind_at_nearest_ms,
                "storm_max_wind_ms": event.storm_max_wind_ms,
                "inside_r34_at_nearest": event.inside_r34,
                "inside_r34_quadrant_at_nearest": event.inside_r34_quadrant_at_nearest,
                "storm_center_bearing_from_cay_deg": center_bearing,
                "storm_motion_bearing_deg": motion_bearing,
                "local_duration_within_250km_h": duration_250,
                "local_duration_within_100km_h": duration_100,
                "local_r34_duration_approx_h": duration_r34,
                "center_wind_ge17_5_within_250km_h": duration_wind_proximity,
                "direction_status": (
                    "geographic track bearings available; not yet aligned to image-axis "
                    "cay orientation"
                ),
                "duration_status": (
                    "track-point trapezoid approximation; R34 uses the cay-bearing quadrant"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["nearest_time_utc", "sid", "sand_cay_id"]
    )


def main() -> None:
    research_root = Path(__file__).resolve().parent
    dataset_root = research_root / "data" / "source_dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-csv",
        type=Path,
        default=dataset_root / "outputs" / "typhoon_events" / "islet_typhoon_events.csv",
    )
    parser.add_argument(
        "--track-csv",
        type=Path,
        default=dataset_root
        / "outputs"
        / "typhoon_events"
        / "noaa_ibtracs_relevant_tracks.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=research_root / "outputs")
    args = parser.parse_args()

    events = pd.read_csv(args.event_csv, low_memory=False)
    tracks = pd.read_csv(args.track_csv, low_memory=False)
    corrected_events = build_quadrant_event_table(events, tracks)
    features = build_features(corrected_events, tracks)
    local_days = build_local_typhoon_days(corrected_events, tracks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "台风事件暴露特征.csv"
    corrected_event_path = args.output_dir / "台风事件正式暴露.csv"
    local_days_path = args.output_dir / "台风局地强风日.csv"
    features.to_csv(output_path, index=False, encoding="utf-8-sig")
    corrected_events.to_csv(corrected_event_path, index=False, encoding="utf-8-sig")
    local_days.to_csv(local_days_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(corrected_event_path)
    print(local_days_path)
    print(
        {
            "event_cay_records": int(len(features)),
            "storms": int(features["sid"].nunique()),
            "cays": int(features["sand_cay_id"].nunique()),
            "local_typhoon_days": int(len(local_days)),
            "quadrant_r34_events": int(
                corrected_events["inside_r34_quadrant_at_nearest"].fillna(0).sum()
            ),
        }
    )


if __name__ == "__main__":
    main()
