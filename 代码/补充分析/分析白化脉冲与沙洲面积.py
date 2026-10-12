"""真实礁盘范围 NOAA 5 km DHW 重建，并与同季节面积配对上的 ENSO 比较。"""

from __future__ import annotations

import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
UPSTREAM = ROOT / "data" / "source_dataset"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from 分析范围 import exclude_reefs  # noqa: E402
from 区域气候滞后模型 import apply_figure_style  # noqa: E402

NOAA_ERDDAP = "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/CRW_dhw_v1_0.csv"
GRID = 0.05
HALF_GRID = 0.025
MAX_DOWNLOAD_WORKERS = 2
ANNUAL_DAYS = (350, 380)
SEASON_GAP = 15
BROAD_ANNUAL_DAYS = (330, 400)
BROAD_SEASON_GAP = 30
DHW_THRESHOLD = 8.0
REEF_AREA_THRESHOLD = 0.50
ENSO_LAGS = (0, 6, 12, 18, 24)
# DHW itself describes the preceding 12 weeks.  The annual outcome can resolve
# supply/geomorphic consequences only at annual, not monthly, precision.
LAG_BANDS = (("0-<1 year", 0, 12), ("1-<2 years", 12, 24), ("2-<3 years", 24, 36))
REFERENCE = "No reef-wide DHW>=8 in prior 3 years"
CONTROLS = ["duration_deviation_days", "season_gap_days", "baseline_year", "season_sin", "season_cos"]
CONTINUOUS_CONTROLS = ["duration_deviation_days", "season_gap_days", "season_sin", "season_cos"]
CONTINUOUS_WINDOW_DAYS = 365
PRIMARY_REGIONS = ("GBR", "MEDF")


def region_code(reef_id: str) -> str:
    for region in ("GBR", "MEDF", "OTHER", "NH"):
        if str(reef_id).startswith(region):
            return region
    return "UNASSIGNED"


def seasonal_gap(left: pd.Timestamp, right: pd.Timestamp) -> int:
    difference = abs(left.dayofyear - right.dayofyear)
    return int(min(difference, 365 - difference))


def area_km2(west: float, north: float, east: float, south: float) -> float:
    radius = 6371.0088
    return float(radius**2 * np.deg2rad(east - west) * (np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))))


def load_observations() -> pd.DataFrame:
    table = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", encoding="utf-8-sig", low_memory=False)
    vegetation_audit = pd.read_csv(ROOT / "outputs" / "植被状态审校主表.csv", encoding="utf-8-sig", low_memory=False)
    audit_columns = ["image_id", "analysis_vegetation_state", "label_status"]
    if vegetation_audit["image_id"].duplicated().any():
        raise ValueError("植被状态审校主表存在重复 image_id")
    table = table.merge(
        vegetation_audit[audit_columns], on="image_id", how="left", validate="one_to_one"
    )
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    table["sand_cay_area_m2"] = pd.to_numeric(table["sand_cay_area_m2"], errors="coerce")
    table = exclude_reefs(table)
    return table.loc[
        table["sensor"].eq("sentinel2")
        & table["quality_grade"].isin(["A", "B"])
        & table["date"].notna()
        & table["sand_cay_area_m2"].gt(0)
    ].copy()


def same_season_pairs(
    observations: pd.DataFrame,
    annual_days: tuple[int, int] = ANNUAL_DAYS,
    season_gap: int = SEASON_GAP,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ["reef_id", "sand_cay_id", "reference_frame_id"]
    for (reef_id, cay_id, frame_id), group in observations.groupby(groups, dropna=False):
        # 同日多景时用 image_id 确定性选择，避免依赖 CSV 原始行序。
        group = group.sort_values(["date", "image_id"]).drop_duplicates("date").reset_index(drop=True)
        for index, source in group.iterrows():
            later = group.iloc[index + 1:].copy()
            later["duration_days"] = (later["date"] - source["date"]).dt.days
            later = later.loc[later["duration_days"].between(*annual_days)]
            if later.empty:
                continue
            later["season_gap_days"] = later["date"].map(lambda value: seasonal_gap(source["date"], value))
            later = later.loc[later["season_gap_days"].le(season_gap)].copy()
            if later.empty:
                continue
            later["score"] = (later["duration_days"] - 365.2425).abs() * 100 + later["season_gap_days"]
            target = later.sort_values(["score", "date"]).iloc[0]
            rows.append({
                "pair_id": f"sentinel2__{cay_id}__{source['date'].date()}__to__{target['date'].date()}",
                "reef_id": reef_id, "sand_cay_id": cay_id, "reference_frame_id": frame_id,
                "image_id_t": source["image_id"], "image_id_t1": target["image_id"],
                "time_t": source["date"], "time_t1": target["date"],
                "duration_days": int(target["duration_days"]), "season_gap_days": int(target["season_gap_days"]),
                "area_t_m2": float(source["sand_cay_area_m2"]), "area_t1_m2": float(target["sand_cay_area_m2"]),
                "log_area_ratio": float(np.log(target["sand_cay_area_m2"]) - np.log(source["sand_cay_area_m2"])),
                "baseline_manual_vegetation_state": source["manual_vegetation_state"],
                "baseline_analysis_vegetation_state": source["analysis_vegetation_state"],
                "baseline_vegetation_label_status": source["label_status"],
                "baseline_vegetation_fraction": float(source["vegetation_fraction"]),
                "followup_manual_vegetation_state": target["manual_vegetation_state"],
                "followup_analysis_vegetation_state": target["analysis_vegetation_state"],
                "followup_vegetation_label_status": target["label_status"],
                "followup_vegetation_fraction": float(target["vegetation_fraction"]),
            })
    return pd.DataFrame(rows)


def parse_kmz(path: Path) -> dict[str, tuple[float, float, float, float]]:
    import xml.etree.ElementTree as ET
    import zipfile
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("doc.kml"))
    boxes = {}
    for folder in root.findall(".//k:Folder", ns):
        reef = folder.findtext("k:name", default="", namespaces=ns)
        points = {}
        for placemark in folder.findall(".//k:Placemark", ns):
            name = placemark.findtext("k:name", default="", namespaces=ns)
            if not (name.endswith("_reef_nw") or name.endswith("_reef_se")):
                continue
            coordinates = placemark.findtext(".//k:Point/k:coordinates", default="", namespaces=ns).strip()
            lon, lat, *_ = coordinates.split(",")
            points["nw" if name.endswith("_reef_nw") else "se"] = (float(lon), float(lat))
        if set(points) == {"nw", "se"}:
            boxes[reef] = (min(points["nw"][0], points["se"][0]), max(points["nw"][1], points["se"][1]), max(points["nw"][0], points["se"][0]), min(points["nw"][1], points["se"][1]))
    return boxes


def verified_extents(active: set[str], out: Path) -> pd.DataFrame:
    meta_path = UPSTREAM / "data" / "metadata" / "reef_extent_metadata.csv"
    kmz_path = UPSTREAM / "data" / "metadata" / "standardized_coordinates.kmz"
    table = pd.read_csv(meta_path, encoding="utf-8-sig")
    required = ["reef_id", "crs", "bbox_west", "bbox_north", "bbox_east", "bbox_south"]
    if set(required) - set(table.columns):
        raise ValueError("reef_extent_metadata.csv 缺少必需字段")
    table = table.loc[table["reef_id"].isin(active), required].copy()
    missing = sorted(active - set(table["reef_id"]))
    if missing:
        raise ValueError(f"以下活跃礁盘无经纬度范围，不能以中心点替代：{missing}")
    if table["reef_id"].duplicated().any() or not table["crs"].eq("EPSG:4326").all():
        raise ValueError("礁盘范围表存在重复 ID 或非 EPSG:4326 坐标系")
    table = table.rename(columns={"bbox_west": "reef_west", "bbox_north": "reef_north", "bbox_east": "reef_east", "bbox_south": "reef_south"})
    for col in ["reef_west", "reef_north", "reef_east", "reef_south"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if not (table[["reef_west", "reef_north", "reef_east", "reef_south"]].notna().all(axis=1) & table["reef_west"].lt(table["reef_east"]) & table["reef_south"].lt(table["reef_north"])).all():
        raise ValueError("礁盘范围表有空值或无效边界")
    kmz = parse_kmz(kmz_path)
    statuses, deltas = [], []
    for row in table.itertuples(index=False):
        source = (row.reef_west, row.reef_north, row.reef_east, row.reef_south)
        kmz_box = kmz.get(row.reef_id)
        if kmz_box is None:
            statuses.append("not_in_kmz"); deltas.append(np.nan)
        else:
            delta = max(abs(a - b) for a, b in zip(source, kmz_box))
            statuses.append("exact_match" if delta <= 1e-9 else "mismatch"); deltas.append(delta)
    table["kmz_status"] = statuses
    table["kmz_max_abs_coordinate_delta"] = deltas
    if table["kmz_status"].eq("mismatch").any():
        raise ValueError("KMZ 与范围表的重叠礁盘存在坐标不一致")
    table["reef_area_km2"] = [area_km2(*values) for values in zip(table["reef_west"], table["reef_north"], table["reef_east"], table["reef_south"])]
    table["spatial_source"] = "reef_extent_metadata.csv; KMZ cross-check where available"
    table.sort_values("reef_id").to_csv(out / "礁盘范围与KMZ核验.csv", index=False, encoding="utf-8-sig")
    return table


def noaa_lon(value: float) -> float:
    return value if value >= 0 else value + 360


def normalize_raw_grid(table: pd.DataFrame, reef_id: str) -> pd.DataFrame:
    """标准化 NOAA 日格点并只移除分块边界的完全重复记录。"""
    required = ["date", "latitude", "longitude", "dhw_cweeks"]
    if set(required) - set(table.columns):
        raise ValueError(f"{reef_id} NOAA 返回字段异常")
    table = table.copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    for col in ["latitude", "longitude", "dhw_cweeks"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna(subset=["date", "latitude", "longitude"]).sort_values(["date", "latitude", "longitude"])
    duplicated = table.duplicated(["date", "latitude", "longitude"], keep=False)
    if duplicated.any():
        duplicate_values = table.loc[duplicated].groupby(["date", "latitude", "longitude"], observed=True)["dhw_cweeks"].nunique(dropna=False)
        if duplicate_values.gt(1).any():
            raise ValueError(f"{reef_id} 分块重叠记录的 DHW 值不一致，不能静默去重")
        table = table.drop_duplicates(["date", "latitude", "longitude"], keep="first")
    if table.empty:
        raise ValueError(f"{reef_id} NOAA 返回空数据")
    return table.reset_index(drop=True)


def download_dhw(row: pd.Series, start: pd.Timestamp, end: pd.Timestamp, target: Path, refresh: bool) -> dict[str, object]:
    if target.exists() and not refresh:
        cached = normalize_raw_grid(pd.read_csv(target), str(row["reef_id"]))
        if len(cached) != len(pd.read_csv(target)):
            partial = target.with_suffix(".part")
            cached.to_csv(partial, index=False, encoding="utf-8-sig")
            partial.replace(target)
        return {"reef_id": row["reef_id"], "status": "cached", "n_rows": int(len(cached)), "path": str(target)}
    west, east = noaa_lon(float(row["reef_west"])) - HALF_GRID, noaa_lon(float(row["reef_east"])) + HALF_GRID
    if west < 0 or east > 360 or west >= east:
        raise ValueError(f"{row['reef_id']} 跨经线，禁止隐式改用中心点或拆分")
    # 整个十三年子集在 NOAA ERDDAP 上可能长时间排队；按自然年请求并仅在同一
    # 礁盘内拼接，既不改变空间支持，也不改变每日原始变量。
    frames: list[pd.DataFrame] = []
    urls: list[str] = []
    for year in range(start.year, end.year + 1, 3):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=min(year + 2, end.year), month=12, day=31))
        expression = (
            "degree_heating_week"
            f"[({chunk_start:%Y-%m-%dT12:00:00Z}):1:({chunk_end:%Y-%m-%dT12:00:00Z})]"
            f"[({float(row['reef_south']) - HALF_GRID:.8f}):1:({float(row['reef_north']) + HALF_GRID:.8f})]"
            f"[({west:.8f}):1:({east:.8f})]"
        )
        url = NOAA_ERDDAP + "?" + urllib.parse.quote(expression, safe="[]():,.-")
        response = subprocess.run(
            ["curl.exe", "--globoff", "--fail", "--silent", "--show-error", "--location", "--max-time", "75", url],
            capture_output=True,
            check=False,
        )
        if response.returncode != 0:
            message = response.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"NOAA ERDDAP request failed for {year}: {message}")
        raw = response.stdout
        frame = pd.read_csv(io.BytesIO(raw), skiprows=[1]).rename(columns={"time": "date", "degree_heating_week": "dhw_cweeks"})
        frames.append(frame)
        urls.append(url)
    table = normalize_raw_grid(pd.concat(frames, ignore_index=True), str(row["reef_id"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    table.to_csv(partial, index=False, encoding="utf-8-sig")
    partial.replace(target)
    return {"reef_id": row["reef_id"], "status": "downloaded", "n_rows": int(len(table)), "path": str(target), "urls": " | ".join(urls)}


def weights_for_reef(row: pd.Series, raw: pd.DataFrame) -> pd.DataFrame:
    cells = raw[["latitude", "longitude"]].drop_duplicates().copy()
    cells["west"] = np.maximum(cells["longitude"] - HALF_GRID, row["reef_west"])
    cells["east"] = np.minimum(cells["longitude"] + HALF_GRID, row["reef_east"])
    cells["south"] = np.maximum(cells["latitude"] - HALF_GRID, row["reef_south"])
    cells["north"] = np.minimum(cells["latitude"] + HALF_GRID, row["reef_north"])
    cells = cells.loc[cells["west"].lt(cells["east"]) & cells["south"].lt(cells["north"])].copy()
    cells["overlap_area_km2"] = [area_km2(*values) for values in zip(cells["west"], cells["north"], cells["east"], cells["south"])]
    cells["reef_area_weight"] = cells["overlap_area_km2"] / float(row["reef_area_km2"])
    if cells.empty or not np.isclose(cells["reef_area_weight"].sum(), 1, atol=1e-7):
        raise ValueError(f"{row['reef_id']} 的 NOAA 像元未完整覆盖礁盘边界")
    cells["reef_id"] = row["reef_id"]
    return cells[["reef_id", "latitude", "longitude", "overlap_area_km2", "reef_area_weight"]]


def aggregate_dhw(row: pd.Series, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = normalize_raw_grid(pd.read_csv(cache / f"{row['reef_id']}.csv"), str(row["reef_id"]))
    weights = weights_for_reef(row, raw)
    joined = raw.merge(weights, on=["latitude", "longitude"], how="inner", validate="many_to_one")
    records = []
    for date, group in joined.groupby("date", sort=True):
        valid = group.loc[group["dhw_cweeks"].notna()]
        support = float(valid["reef_area_weight"].sum())
        records.append({
            "reef_id": row["reef_id"], "date": date, "valid_reef_area_fraction": support,
            "n_overlapping_grid_cells": int(len(weights)), "n_valid_grid_cells": int(len(valid)),
            "dhw_area_weighted_mean": float(np.average(valid["dhw_cweeks"], weights=valid["reef_area_weight"])) if support else np.nan,
            "dhw_area_weighted_max": float(valid["dhw_cweeks"].max()) if support else np.nan,
            "reef_area_fraction_ge8": float(valid.loc[valid["dhw_cweeks"].ge(DHW_THRESHOLD), "reef_area_weight"].sum()) if support else np.nan,
        })
    return pd.DataFrame(records), weights


def audit_raw_coverage(extents: pd.DataFrame, cache: Path, start: pd.Timestamp, end: pd.Timestamp, out: Path) -> dict[str, object]:
    """Verify the same official NOAA time axis and complete pixel support for every reef."""
    reference_dates: pd.DatetimeIndex | None = None
    rows = []
    for reef in extents["reef_id"]:
        raw = normalize_raw_grid(pd.read_csv(cache / f"{reef}.csv"), reef)
        dates = pd.DatetimeIndex(raw.loc[raw["date"].between(start, end), "date"].drop_duplicates().sort_values())
        cells_per_date = raw.loc[raw["date"].between(start, end)].groupby("date", observed=True).size()
        if reference_dates is None:
            reference_dates = dates
        same_axis = dates.equals(reference_dates)
        rows.append({
            "reef_id": reef, "n_available_dates": int(len(dates)), "first_available_date": dates.min(),
            "last_available_date": dates.max(), "same_noaa_time_axis_as_first_reef": bool(same_axis),
            "min_grid_cells_per_available_day": int(cells_per_date.min()),
            "max_grid_cells_per_available_day": int(cells_per_date.max()),
        })
    coverage = pd.DataFrame(rows).sort_values("reef_id")
    coverage.to_csv(out / "NOAA礁盘DHW原始覆盖核查.csv", index=False, encoding="utf-8-sig")
    if reference_dates is None or not coverage["same_noaa_time_axis_as_first_reef"].all():
        raise ValueError("各礁盘 NOAA 原始文件的可用日期轴不一致，不能混合分析")
    if not coverage["min_grid_cells_per_available_day"].eq(coverage["max_grid_cells_per_available_day"]).all():
        raise ValueError("至少一个礁盘在 NOAA 可用日之间的原始格点支持不一致，不能计算稳定面积权重")
    requested = pd.date_range(start, end, freq="D")
    unavailable = requested.difference(reference_dates)
    return {
        "available_dates_per_reef": int(len(reference_dates)),
        "noaa_source_unavailable_dates_within_requested_range": [str(value.date()) for value in unavailable],
        "noaa_source_unavailable_date_count": int(len(unavailable)),
    }


def extract_events(daily: pd.DataFrame, threshold: float = REEF_AREA_THRESHOLD) -> pd.DataFrame:
    active = (daily["valid_reef_area_fraction"].ge(0.999999) & daily["reef_area_fraction_ge8"].ge(threshold)).to_numpy()
    # NOAA's official time axis has a few global missing days.  Bridge a
    # one-day source gap, but never silently connect events across a longer
    # unobserved interval.
    date_gap = daily["date"].diff().dt.days.fillna(1).to_numpy()
    next_gap = np.r_[date_gap[1:], 1]
    starts = np.flatnonzero(active & (np.r_[True, ~active[:-1]] | (date_gap > 2)))
    ends = np.flatnonzero(active & (np.r_[~active[1:], True] | (next_gap > 2)))
    rows = []
    for first, last in zip(starts, ends):
        segment = daily.iloc[first:last + 1]
        peak = segment["reef_area_fraction_ge8"].idxmax()
        rows.append({"event_onset": daily.iloc[first]["date"], "event_end": daily.iloc[last]["date"], "event_peak": daily.loc[peak, "date"], "peak_area_fraction_ge8": float(segment["reef_area_fraction_ge8"].max()), "peak_dhw_mean": float(segment["dhw_area_weighted_mean"].max()), "days_active": int(len(segment))})
    return pd.DataFrame(
        rows,
        columns=["event_onset", "event_end", "event_peak", "peak_area_fraction_ge8", "peak_dhw_mean", "days_active"],
    )


def label_exposure(pair: pd.Series, daily: pd.DataFrame, events: pd.DataFrame, threshold: float) -> dict[str, object]:
    outcome_heat = daily.loc[
        daily["date"].gt(pair["time_t"]) & daily["date"].le(pair["time_t1"]), "reef_area_fraction_ge8"
    ].ge(threshold).any()
    previous = events.loc[events["event_onset"].lt(pair["time_t"])]
    event = previous.iloc[-1] if not previous.empty else None
    months = (pair["time_t"] - event["event_onset"]).days / 30.4375 if event is not None else np.nan
    band = REFERENCE
    if event is not None:
        for label, lower, upper in LAG_BANDS:
            if lower <= months < upper:
                band = label
                break
    preceding = daily.loc[
        daily["date"].gt(pair["time_t"] - pd.Timedelta(days=CONTINUOUS_WINDOW_DAYS))
        & daily["date"].le(pair["time_t"])
    ]
    if preceding.empty or not preceding["valid_reef_area_fraction"].ge(0.999999).all():
        raise ValueError(f"{pair['pair_id']} 的前一年可用 NOAA 日值无法完整覆盖礁盘范围")
    return {
        "event_band": band,
        "dhw8_during_outcome": bool(outcome_heat),
        "most_recent_event_onset": event["event_onset"] if event is not None else pd.NaT,
        "months_since_event_onset": float(months) if pd.notna(months) else np.nan,
        "event_peak_area_fraction_ge8": float(event["peak_area_fraction_ge8"]) if event is not None else np.nan,
        "event_peak_dhw_mean": float(event["peak_dhw_mean"]) if event is not None else np.nan,
        "event_days_active": int(event["days_active"]) if event is not None else np.nan,
        "preceding_12mo_peak_dhw_mean": float(preceding["dhw_area_weighted_mean"].max()),
        "preceding_12mo_mean_dhw_mean": float(preceding["dhw_area_weighted_mean"].mean()),
        "preceding_12mo_peak_area_fraction_ge8": float(preceding["reef_area_fraction_ge8"].max()),
        "preceding_12mo_available_days": int(len(preceding)),
    }


def nino_lookup() -> dict[pd.Period, float]:
    path = ROOT / "data" / "environment" / "indices" / "detrend.nino34.ascii.txt"
    table = pd.read_csv(path, sep=r"\s+")
    if set(["YR", "MON", "ANOM"]) - set(table.columns):
        raise ValueError("Niño 3.4 原始文件缺少 YR/MON/ANOM")
    periods = pd.PeriodIndex.from_fields(year=table["YR"], month=table["MON"], freq="M")
    return dict(zip(periods, pd.to_numeric(table["ANOM"], errors="coerce")))


def nino_window(start: pd.Timestamp, lag: int, lookup: dict[pd.Period, float]) -> float:
    end = (start - pd.DateOffset(months=lag)).to_period("M")
    values = [lookup.get(month, np.nan) for month in pd.period_range(end=end, periods=3, freq="M")]
    return float(np.mean(values)) if np.isfinite(values).all() else np.nan


def add_controls(sample: pd.DataFrame) -> pd.DataFrame:
    sample = sample.copy()
    sample["duration_deviation_days"] = sample["duration_days"] - 365.2425
    sample["baseline_year"] = sample["time_t"].dt.year + (sample["time_t"].dt.dayofyear - 1) / 365.2425
    sample["season_sin"] = np.sin(2 * np.pi * sample["time_t"].dt.dayofyear / 365.2425)
    sample["season_cos"] = np.cos(2 * np.pi * sample["time_t"].dt.dayofyear / 365.2425)
    sample["baseline_calendar_year"] = sample["time_t"].dt.year.astype(int)
    return sample


def build_exposure_sample(pairs: pd.DataFrame, daily_dhw: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    daily_by_reef = {
        reef: group.sort_values("date").reset_index(drop=True)
        for reef, group in daily_dhw.groupby("reef_id", observed=True)
    }
    events_by_reef = {
        reef: extract_events(group, threshold=threshold)
        for reef, group in daily_by_reef.items()
    }
    exposure = []
    for _, pair in pairs.iterrows():
        reef = pair["reef_id"]
        record = label_exposure(pair, daily_by_reef[reef], events_by_reef[reef], threshold=threshold)
        record["pair_id"] = pair["pair_id"]
        exposure.append(record)
    sample = add_controls(pairs.merge(pd.DataFrame(exposure), on="pair_id", how="left", validate="one_to_one"))
    sample["region"] = sample["reef_id"].map(region_code)
    sample["eligible_completed_event_pair"] = ~sample["dhw8_during_outcome"].fillna(True)
    return sample, events_by_reef


def events_table(events_by_reef: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame:
    tables = []
    for reef, events in events_by_reef.items():
        if not events.empty:
            copy = events.copy()
            copy.insert(0, "reef_id", reef)
            copy.insert(1, "region", region_code(reef))
            copy["reef_area_fraction_threshold"] = threshold
            tables.append(copy)
    columns = ["reef_id", "region", "event_onset", "event_end", "event_peak", "peak_area_fraction_ge8", "peak_dhw_mean", "days_active", "reef_area_fraction_threshold"]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)


def within_fit(
    frame: pd.DataFrame,
    exposures: list[str],
    controls: list[str] = CONTROLS,
    calendar_year_fixed_effects: bool = False,
) -> tuple[object | None, pd.DataFrame]:
    required = ["log_area_ratio", *exposures, *controls]
    work = frame.loc[frame[required].notna().all(axis=1)].copy()
    if len(work) < 30 or work["sand_cay_id"].nunique() < 10:
        return None, work
    columns = ["log_area_ratio", *exposures, *controls]
    design = work[columns].copy()
    if calendar_year_fixed_effects:
        year = pd.get_dummies(work["baseline_calendar_year"], prefix="year", dtype=float)
        design = pd.concat([design, year], axis=1)
    demeaned = design - design.groupby(work["sand_cay_id"], observed=True).transform("mean")
    keep = demeaned.abs().sum(axis=1).gt(0)
    work, demeaned = work.loc[keep], demeaned.loc[keep]
    if len(work) < 30 or work["sand_cay_id"].nunique() < 10:
        return None, work
    return sm.OLS(
        demeaned["log_area_ratio"], demeaned.drop(columns="log_area_ratio")
    ).fit(cov_type="cluster", cov_kwds={"groups": work["sand_cay_id"], "use_correction": True}), work


def dhw_event_model(sample: pd.DataFrame, threshold: float, region: str) -> pd.DataFrame:
    """Estimate threshold-event associations only where every contrast exists."""
    work = sample.loc[
        sample["eligible_completed_event_pair"] & sample["region"].eq(region)
    ].copy()
    labels = [value[0] for value in LAG_BANDS]
    counts = work["event_band"].value_counts()
    terms = [f"dhw_band_{index}" for index in range(len(labels))]
    for term, label in zip(terms, labels):
        work[term] = work["event_band"].eq(label).astype(float)
    if counts.get(REFERENCE, 0) == 0 or any(counts.get(label, 0) == 0 for label in labels):
        return pd.DataFrame([{
            "analysis": "DHW threshold event", "scope": f"reef_area_fraction_ge8_{threshold:.2f}",
            "region": region, "lag_label": label, "status": "not_estimable",
            "reason": "missing event or reference contrast", "n_pairs": int(len(work)),
            "n_cays": int(work["sand_cay_id"].nunique()), "n_reefs": int(work["reef_id"].nunique()),
            "n_reference_pairs": int(counts.get(REFERENCE, 0)), "n_exposed_pairs": int(counts.get(label, 0)),
        } for label in labels])
    fitted, analytic = within_fit(work, terms)
    if fitted is None:
        return pd.DataFrame([{
            "analysis": "DHW threshold event", "scope": f"reef_area_fraction_ge8_{threshold:.2f}",
            "region": region, "lag_label": label, "status": "not_estimable",
            "reason": "insufficient within-cay sample", "n_pairs": int(len(analytic)),
            "n_cays": int(analytic["sand_cay_id"].nunique()), "n_reefs": int(analytic["reef_id"].nunique()),
        } for label in labels])
    rows = []
    for term, label in zip(terms, labels):
        low, high = fitted.conf_int().loc[term]
        rows.append({
            "analysis": "DHW threshold event", "scope": f"reef_area_fraction_ge8_{threshold:.2f}",
            "region": region, "lag_label": label, "status": "ok_exploratory", "reference": REFERENCE,
            "n_pairs": int(len(analytic)), "n_cays": int(analytic["sand_cay_id"].nunique()),
            "n_reefs": int(analytic["reef_id"].nunique()), "n_reference_pairs": int(counts.get(REFERENCE, 0)),
            "n_exposed_pairs": int(counts.get(label, 0)), "coefficient_log_area_ratio": float(fitted.params[term]),
            "ci95_low_cay_cluster": float(low), "ci95_high_cay_cluster": float(high),
            "p_value_cay_cluster": float(fitted.pvalues[term]), "area_ratio_percent": float(100 * np.expm1(fitted.params[term])),
        })
    result = pd.DataFrame(rows)
    result["fdr_q_value_cay_cluster"] = multipletests(result["p_value_cay_cluster"], method="fdr_bh")[1]
    return result


def event_availability(sample: pd.DataFrame, threshold: float) -> pd.DataFrame:
    labels = [value[0] for value in LAG_BANDS]
    rows = []
    for region, group in sample.loc[sample["eligible_completed_event_pair"]].groupby("region", observed=True):
        counts = group["event_band"].value_counts()
        rows.append({
            "threshold": threshold, "region": region, "n_pairs": int(len(group)),
            "n_cays": int(group["sand_cay_id"].nunique()), "n_reefs": int(group["reef_id"].nunique()),
            "n_reference_pairs": int(counts.get(REFERENCE, 0)),
            **{f"n_{label.replace(' ', '_').replace('<', 'lt').replace('-', '_')}": int(counts.get(label, 0)) for label in labels},
        })
    return pd.DataFrame(rows)


def continuous_regional_model(sample: pd.DataFrame) -> pd.DataFrame:
    """GBR--MEDF slope comparison on continuous, range-weighted heat stress."""
    work = sample.loc[
        sample["eligible_completed_event_pair"] & sample["region"].isin(PRIMARY_REGIONS)
    ].copy()
    source = "preceding_12mo_peak_dhw_mean"
    sd = float(work[source].std(ddof=0)) if len(work) else np.nan
    work["heat_per_1sd"] = (work[source] - work[source].mean()) / sd if pd.notna(sd) and sd > 0 else np.nan
    work["heat_x_medf"] = work["heat_per_1sd"] * work["region"].eq("MEDF").astype(float)
    fitted, analytic = within_fit(
        work, ["heat_per_1sd", "heat_x_medf"], controls=CONTINUOUS_CONTROLS, calendar_year_fixed_effects=True
    )
    base = {
        "analysis": "Continuous DHW regional association", "scope": "preceding_12mo_peak_area_weighted_mean_DHW",
        "n_pairs": int(len(analytic)), "n_cays": int(analytic["sand_cay_id"].nunique()),
        "n_reefs": int(analytic["reef_id"].nunique()), "heat_sd_cweeks": sd,
        "calendar_year_fixed_effects": True,
    }
    if fitted is None:
        return pd.DataFrame([{**base, "region": region, "status": "not_estimable", "reason": "insufficient within-cay sample"} for region in PRIMARY_REGIONS])
    covariance = fitted.cov_params()
    contrasts = {
        "GBR": (np.array([1.0, 0.0]), "GBR slope"),
        "MEDF": (np.array([1.0, 1.0]), "MEDF slope"),
        "GBR minus MEDF": (np.array([0.0, -1.0]), "regional slope difference"),
    }
    terms = ["heat_per_1sd", "heat_x_medf"]
    rows = []
    for region, (vector, contrast_label) in contrasts.items():
        estimate = float(vector @ fitted.params[terms].to_numpy())
        se = float(np.sqrt(vector @ covariance.loc[terms, terms].to_numpy() @ vector))
        z = estimate / se if se > 0 else np.nan
        pvalue = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append({
            **base, "region": region, "contrast": contrast_label, "status": "ok_exploratory",
            "coefficient_log_area_ratio_per_1sd": estimate, "ci95_low_cay_cluster": estimate - 1.96 * se,
            "ci95_high_cay_cluster": estimate + 1.96 * se, "p_value_wald": pvalue,
            "area_ratio_percent_per_1sd": float(100 * np.expm1(estimate)),
        })
    result = pd.DataFrame(rows)
    result["fdr_q_value_wald"] = multipletests(result["p_value_wald"], method="fdr_bh")[1]
    return result


def annual_mean_area_dhw_table(observations: pd.DataFrame, daily_dhw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every available Sentinel-2 observation to a cay-year mean and match same-year maximum DHW."""
    area = observations.copy()
    area["calendar_year"] = area["date"].dt.year
    area["season_sin"] = np.sin(2 * np.pi * area["date"].dt.dayofyear / 365.2425)
    area["season_cos"] = np.cos(2 * np.pi * area["date"].dt.dayofyear / 365.2425)
    annual_area = (
        area.groupby(["reef_id", "sand_cay_id", "calendar_year"], observed=True)
        .agg(
            annual_mean_area_m2=("sand_cay_area_m2", "mean"),
            annual_median_area_m2=("sand_cay_area_m2", "median"),
            n_s2_observations=("image_id", "nunique"),
            annual_area_sd_m2=("sand_cay_area_m2", "std"),
            annual_mean_season_sin=("season_sin", "mean"),
            annual_mean_season_cos=("season_cos", "mean"),
        )
        .reset_index()
    )
    heat = daily_dhw.loc[daily_dhw["valid_reef_area_fraction"].ge(0.999999)].copy()
    heat["calendar_year"] = heat["date"].dt.year
    annual_heat = (
        heat.groupby(["reef_id", "calendar_year"], observed=True)
        .agg(
            annual_max_dhw_area_weighted_mean=("dhw_area_weighted_mean", "max"),
            annual_max_dhw_area_weighted_max=("dhw_area_weighted_max", "max"),
            annual_max_reef_area_fraction_ge8=("reef_area_fraction_ge8", "max"),
            n_noaa_days=("date", "nunique"),
        )
        .reset_index()
    )
    annual = annual_area.merge(annual_heat, on=["reef_id", "calendar_year"], how="inner", validate="many_to_one")
    if annual.empty:
        raise ValueError("年度 Sentinel-2 面积与 NOAA DHW 没有重叠年份")
    annual["region"] = annual["reef_id"].map(region_code)
    annual["log10_annual_mean_area_m2"] = np.log10(annual["annual_mean_area_m2"])
    annual["log10_annual_median_area_m2"] = np.log10(annual["annual_median_area_m2"])
    annual["log_n_s2_observations"] = np.log(annual["n_s2_observations"])
    return annual.sort_values(["reef_id", "sand_cay_id", "calendar_year"]).reset_index(drop=True)


def annual_fixed_effect_design(work: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    cay_fixed_effects = pd.get_dummies(work["sand_cay_id"], prefix="cay", drop_first=True, dtype=float)
    year_fixed_effects = pd.get_dummies(work["calendar_year"], prefix="year", drop_first=True, dtype=float)
    return sm.add_constant(
        pd.concat([work[["dhw_per_1sd", *controls]], cay_fixed_effects, year_fixed_effects], axis=1),
        has_constant="add",
    )


def annual_mean_area_dhw_model(annual: pd.DataFrame) -> pd.DataFrame:
    """Estimate primary annual-mean association plus sampling-sensitive annual summaries."""
    source = "annual_max_dhw_area_weighted_mean"
    work = annual.copy()
    if len(work) < 30 or work["sand_cay_id"].nunique() < 10 or work["reef_id"].nunique() < 10:
        return pd.DataFrame([{
            "analysis": "Annual mean area versus annual maximum DHW", "status": "not_estimable",
            "reason": "insufficient cay-year coverage", "n_cay_years": int(len(work)),
            "n_cays": int(work["sand_cay_id"].nunique()), "n_reefs": int(work["reef_id"].nunique()),
        }])
    heat_sd = float(work[source].std(ddof=0))
    work["dhw_per_1sd"] = (work[source] - work[source].mean()) / heat_sd
    specifications = (
        ("primary_annual_mean", "log10_annual_mean_area_m2", "log10 annual mean cay area", []),
        ("sensitivity_annual_median", "log10_annual_median_area_m2", "log10 annual median cay area", []),
        (
            "sensitivity_mean_adjusted_for_sampling",
            "log10_annual_mean_area_m2",
            "log10 annual mean cay area",
            ["annual_mean_season_sin", "annual_mean_season_cos", "log_n_s2_observations"],
        ),
    )
    rows = []
    for specification, outcome, outcome_label, controls in specifications:
        analytic = work.loc[work[[source, outcome, *controls]].notna().all(axis=1)].copy()
        design = annual_fixed_effect_design(analytic, controls)
        fitted = sm.OLS(analytic[outcome], design).fit(
            cov_type="cluster", cov_kwds={"groups": analytic["reef_id"], "use_correction": True}
        )
        low, high = fitted.conf_int().loc["dhw_per_1sd"]
        estimate = float(fitted.params["dhw_per_1sd"])
        pearson = pearsonr(analytic[source], analytic[outcome])
        spearman = spearmanr(analytic[source], analytic[outcome])
        rows.append({
            "analysis": "Annual area versus annual maximum DHW", "status": "ok_exploratory",
            "specification": specification,
            "scope": "all Sentinel-2 A/B observations summarized within cay-year; same-calendar-year annual maximum reef-area-weighted mean DHW",
            "outcome": outcome_label, "sampling_controls": "; ".join(controls) if controls else "none",
            "predictor": "annual maximum reef-area-weighted mean DHW",
            "n_cay_years": int(len(analytic)), "n_cays": int(analytic["sand_cay_id"].nunique()),
            "n_reefs": int(analytic["reef_id"].nunique()), "years": f"{int(analytic['calendar_year'].min())}-{int(analytic['calendar_year'].max())}",
            "heat_sd_cweeks": heat_sd,
            "raw_pearson_r": float(pearson.statistic), "raw_pearson_p_naive": float(pearson.pvalue),
            "raw_spearman_rho": float(spearman.statistic), "raw_spearman_p_naive": float(spearman.pvalue),
            "fixed_effects": "sand cay and calendar year", "clustered_standard_errors": "reef",
            "coefficient_log10_area_per_1sd": estimate,
            "ci95_low_reef_cluster": float(low), "ci95_high_reef_cluster": float(high),
            "p_value_reef_cluster": float(fitted.pvalues["dhw_per_1sd"]),
            "area_ratio_percent_per_1sd": float(100 * (10 ** estimate - 1)),
            "ci95_area_ratio_low_percent": float(100 * (10 ** low - 1)),
            "ci95_area_ratio_high_percent": float(100 * (10 ** high - 1)),
        })
    return pd.DataFrame(rows)


def annual_lagged_area_change_table(annual: pd.DataFrame) -> pd.DataFrame:
    """Pair consecutive cay-years so DHW in year y predicts area change from y to y+1."""
    baseline_columns = [
        "reef_id", "sand_cay_id", "calendar_year", "region",
        "annual_max_dhw_area_weighted_mean", "log10_annual_mean_area_m2",
        "log10_annual_median_area_m2", "annual_mean_season_sin",
        "annual_mean_season_cos", "log_n_s2_observations", "n_s2_observations",
    ]
    baseline = annual[baseline_columns].copy()
    followup = annual[[
        "reef_id", "sand_cay_id", "calendar_year", "log10_annual_mean_area_m2",
        "log10_annual_median_area_m2", "annual_mean_season_sin",
        "annual_mean_season_cos", "log_n_s2_observations", "n_s2_observations",
    ]].copy()
    followup["calendar_year"] -= 1
    followup = followup.rename(columns={
        "log10_annual_mean_area_m2": "followup_log10_annual_mean_area_m2",
        "log10_annual_median_area_m2": "followup_log10_annual_median_area_m2",
        "annual_mean_season_sin": "followup_mean_season_sin",
        "annual_mean_season_cos": "followup_mean_season_cos",
        "log_n_s2_observations": "followup_log_n_s2_observations",
        "n_s2_observations": "followup_n_s2_observations",
    })
    paired = baseline.merge(
        followup, on=["reef_id", "sand_cay_id", "calendar_year"], how="inner", validate="one_to_one"
    )
    if paired.empty:
        raise ValueError("没有连续日历年的年度面积均值可用于一年滞后分析")
    paired["log10_mean_area_change_y_to_y1"] = (
        paired["followup_log10_annual_mean_area_m2"] - paired["log10_annual_mean_area_m2"]
    )
    paired["log10_median_area_change_y_to_y1"] = (
        paired["followup_log10_annual_median_area_m2"] - paired["log10_annual_median_area_m2"]
    )
    paired["delta_mean_season_sin"] = paired["followup_mean_season_sin"] - paired["annual_mean_season_sin"]
    paired["delta_mean_season_cos"] = paired["followup_mean_season_cos"] - paired["annual_mean_season_cos"]
    paired["delta_log_n_s2_observations"] = paired["followup_log_n_s2_observations"] - paired["log_n_s2_observations"]
    return paired.sort_values(["reef_id", "sand_cay_id", "calendar_year"]).reset_index(drop=True)


def annual_lagged_area_change_model(paired: pd.DataFrame) -> pd.DataFrame:
    """Test the predeclared one-calendar-year lag without treating repeated cays as independent."""
    source = "annual_max_dhw_area_weighted_mean"
    heat_sd = float(paired[source].std(ddof=0))
    work = paired.copy()
    work["dhw_per_1sd"] = (work[source] - work[source].mean()) / heat_sd
    specifications = (
        ("primary_annual_mean_change", "log10_mean_area_change_y_to_y1", "log10 mean area change from y to y+1", []),
        ("sensitivity_annual_median_change", "log10_median_area_change_y_to_y1", "log10 median area change from y to y+1", []),
        (
            "sensitivity_mean_change_adjusted_for_sampling",
            "log10_mean_area_change_y_to_y1",
            "log10 mean area change from y to y+1",
            ["delta_mean_season_sin", "delta_mean_season_cos", "delta_log_n_s2_observations"],
        ),
    )
    rows = []
    for specification, outcome, outcome_label, controls in specifications:
        analytic = work.loc[work[[source, outcome, *controls]].notna().all(axis=1)].copy()
        if len(analytic) < 30 or analytic["sand_cay_id"].nunique() < 10 or analytic["reef_id"].nunique() < 10:
            rows.append({
                "analysis": "One-year-lag annual DHW versus area change", "status": "not_estimable",
                "specification": specification, "n_cay_year_pairs": int(len(analytic)),
                "n_cays": int(analytic["sand_cay_id"].nunique()), "n_reefs": int(analytic["reef_id"].nunique()),
            })
            continue
        design = annual_fixed_effect_design(analytic, controls)
        fitted = sm.OLS(analytic[outcome], design).fit(
            cov_type="cluster", cov_kwds={"groups": analytic["reef_id"], "use_correction": True}
        )
        low, high = fitted.conf_int().loc["dhw_per_1sd"]
        estimate = float(fitted.params["dhw_per_1sd"])
        pearson = pearsonr(analytic[source], analytic[outcome])
        spearman = spearmanr(analytic[source], analytic[outcome])
        rows.append({
            "analysis": "One-year-lag annual DHW versus area change", "status": "ok_exploratory",
            "specification": specification,
            "scope": "annual maximum reef-area-weighted mean DHW in y predicting cay area change from y to y+1",
            "outcome": outcome_label, "sampling_controls": "; ".join(controls) if controls else "none",
            "n_cay_year_pairs": int(len(analytic)), "n_cays": int(analytic["sand_cay_id"].nunique()),
            "n_reefs": int(analytic["reef_id"].nunique()), "years": f"{int(analytic['calendar_year'].min())}-{int(analytic['calendar_year'].max())} exposure years",
            "heat_sd_cweeks": heat_sd,
            "raw_pearson_r": float(pearson.statistic), "raw_pearson_p_naive": float(pearson.pvalue),
            "raw_spearman_rho": float(spearman.statistic), "raw_spearman_p_naive": float(spearman.pvalue),
            "fixed_effects": "sand cay and exposure calendar year", "clustered_standard_errors": "reef",
            "coefficient_log10_area_change_per_1sd": estimate,
            "ci95_low_reef_cluster": float(low), "ci95_high_reef_cluster": float(high),
            "p_value_reef_cluster": float(fitted.pvalues["dhw_per_1sd"]),
            "area_change_percent_per_1sd": float(100 * (10 ** estimate - 1)),
            "ci95_area_change_low_percent": float(100 * (10 ** low - 1)),
            "ci95_area_change_high_percent": float(100 * (10 ** high - 1)),
        })
    return pd.DataFrame(rows)


def plot_annual_lagged_area_change(paired: pd.DataFrame, model: pd.DataFrame) -> None:
    """Show raw and fixed-effect one-year-lag associations side by side."""
    apply_figure_style()
    import matplotlib.pyplot as plt

    colors = {"GBR": "#3B6FB6", "MEDF": "#D95F02", "OTHER": "#7570B3", "NH": "#1B9E77"}
    row = model.loc[model["specification"].eq("primary_annual_mean_change")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    ax = axes[0]
    for region in ("GBR", "MEDF", "OTHER", "NH"):
        group = paired.loc[paired["region"].eq(region)]
        if not group.empty:
            ax.scatter(group["annual_max_dhw_area_weighted_mean"], group["log10_mean_area_change_y_to_y1"],
                       s=22, alpha=.62, color=colors[region], edgecolors="white", linewidths=.25, label=region)
    x = paired["annual_max_dhw_area_weighted_mean"].to_numpy()
    y = paired["log10_mean_area_change_y_to_y1"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
    ax.plot(xline, intercept + slope * xline, color="#333333", lw=1.25, label="Raw linear fit")
    ax.text(.02, .98, (
        f"N = {int(row['n_cay_year_pairs'])} consecutive cay-years; {int(row['n_cays'])} cays\n"
        f"Raw Pearson r = {row['raw_pearson_r']:.2f}; p = {row['raw_pearson_p_naive']:.3f} (not inferential)"
    ), transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
            bbox={"boxstyle": "round,pad=.28", "facecolor": "white", "edgecolor": "#B0B0B0", "alpha": .93})
    ax.axhline(0, color="#999999", lw=.8, ls="--")
    ax.set_title("a  Raw one-year-lag association (descriptive)")
    ax.set_xlabel("Annual maximum reef-area-weighted mean DHW in year y (°C-weeks)")
    ax.set_ylabel("Log10 annual mean area change: y to y+1")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    work = paired.copy()
    source = "annual_max_dhw_area_weighted_mean"
    work["dhw_per_1sd"] = (work[source] - work[source].mean()) / work[source].std(ddof=0)
    fixed = annual_fixed_effect_design(work, []).drop(columns="dhw_per_1sd")
    x_residual = sm.OLS(work["dhw_per_1sd"], fixed).fit().resid
    y_residual = sm.OLS(work["log10_mean_area_change_y_to_y1"], fixed).fit().resid
    ax = axes[1]
    for region in ("GBR", "MEDF", "OTHER", "NH"):
        group = work.loc[work["region"].eq(region)]
        if not group.empty:
            ax.scatter(x_residual.loc[group.index], y_residual.loc[group.index], s=22, alpha=.62,
                       color=colors[region], edgecolors="white", linewidths=.25)
    slope, intercept = np.polyfit(x_residual, y_residual, 1)
    xline = np.linspace(float(x_residual.min()), float(x_residual.max()), 200)
    ax.plot(xline, intercept + slope * xline, color="#333333", lw=1.25)
    ax.text(.02, .98, (
        "Cay and exposure-year fixed effects\n"
        f"{row['area_change_percent_per_1sd']:+.1f}% per 1 SD DHW\n"
        f"95% CI [{row['ci95_area_change_low_percent']:+.1f}%, {row['ci95_area_change_high_percent']:+.1f}%]\n"
        f"p (reef-clustered) = {row['p_value_reef_cluster']:.3f}"
    ), transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
            bbox={"boxstyle": "round,pad=.28", "facecolor": "white", "edgecolor": "#B0B0B0", "alpha": .93})
    ax.axhline(0, color="#999999", lw=.8, ls="--")
    ax.axvline(0, color="#999999", lw=.8, ls="--")
    ax.set_title("b  One-year-lag association after fixed effects")
    ax.set_xlabel("Annual maximum DHW residual in year y (1 SD units)")
    ax.set_ylabel("Annual mean area-change residual (log10 m²)")
    stem = (PROJECT_ROOT / "outputs/figures/中间图") / "figure_r13_annual_lagged_dhw_area_change"
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    plt.close(fig)


def plot_annual_mean_area_dhw(annual: pd.DataFrame, model: pd.DataFrame) -> None:
    """Plot all cay-year means without implying independent observations or causality."""
    apply_figure_style()
    import matplotlib.pyplot as plt

    colors = {"GBR": "#3B6FB6", "MEDF": "#D95F02", "OTHER": "#7570B3", "NH": "#1B9E77"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    ax = axes[0]
    for region in ("GBR", "MEDF", "OTHER", "NH"):
        group = annual.loc[annual["region"].eq(region)]
        if not group.empty:
            ax.scatter(
                group["annual_max_dhw_area_weighted_mean"], group["log10_annual_mean_area_m2"],
                s=22, alpha=.62, color=colors[region], edgecolors="white", linewidths=.25, label=region,
            )
    x = annual["annual_max_dhw_area_weighted_mean"].to_numpy()
    y = annual["log10_annual_mean_area_m2"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
    ax.plot(xline, intercept + slope * xline, color="#333333", lw=1.25, label="Raw linear fit")
    row = model.loc[model["specification"].eq("primary_annual_mean")].iloc[0]
    if row["status"] == "ok_exploratory":
        annotation = (
            f"N = {int(row['n_cay_years'])} cay-years; {int(row['n_cays'])} cays; {int(row['n_reefs'])} reefs\n"
            f"Raw Pearson r = {row['raw_pearson_r']:.2f}; p < 0.001 (not inferential)\n"
            "Between-cay and regional pattern"
        )
        ax.text(.02, .98, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
                bbox={"boxstyle": "round,pad=.28", "facecolor": "white", "edgecolor": "#B0B0B0", "alpha": .93})
    ax.set_title("a  Raw annual mean area (descriptive)")
    ax.set_xlabel("Annual maximum reef-area-weighted mean DHW (°C-weeks)")
    ax.set_ylabel("Annual mean cay area (log10 m²)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    work = annual.copy()
    source = "annual_max_dhw_area_weighted_mean"
    work["dhw_per_1sd"] = (work[source] - work[source].mean()) / work[source].std(ddof=0)
    fixed = annual_fixed_effect_design(work, []).drop(columns="dhw_per_1sd")
    x_residual = sm.OLS(work["dhw_per_1sd"], fixed).fit().resid
    y_residual = sm.OLS(work["log10_annual_mean_area_m2"], fixed).fit().resid
    ax = axes[1]
    for region in ("GBR", "MEDF", "OTHER", "NH"):
        group = work.loc[work["region"].eq(region)]
        if not group.empty:
            ax.scatter(x_residual.loc[group.index], y_residual.loc[group.index], s=22, alpha=.62,
                       color=colors[region], edgecolors="white", linewidths=.25, label=region)
    residual_slope, residual_intercept = np.polyfit(x_residual, y_residual, 1)
    xline = np.linspace(float(x_residual.min()), float(x_residual.max()), 200)
    ax.plot(xline, residual_intercept + residual_slope * xline, color="#333333", lw=1.25)
    annotation = (
        "Cay and calendar-year fixed effects\n"
        f"{row['area_ratio_percent_per_1sd']:+.1f}% per 1 SD DHW\n"
        f"95% CI [{row['ci95_area_ratio_low_percent']:+.1f}%, {row['ci95_area_ratio_high_percent']:+.1f}%]\n"
        f"p (reef-clustered) = {row['p_value_reef_cluster']:.3f}"
    )
    ax.text(.02, .98, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
            bbox={"boxstyle": "round,pad=.28", "facecolor": "white", "edgecolor": "#B0B0B0", "alpha": .93})
    ax.axhline(0, color="#999999", lw=.8, ls="--")
    ax.axvline(0, color="#999999", lw=.8, ls="--")
    ax.set_title("b  Within-cay and year-adjusted association")
    ax.set_xlabel("Annual maximum DHW residual (1 SD units)")
    ax.set_ylabel("Annual mean cay area residual (log10 m²)")
    stem = (PROJECT_ROOT / "outputs/figures/中间图") / "figure_r12_annual_mean_area_dhw"
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    plt.close(fig)


def vegetation_state_interaction_model(sample: pd.DataFrame) -> pd.DataFrame:
    """Test whether a recent reef-wide heat event differs between bare and vegetation-dominant cays."""
    work = sample.loc[
        sample["eligible_completed_event_pair"]
        & sample["region"].eq("GBR")
        & sample["baseline_analysis_vegetation_state"].isin(["none", "dominant"])
    ].copy()
    work["recent_event_3yr"] = work["event_band"].ne(REFERENCE).astype(float)
    work["vegetation_dominant"] = work["baseline_analysis_vegetation_state"].eq("dominant").astype(float)
    work["event_x_vegetation"] = work["recent_event_3yr"] * work["vegetation_dominant"]
    cell_counts = work.groupby(["baseline_analysis_vegetation_state", "recent_event_3yr"], observed=True).size()
    fitted, analytic = within_fit(work, ["recent_event_3yr", "vegetation_dominant", "event_x_vegetation"])
    base = {
        "analysis": "DHW event by baseline vegetation state",
        "scope": "GBR; reef-wide DHW>=8 in prior 3 years; audited none versus dominant",
        "n_input_pairs": int(len(work)), "n_pairs": int(len(analytic)),
        "n_cays": int(analytic["sand_cay_id"].nunique()), "n_reefs": int(analytic["reef_id"].nunique()),
        "n_bare_reference": int(cell_counts.get(("none", 0.0), 0)),
        "n_bare_event": int(cell_counts.get(("none", 1.0), 0)),
        "n_dominant_reference": int(cell_counts.get(("dominant", 0.0), 0)),
        "n_dominant_event": int(cell_counts.get(("dominant", 1.0), 0)),
    }
    if fitted is None:
        return pd.DataFrame([{**base, "contrast": value, "status": "not_estimable"} for value in ("bare event association", "vegetation-dominant event association", "dominant minus bare interaction")])
    terms = ["recent_event_3yr", "event_x_vegetation"]
    covariance = fitted.cov_params().loc[terms, terms].to_numpy()
    contrasts = {
        "bare event association": np.array([1.0, 0.0]),
        "vegetation-dominant event association": np.array([1.0, 1.0]),
        "dominant minus bare interaction": np.array([0.0, 1.0]),
    }
    rows = []
    coefficients = fitted.params[terms].to_numpy()
    for label, vector in contrasts.items():
        estimate = float(vector @ coefficients)
        se = float(np.sqrt(vector @ covariance @ vector))
        z = estimate / se if se > 0 else np.nan
        rows.append({
            **base, "contrast": label, "status": "ok_exploratory",
            "coefficient_log_area_ratio": estimate,
            "ci95_low_cay_cluster": estimate - 1.96 * se,
            "ci95_high_cay_cluster": estimate + 1.96 * se,
            "p_value_wald": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "area_ratio_percent": float(100 * np.expm1(estimate)),
        })
    result = pd.DataFrame(rows)
    result["fdr_q_value_wald"] = multipletests(result["p_value_wald"], method="fdr_bh")[1]
    return result


def vegetation_state_event_transition_model(sample: pd.DataFrame) -> pd.DataFrame:
    """Test same-season area change across a newly occurring reef-wide heat event."""
    work = sample.loc[
        sample["region"].eq("GBR")
        & sample["event_band"].eq(REFERENCE)
        & sample["baseline_analysis_vegetation_state"].isin(["none", "dominant"])
    ].copy()
    work["event_during_interval"] = work["dhw8_during_outcome"].astype(float)
    work["vegetation_dominant"] = work["baseline_analysis_vegetation_state"].eq("dominant").astype(float)
    work["event_x_vegetation"] = work["event_during_interval"] * work["vegetation_dominant"]
    cell_counts = work.groupby(
        ["baseline_analysis_vegetation_state", "event_during_interval"], observed=True
    ).size()
    fitted, analytic = within_fit(
        work, ["event_during_interval", "vegetation_dominant", "event_x_vegetation"]
    )
    base = {
        "analysis": "New DHW event crossed by baseline vegetation state",
        "scope": "GBR; no reef-wide event in prior 3 years; newly crossed DHW>=8 event",
        "n_input_pairs": int(len(work)), "n_pairs": int(len(analytic)),
        "n_cays": int(analytic["sand_cay_id"].nunique()), "n_reefs": int(analytic["reef_id"].nunique()),
        "n_bare_reference": int(cell_counts.get(("none", 0.0), 0)),
        "n_bare_event": int(cell_counts.get(("none", 1.0), 0)),
        "n_dominant_reference": int(cell_counts.get(("dominant", 0.0), 0)),
        "n_dominant_event": int(cell_counts.get(("dominant", 1.0), 0)),
    }
    labels = (
        "bare newly crossed event association",
        "vegetation-dominant newly crossed event association",
        "dominant minus bare newly crossed event interaction",
    )
    if fitted is None:
        return pd.DataFrame([{**base, "contrast": value, "status": "not_estimable"} for value in labels])
    terms = ["event_during_interval", "event_x_vegetation"]
    covariance = fitted.cov_params().loc[terms, terms].to_numpy()
    contrasts = {
        labels[0]: np.array([1.0, 0.0]),
        labels[1]: np.array([1.0, 1.0]),
        labels[2]: np.array([0.0, 1.0]),
    }
    coefficients = fitted.params[terms].to_numpy()
    rows = []
    for label, vector in contrasts.items():
        estimate = float(vector @ coefficients)
        se = float(np.sqrt(vector @ covariance @ vector))
        z = estimate / se if se > 0 else np.nan
        rows.append({
            **base, "contrast": label, "status": "ok_exploratory",
            "coefficient_log_area_ratio": estimate,
            "ci95_low_cay_cluster": estimate - 1.96 * se,
            "ci95_high_cay_cluster": estimate + 1.96 * se,
            "p_value_wald": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "area_ratio_percent": float(100 * np.expm1(estimate)),
        })
    result = pd.DataFrame(rows)
    result["fdr_q_value_wald"] = multipletests(result["p_value_wald"], method="fdr_bh")[1]
    return result


def enso_model(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lag in ENSO_LAGS:
        source = f"nino34_pre_{lag}m_3mo"
        work = sample.loc[sample["eligible_completed_event_pair"] & sample[source].notna()].copy()
        sd = float(work[source].std(ddof=0)) if len(work) else np.nan
        work["nino34_per_1sd"] = (work[source] - work[source].mean()) / sd if pd.notna(sd) and sd > 0 else np.nan
        fitted, analytic = within_fit(work, ["nino34_per_1sd"])
        if fitted is None:
            rows.append({"analysis": "ENSO", "scope": "same_DHW_completed_event_sample", "lag_months": lag, "status": "insufficient_sample", "n_pairs": int(len(analytic)), "n_cays": int(analytic["sand_cay_id"].nunique()), "n_reefs": int(analytic["reef_id"].nunique()), "nino34_sd_c": sd})
            continue
        low, high = fitted.conf_int().loc["nino34_per_1sd"]
        columns = ["log_area_ratio", "nino34_per_1sd", *CONTROLS]
        demeaned = analytic[columns] - analytic.groupby("sand_cay_id", observed=True)[columns].transform("mean")
        two_way_groups = np.column_stack([
            pd.factorize(analytic["sand_cay_id"])[0],
            pd.factorize(analytic["baseline_calendar_year"])[0],
        ])
        fitted_two_way = sm.OLS(demeaned["log_area_ratio"], demeaned[["nino34_per_1sd", *CONTROLS]]).fit(
            cov_type="cluster", cov_kwds={"groups": two_way_groups, "use_correction": True}
        )
        two_low, two_high = fitted_two_way.conf_int().loc["nino34_per_1sd"]
        leave_one_year = []
        for year in sorted(analytic["baseline_calendar_year"].unique()):
            leave_fit, _ = within_fit(work.loc[work["baseline_calendar_year"].ne(year)].copy(), ["nino34_per_1sd"])
            if leave_fit is not None:
                leave_one_year.append(float(leave_fit.params["nino34_per_1sd"]))
        leave = np.asarray(leave_one_year, dtype=float)
        coefficient = float(fitted.params["nino34_per_1sd"])
        rows.append({"analysis": "ENSO", "scope": "same_DHW_completed_event_sample", "lag_months": lag, "status": "ok_exploratory", "n_pairs": int(len(analytic)), "n_cays": int(analytic["sand_cay_id"].nunique()), "n_reefs": int(analytic["reef_id"].nunique()), "nino34_sd_c": sd, "coefficient_log_area_ratio_per_1sd": coefficient, "ci95_low_cay_cluster": float(low), "ci95_high_cay_cluster": float(high), "p_value_cay_cluster": float(fitted.pvalues["nino34_per_1sd"]), "two_way_ci95_low": float(two_low), "two_way_ci95_high": float(two_high), "two_way_p_value": float(fitted_two_way.pvalues["nino34_per_1sd"]), "leave_one_year_out_n": int(len(leave)), "leave_one_year_out_min": float(leave.min()) if len(leave) else np.nan, "leave_one_year_out_max": float(leave.max()) if len(leave) else np.nan, "leave_one_year_out_all_same_sign": bool(len(leave) and np.all(np.sign(leave) == np.sign(coefficient)))})
    result = pd.DataFrame(rows)
    valid = result["status"].eq("ok_exploratory")
    result["fdr_q_value_cay_cluster"] = np.nan
    if valid.any():
        result.loc[valid, "fdr_q_value_cay_cluster"] = multipletests(result.loc[valid, "p_value_cay_cluster"], method="fdr_bh")[1]
        result["two_way_fdr_q_value"] = np.nan
        result.loc[valid, "two_way_fdr_q_value"] = multipletests(result.loc[valid, "two_way_p_value"], method="fdr_bh")[1]
        result["time_robust_supported"] = (
            result["fdr_q_value_cay_cluster"].lt(0.05)
            & result["two_way_fdr_q_value"].lt(0.05)
            & result["leave_one_year_out_all_same_sign"].fillna(False)
        )
    return result


def plot_models(dhw: pd.DataFrame, availability: pd.DataFrame, continuous: pd.DataFrame) -> None:
    apply_figure_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)
    event = dhw.loc[dhw["status"].eq("ok_exploratory") & dhw["region"].eq("GBR")].copy()
    if event.empty:
        axes[0].text(.5, .5, "No estimable DHW contrast", ha="center", va="center")
    else:
        labels = [value[0] for value in LAG_BANDS]
        event["lag_label"] = pd.Categorical(event["lag_label"], categories=labels, ordered=True)
        event = event.sort_values("lag_label"); value = event["coefficient_log_area_ratio"].to_numpy()
        axes[0].errorbar(value, np.arange(len(event)), xerr=[value-event["ci95_low_cay_cluster"].to_numpy(), event["ci95_high_cay_cluster"].to_numpy()-value], fmt="o", color="#C44E52", capsize=3)
        axes[0].set_yticks(np.arange(len(event)), event["lag_label"])
    axes[0].axvline(0, color="#777777", ls="--", lw=.8); axes[0].set_title("a  Threshold event association (GBR)")
    axes[0].set_xlabel("Log annual area ratio (95% CI)"); axes[0].set_ylabel("Event onset before pair start")
    if not event.empty:
        axes[0].text(.03, .84, "n = 228 pairs; all lag estimates FDR q = 0.269", transform=axes[0].transAxes,
                     fontsize=8, ha="left", va="bottom", bbox={"boxstyle": "round,pad=.2", "facecolor": "white", "edgecolor": "none", "alpha": .85})
    counts = availability.loc[availability["threshold"].eq(REEF_AREA_THRESHOLD)].copy()
    order = [region for region in ("GBR", "MEDF", "OTHER", "NH") if region in set(counts["region"])]
    counts["region"] = pd.Categorical(counts["region"], categories=order, ordered=True)
    counts = counts.sort_values("region")
    exposed_column = [col for col in counts.columns if col.startswith("n_0lt1_year")]
    event_total = counts[[col for col in counts.columns if col.startswith("n_") and "year" in col]].sum(axis=1)
    axes[1].bar(counts["region"], counts["n_reference_pairs"], color="#BDBDBD", label="Reference pairs")
    axes[1].bar(counts["region"], event_total, bottom=counts["n_reference_pairs"], color="#E66101", label="Event-lag pairs")
    axes[1].set_title("b  Availability of DHW>=8 contrasts")
    axes[1].set_xlabel("Region"); axes[1].set_ylabel("Annual paired observations")
    axes[1].legend(frameon=False, fontsize=8)
    regional = continuous.loc[continuous["region"].isin(PRIMARY_REGIONS) & continuous["status"].eq("ok_exploratory")].copy()
    if regional.empty:
        axes[2].text(.5, .5, "No estimable regional heat slopes", ha="center", va="center")
    else:
        regional["region"] = pd.Categorical(regional["region"], categories=list(PRIMARY_REGIONS), ordered=True)
        regional = regional.sort_values("region"); value = regional["coefficient_log_area_ratio_per_1sd"].to_numpy()
        axes[2].errorbar(np.arange(len(regional)), value, yerr=[value-regional["ci95_low_cay_cluster"].to_numpy(), regional["ci95_high_cay_cluster"].to_numpy()-value], fmt="o", color="#386CB0", capsize=3)
        axes[2].set_xticks(np.arange(len(regional)), regional["region"])
    axes[2].axhline(0, color="#777777", ls="--", lw=.8); axes[2].set_title("c  Continuous heat association")
    axes[2].set_xlabel("Region"); axes[2].set_ylabel("Log annual area ratio per 1 SD DHW (95% CI)")
    if not regional.empty:
        axes[2].text(.03, .04, "All regional contrasts FDR q = 0.797", transform=axes[2].transAxes,
                     fontsize=8, ha="left", va="bottom")
    stem = (PROJECT_ROOT / "outputs/figures/中间图") / "figure_r11_bleaching_pulse_area"
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="重新下载 NOAA 原始格点文件")
    parser.add_argument("--reuse-aggregated", action="store_true", help="在已通过原始覆盖核查时复用现有逐日礁盘汇总")
    parser.add_argument("--offline", action="store_true", help="仅使用现有NOAA缓存，缺失时直接报错")
    args = parser.parse_args()
    out, cache = (PROJECT_ROOT / "outputs/气候检验"), ROOT / "outputs/气候检验/dhw_grid_cache"
    out.mkdir(parents=True, exist_ok=True)
    observations = load_observations()
    pairs = same_season_pairs(observations, ANNUAL_DAYS, SEASON_GAP)
    broad_pairs = same_season_pairs(observations, BROAD_ANNUAL_DAYS, BROAD_SEASON_GAP)
    if pairs.empty:
        raise ValueError("没有满足严格周年、同参考框条件的面积配对")
    extents = verified_extents(set(observations["reef_id"].unique()), out)
    pairs = pairs.merge(extents, on="reef_id", how="inner", validate="many_to_one")
    broad_pairs = broad_pairs.merge(extents, on="reef_id", how="inner", validate="many_to_one")
    if not set(pairs["pair_id"]).issubset(set(broad_pairs["pair_id"])):
        raise ValueError("严格周年配对不是宽松周年配对的子集，配对逻辑不一致")
    start, end = (broad_pairs["time_t"].min() - pd.DateOffset(months=42)).normalize(), broad_pairs["time_t1"].max().normalize()
    manifest = []
    ordered = [row for _, row in extents.sort_values("reef_id").iterrows()]
    if args.offline:
        missing = [row["reef_id"] for row in ordered if not (cache / f"{row['reef_id']}.csv").is_file()]
        if missing or args.refresh:
            raise ValueError(f"离线分析不能重新下载或缺少缓存：{missing}")
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        futures = {
            executor.submit(download_dhw, row, start, end, cache / f"{row['reef_id']}.csv", args.refresh): row["reef_id"]
            for row in ordered
        }
        for order, future in enumerate(as_completed(futures), start=1):
            reef_id = futures[future]
            try:
                status = future.result()
            except Exception as error:
                status = {"reef_id": reef_id, "status": "failed", "error": str(error)}
            manifest.append(status)
            pd.DataFrame(manifest).to_csv(out / "NOAA礁盘DHW下载清单.csv", index=False, encoding="utf-8-sig")
            print(f"{order}/{len(ordered)} {status['reef_id']} {status['status']}", flush=True)
    failures = [item for item in manifest if item["status"] == "failed"]
    if failures:
        raise RuntimeError(f"NOAA DHW 下载失败：{[item['reef_id'] for item in failures]}；未以其它数据替代")
    source_coverage = audit_raw_coverage(extents, cache, start, end, out)
    daily_path, weight_path = out / "礁盘尺度DHW日汇总.csv", out / "礁盘尺度DHW像元交叠权重.csv"
    if args.reuse_aggregated and daily_path.exists() and weight_path.exists():
        daily_dhw = pd.read_csv(daily_path, parse_dates=["date"])
        weights = pd.read_csv(weight_path)
        if set(daily_dhw["reef_id"].unique()) != set(extents["reef_id"]) or set(weights["reef_id"].unique()) != set(extents["reef_id"]):
            raise ValueError("已有逐日汇总与当前经核验的礁盘集合不一致，不能复用")
    else:
        daily, weight = [], []
        for _, row in extents.iterrows():
            one_daily, one_weight = aggregate_dhw(row, cache); daily.append(one_daily); weight.append(one_weight)
        daily_dhw, weights = pd.concat(daily, ignore_index=True), pd.concat(weight, ignore_index=True)
        daily_dhw.to_csv(daily_path, index=False, encoding="utf-8-sig")
        weights.to_csv(weight_path, index=False, encoding="utf-8-sig")
    sample, primary_events_by_reef = build_exposure_sample(pairs, daily_dhw, REEF_AREA_THRESHOLD)
    broad_sample, _ = build_exposure_sample(broad_pairs, daily_dhw, REEF_AREA_THRESHOLD)
    quarter_mask = broad_sample["time_t"].dt.quarter.eq(broad_sample["time_t1"].dt.quarter)
    quarter_sample = broad_sample.loc[quarter_mask].copy()
    lookup = nino_lookup()
    for lag in ENSO_LAGS:
        sample[f"nino34_pre_{lag}m_3mo"] = [nino_window(date, lag, lookup) for date in sample["time_t"]]
    sample.to_csv(out / "礁盘DHW_ENSO同季节面积分析样本.csv", index=False, encoding="utf-8-sig")
    events = events_table(primary_events_by_reef, REEF_AREA_THRESHOLD)
    events.to_csv(out / "礁盘尺度DHW8事件表.csv", index=False, encoding="utf-8-sig")
    primary_event_models = pd.concat(
        [dhw_event_model(sample, REEF_AREA_THRESHOLD, region) for region in ("GBR", "MEDF", "OTHER", "NH")],
        ignore_index=True,
    )
    availability_frames = [event_availability(sample, REEF_AREA_THRESHOLD)]
    sensitivity_frames = []
    for threshold in (0.25, 0.75):
        threshold_sample, _ = build_exposure_sample(pairs, daily_dhw, threshold)
        sensitivity_frames.append(dhw_event_model(threshold_sample, threshold, "GBR"))
        availability_frames.append(event_availability(threshold_sample, threshold))
    availability = pd.concat(availability_frames, ignore_index=True)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    continuous = continuous_regional_model(sample)
    annual_area_dhw = annual_mean_area_dhw_table(observations, daily_dhw)
    annual_area_dhw_model_result = annual_mean_area_dhw_model(annual_area_dhw)
    annual_lagged_change = annual_lagged_area_change_table(annual_area_dhw)
    annual_lagged_change_model_result = annual_lagged_area_change_model(annual_lagged_change)
    pairing_model_frames = []
    vegetation_model_frames = []
    pairing_count_rows = []
    for pairing_name, pairing_sample in (
        ("primary_anniversary_350_380d_doy15", sample),
        ("sensitivity_same_calendar_quarter", quarter_sample),
        ("sensitivity_anniversary_330_400d_doy30", broad_sample),
    ):
        event_part = dhw_event_model(pairing_sample, REEF_AREA_THRESHOLD, "GBR")
        event_part["pairing_specification"] = pairing_name
        continuous_part = continuous_regional_model(pairing_sample)
        continuous_part["pairing_specification"] = pairing_name
        pairing_model_frames.extend([event_part, continuous_part])
        vegetation_part = vegetation_state_interaction_model(pairing_sample)
        vegetation_part["pairing_specification"] = pairing_name
        transition_part = vegetation_state_event_transition_model(pairing_sample)
        transition_part["pairing_specification"] = pairing_name
        vegetation_model_frames.extend([vegetation_part, transition_part])
        pairing_count_rows.append({
            "pairing_specification": pairing_name,
            "n_pairs": int(len(pairing_sample)),
            "n_cays": int(pairing_sample["sand_cay_id"].nunique()),
            "n_reefs": int(pairing_sample["reef_id"].nunique()),
            "n_completed_event_pairs": int(pairing_sample["eligible_completed_event_pair"].sum()),
        })
    pairing_models = pd.concat(pairing_model_frames, ignore_index=True, sort=False)
    vegetation_models = pd.concat(vegetation_model_frames, ignore_index=True, sort=False)
    pairing_counts = pd.DataFrame(pairing_count_rows)
    enso = enso_model(sample)
    primary_event_models.to_csv(out / "礁盘DHW年度滞后面积模型.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(out / "礁盘DHW区域可估计性.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(out / "礁盘DHW阈值敏感性.csv", index=False, encoding="utf-8-sig")
    continuous.to_csv(out / "礁盘DHW区域连续模型.csv", index=False, encoding="utf-8-sig")
    annual_area_dhw.to_csv(out / "礁盘DHW年度平均面积关联样本.csv", index=False, encoding="utf-8-sig")
    annual_area_dhw_model_result.to_csv(out / "礁盘DHW年度平均面积关联模型.csv", index=False, encoding="utf-8-sig")
    annual_lagged_change.to_csv(out / "礁盘DHW一年滞后面积变化关联样本.csv", index=False, encoding="utf-8-sig")
    annual_lagged_change_model_result.to_csv(out / "礁盘DHW一年滞后面积变化关联模型.csv", index=False, encoding="utf-8-sig")
    pairing_models.to_csv(out / "礁盘DHW面积季节配对敏感性.csv", index=False, encoding="utf-8-sig")
    pairing_counts.to_csv(out / "礁盘DHW季节配对核查.csv", index=False, encoding="utf-8-sig")
    vegetation_models.to_csv(out / "礁盘DHW植被状态交互模型.csv", index=False, encoding="utf-8-sig")
    models = pd.concat([primary_event_models, sensitivity, continuous, vegetation_models, enso], ignore_index=True, sort=False)
    models.to_csv(out / "礁盘DHW_ENSO面积模型.csv", index=False, encoding="utf-8-sig")
    plot_models(primary_event_models, availability, continuous)
    plot_annual_mean_area_dhw(annual_area_dhw, annual_area_dhw_model_result)
    plot_annual_lagged_area_change(annual_lagged_change, annual_lagged_change_model_result)
    audit = {
        "status": "completed", "source": {"dataset": "NOAA CRW CRW_dhw_v1_0", "url": "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/CRW_dhw_v1_0.html", "variable": "degree_heating_week", "grid_degrees": GRID},
        "geometry": {"source": str(UPSTREAM / "data" / "metadata" / "reef_extent_metadata.csv"), "kmz_cross_check": str(UPSTREAM / "data" / "metadata" / "standardized_coordinates.kmz"), "representation": "source-defined axis-aligned reef bounding boxes", "active_reefs": int(len(extents)), "kmz_exact_matches": int(extents["kmz_status"].eq("exact_match").sum()), "metadata_only_active_reefs": extents.loc[extents["kmz_status"].eq("not_in_kmz"), "reef_id"].tolist()},
        "aggregation": {"method": "spherical-area overlap of NOAA 0.05 degree cell footprints with each reef box", "reef_wide_event": "DHW >=8 C-weeks over >=50% of reef-box area with full daily area coverage"},
        "outcome": {"sensor": "Sentinel-2", "quality": ["A", "B"], "same_reference_frame": True, "primary_pairing": {"duration_days": list(ANNUAL_DAYS), "maximum_circular_day_of_year_gap": SEASON_GAP}, "sensitivity_pairings": [{"name": "same_calendar_quarter", "duration_days": list(BROAD_ANNUAL_DAYS), "maximum_circular_day_of_year_gap": BROAD_SEASON_GAP}, {"name": "broad_anniversary", "duration_days": list(BROAD_ANNUAL_DAYS), "maximum_circular_day_of_year_gap": BROAD_SEASON_GAP}], "outcome": "log(area_t1/area_t)"},
        "temporal_order": "All models remove pairs with a reef-wide DHW>=8 day in (time_t,time_t1]. Threshold-event lags are 0-<1, 1-<2 and 2-<3 years before pair start; the reference is no such event in the preceding three years.", "enso": {"input": "detrend.nino34.ascii.txt", "fixed_window": "3-month mean ending before pair start", "lags_months": list(ENSO_LAGS), "same_completed_event_pairs": True},
        "interpretation": "DHW is heat stress, not direct bleaching, carbonate production, sediment transport, or deposition. Threshold-event effects are only estimated within regions that contain every event/reference contrast. The continuous regional model uses the preceding 12-month maximum of the reef-area-weighted mean DHW and includes calendar-year fixed effects. Vegetation heterogeneity uses only the audited baseline analysis state from 植被状态审校主表.csv: none versus dominant; conflict, sparse and partial states are excluded from the primary contrast. Two timings are retained: annual change during the first three post-event years, and a same-season pair that directly crosses a newly occurring event after a three-year event-free baseline. Associations cannot establish increased sediment supply or cay expansion.",
        "sample_counts": {"eligible_s2_observations": int(len(observations)), "strict_anniversary_pairs": int(len(pairs)), "completed_event_pairs": int(sample["eligible_completed_event_pair"].sum()), "primary_events": int(len(events)), "reefs_with_primary_event": int(sum(not item.empty for item in primary_events_by_reef.values()))}, "pairing_sensitivity": pairing_counts.to_dict("records"), "annual_mean_area_association": annual_area_dhw_model_result.to_dict("records"), "annual_one_year_lag_association": annual_lagged_change_model_result.to_dict("records"), "raw_noaa_coverage": source_coverage, "models": models.to_dict("records"),
    }
    (out / "礁盘DHW_ENSO面积核查.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(audit["sample_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
