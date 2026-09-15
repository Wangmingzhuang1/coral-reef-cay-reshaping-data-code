"""Copernicus Marine in-situ 潮位站观测同步（GESLA/GLOSS 来源的替代可达通道）。

背景：GESLA-4.1 官方通道（UHSLC ERDDAP 与 GESLA 站点）在本网络不可达
（连接超时 / 403），改用 Copernicus Marine in-situ 离散观测中的潮位站
sea_level 变量；该集合整合 GLOSS/GESLA 贡献的验潮站小时级水位。

两个阶段：
- catalog：describe 目录，筛选含 sea_level 变量的 in-situ 数据集，写入
  outputs/潮位站数据集目录.json；
- sync：站点清点（近期+历史窗口）→ 礁盘匹配 → 按站下载小时水位 →
  影像时刻水位插值 → manifest 与审计日志。

凭据由 copernicusmarine 本地凭据文件提供，脚本不读取、不打印凭据。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IMAGE_CSV = ROOT / "outputs" / "潮位影像时刻与坐标.csv"
RAW_DIR = ROOT / "data" / "environment" / "raw" / "tide_gauge"
CATALOG_JSON = ROOT / "outputs" / "潮位站数据集目录.json"
MATCH_CSV = ROOT / "outputs" / "潮位站站点匹配.csv"
LEVEL_CSV = ROOT / "outputs" / "潮位站影像时刻水位.csv"
MANIFEST = ROOT / "data" / "environment" / "manifests" / "tide_gauge_manifest.json"
AUDIT = ROOT / "data" / "environment" / "logs" / "tide_gauge_sync_audit.json"
LOG_PATH = ROOT / "data" / "environment" / "logs" / "tide_gauge_sync.log"
MAX_DISTANCE_KM = 250.0


LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=float); lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float); lon2 = np.asarray(lon2, dtype=float)
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def run_catalog() -> None:
    import copernicusmarine as cm

    description = cm.describe()
    products = getattr(description, "products", None)
    if products is None:
        products = description["products"]
    candidates = []
    for product in products:
        pid = str(getattr(product, "product_id", ""))
        if "ins" not in pid.lower():
            continue
        for dataset in getattr(product, "datasets", None) or []:
            did = str(getattr(dataset, "dataset_id", ""))
            names = []
            platform_types = set()
            for version in getattr(dataset, "versions", None) or []:
                for part in getattr(version, "parts", None) or []:
                    for service in getattr(part, "services", None) or []:
                        for variable in getattr(service, "variables", None) or []:
                            names.append(str(getattr(variable, "short_name", variable)).upper())
                        meta = getattr(service, "platforms_metadata", None)
                        if isinstance(meta, dict):
                            for kind in meta.get("types", []) or []:
                                platform_types.add(str(kind))
            variables = sorted(set(names))
            sea = [v for v in variables if "SEA_LEVEL" in v]
            if not sea:
                continue
            candidates.append(
                {
                    "product_id": pid,
                    "dataset_id": did,
                    "sea_level_variable": sea[0],
                    "variables": variables,
                    "platform_types": sorted(platform_types),
                    "title": str(getattr(dataset, "dataset_name", "")),
                }
            )
    preferred = [c for c in candidates if "_my_" in c["dataset_id"] or "_my-" in c["dataset_id"]]
    pool = preferred or candidates
    pool.sort(key=lambda c: c["dataset_id"])
    CATALOG_JSON.write_text(json.dumps({"candidates": candidates, "selected": pool[:3]}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"catalog candidates={len(candidates)} preferred={len(preferred)} selected={[c['dataset_id'] for c in pool[:3]]}")


def reef_centers():
    images = pd.read_csv(IMAGE_CSV)
    images = images[images.coordinate_status.eq("ok")].copy()
    images["acquisition_datetime_utc"] = pd.to_datetime(images.acquisition_datetime_utc, utc=True, errors="coerce")
    images = images.dropna(subset=["acquisition_datetime_utc", "sand_cay_center_lat", "sand_cay_center_lon"])
    grouped = images.groupby("reef_id").agg(
        lat=("sand_cay_center_lat", "mean"),
        lon=("sand_cay_center_lon", "mean"),
        n_images=("image_id", "size"),
        first_image=("acquisition_datetime_utc", "min"),
        last_image=("acquisition_datetime_utc", "max"),
    ).reset_index()
    return grouped, images


def run_sync(dataset_id: str, sea_var: str) -> None:
    import copernicusmarine as cm

    reefs, images = reef_centers()
    window_start, window_end = datetime(2025, 6, 1), datetime(2025, 6, 8)
    if MATCH_CSV.is_file() and MATCH_CSV.stat().st_size > 0:
        match = pd.read_csv(MATCH_CSV)
        log(f"reuse existing match table rows={len(match)}")
        match = match.sort_values(["reef_id", "distance_km"])
        _skip_inventory = True
    else:
        _skip_inventory = False

    def reef_matches(reef):
        out = []
        for radius in (2.0, 4.0):
            df = None
            for attempt in range(2):
                try:
                    df = cm.read_dataframe(
                        dataset_id=dataset_id,
                        variables=[sea_var],
                        start_datetime=window_start,
                        end_datetime=window_end,
                        minimum_latitude=reef.lat - radius,
                        maximum_latitude=reef.lat + radius,
                        minimum_longitude=reef.lon - radius,
                        maximum_longitude=reef.lon + radius,
                        disable_progress_bar=True,
                    )
                    break
                except Exception as exc:
                    log(f"reef {reef.reef_id} box {radius} attempt {attempt} failed: {type(exc).__name__}")
            if df is None or len(df) == 0:
                continue
            flat = df.reset_index()
            platform = flat["platform_id"] if "platform_id" in flat.columns else pd.Series(["unknown"] * len(flat))
            grp = pd.DataFrame({"platform_id": platform.values, "lat": flat["latitude"].values, "lon": flat["longitude"].values}).groupby("platform_id").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
            dist = haversine_km(reef.lat, reef.lon, grp.lat.values, grp.lon.values)
            for i2 in np.argsort(dist):
                if dist[i2] <= MAX_DISTANCE_KM:
                    out.append({"reef_id": reef.reef_id, "platform_id": grp.platform_id.iloc[i2], "station_lat": float(grp.lat.iloc[i2]), "station_lon": float(grp.lon.iloc[i2]), "distance_km": float(dist[i2]), "reef_n_images": int(reef.n_images), "reef_first_image": str(reef.first_image), "reef_last_image": str(reef.last_image)})
            if out:
                break
        return out

    if not _skip_inventory:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as pool:
            per_reef = list(pool.map(reef_matches, list(reefs.itertuples())))
        seen = set()
        rows = []
        for chunk in per_reef:
            for row in chunk:
                key = (row["reef_id"], row["platform_id"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    if not _skip_inventory:
        match = pd.DataFrame(rows).sort_values(["reef_id", "distance_km"])
        match.to_csv(MATCH_CSV, index=False, encoding="utf-8-sig")
    log(f"matched pairs={len(match)} reefs_with_station={match.reef_id.nunique() if len(match) else 0} stations={match.platform_id.nunique() if len(match) else 0}")
    if match.empty:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps({"status": "no_stations_within_threshold", "max_distance_km": MAX_DISTANCE_KM}, indent=2), encoding="utf-8")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start_all = pd.to_datetime(match.reef_first_image.min(), utc=True).to_pydatetime() - timedelta(days=1)
    end_all = pd.to_datetime(match.reef_last_image.max(), utc=True).to_pydatetime() + timedelta(days=1)
    dataset_end = datetime(2025, 6, 30, 23, 0, 0, tzinfo=timezone.utc)
    if end_all > dataset_end:
        end_all = dataset_end
    per_station = []
    for platform in match.platform_id.unique():
        sub = match[match.platform_id.eq(platform)]
        lat, lon = float(sub.station_lat.iloc[0]), float(sub.station_lon.iloc[0])
        out = RAW_DIR / f"{platform}.csv"
        if out.is_file() and out.stat().st_size > 0:
            per_station.append({"platform_id": platform, "status": "exists", "rows": sum(1 for _ in open(out, encoding="utf-8-sig")) - 1})
            continue
        try:
            df = None
            for attempt in range(2):
                try:
                    df = cm.read_dataframe(
                        dataset_id=dataset_id,
                        variables=[sea_var],
                        start_datetime=start_all,
                        end_datetime=end_all,
                        minimum_latitude=lat - 0.1,
                        maximum_latitude=lat + 0.1,
                        minimum_longitude=lon - 0.1,
                        maximum_longitude=lon + 0.1,
                        disable_progress_bar=True,
                    )
                    break
                except Exception as exc:
                    log(f"station {platform} attempt {attempt} failed: {type(exc).__name__}")
            if df is None:
                per_station.append({"platform_id": platform, "status": "failed"})
                continue
        except Exception as exc:
            log(f"station {platform} download failed: {exc}")
            per_station.append({"platform_id": platform, "status": f"failed:{type(exc).__name__}"})
            continue
        flat = df.reset_index()
        flat = flat[["time", "latitude", "longitude", sea_var]].copy()
        flat.columns = ["time", "lat", "lon", "sea_level_m"]
        flat = flat.dropna(subset=["sea_level_m"]).sort_values("time")
        flat.to_csv(out, index=False, encoding="utf-8-sig")
        per_station.append({"platform_id": platform, "status": "ok", "rows": int(len(flat)), "first": str(flat.time.iloc[0]) if len(flat) else None, "last": str(flat.time.iloc[-1]) if len(flat) else None})
        log(f"station {platform} rows={len(flat)}")
    level_rows = []
    stations = {}
    for p in match.platform_id.unique():
        path = RAW_DIR / f"{p}.csv"
        if path.is_file():
            stations[p] = pd.read_csv(path, parse_dates=["time"])
    for image in images.itertuples():
        sub = match[match.reef_id.eq(image.reef_id)]
        if sub.empty:
            continue
        best = sub.iloc[0]
        series = stations.get(best.platform_id)
        if series is None or series.empty:
            continue
        target = image.acquisition_datetime_utc
        pos = series.time.searchsorted(target)
        if pos <= 0 or pos >= len(series):
            continue
        t0, t1 = series.time.iloc[pos - 1], series.time.iloc[pos]
        v0, v1 = series.sea_level_m.iloc[pos - 1], series.sea_level_m.iloc[pos]
        gap_hours = (t1 - t0).total_seconds() / 3600.0
        if gap_hours > 3.0:
            continue
        weight = (target - t0).total_seconds() / (t1 - t0).total_seconds()
        level_rows.append(
            {
                "image_id": image.image_id,
                "sensor": image.sensor,
                "reef_id": image.reef_id,
                "sand_cay_id": image.sand_cay_id,
                "acquisition_datetime_utc": str(target),
                "station_platform_id": best.platform_id,
                "station_distance_km": round(float(best.distance_km), 2),
                "tide_gauge_level_m": round(float(v0 + weight * (v1 - v0)), 4),
                "interpolation_gap_hours": round(gap_hours, 2),
                "source": "copernicus_marine_insitu_tide_gauge",
            }
        )
    levels = pd.DataFrame(level_rows)
    if not levels.empty:
        levels.to_csv(LEVEL_CSV, index=False, encoding="utf-8-sig")
    audit = {
        "dataset_id": dataset_id,
        "sea_level_variable": sea_var,
        "max_distance_km": MAX_DISTANCE_KM,
        "matched_pairs": int(len(match)),
        "reefs_with_station": int(match.reef_id.nunique()),
        "stations": per_station,
        "image_levels": int(len(levels)),
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "note": "GESLA-4.1 official endpoints unreachable from this network; Copernicus Marine in-situ tide-gauge sea_level used as observation-based substitute.",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"sync done image_levels={len(levels)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["catalog", "sync"], default="catalog")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--sea-var", default=None)
    args = parser.parse_args()
    if args.mode == "catalog":
        run_catalog()
        return
    dataset_id, sea_var = args.dataset_id, args.sea_var
    if not dataset_id and CATALOG_JSON.is_file():
        catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
        if catalog.get("selected"):
            dataset_id = catalog["selected"][0]["dataset_id"]
            sea_var = catalog["selected"][0]["sea_level_variable"]
    if not dataset_id:
        raise SystemExit("run --mode catalog first")
    run_sync(dataset_id, sea_var)


if __name__ == "__main__":
    main()
