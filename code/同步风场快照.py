"""从时空模型训练项目同步已下载的 ERA5 风场快照，并增量重建本项目日尺度场。

安全约束：
- 对训练项目目录只读，绝不触碰正在运行的下载进程；
- 活跃下载 tile 中最近 3 分钟内写入的文件跳过（避免复制中间态）；
- 只重建 raw 文件数发生变化的 tile 的 daily 缓存（派生数据，可再生）；
- reef_daily_features.csv 中仅替换 era5 行，保留已验证的 glorys/waverys 行；
- 替换前先备份原 CSV 与 point_selection_audit.csv。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def copy_new_files(train_root: Path, proj_root: Path, active_tiles: set[str], quiet_minutes: int) -> dict:
    cutoff = datetime.now() - timedelta(minutes=quiet_minutes)
    copied = 0
    skipped_recent = 0
    per_tile: dict[str, int] = {}
    for tile_dir in sorted(train_root.iterdir()):
        if not tile_dir.is_dir():
            continue
        dst_dir = proj_root / tile_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in tile_dir.glob("*.nc"):
            dst = dst_dir / f.name
            if dst.exists():
                continue
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if tile_dir.name in active_tiles and mtime > cutoff:
                skipped_recent += 1
                continue
            shutil.copy2(f, dst)
            copied += 1
            per_tile[tile_dir.name] = per_tile.get(tile_dir.name, 0) + 1
    return {"copied": copied, "skipped_recent": skipped_recent, "per_tile": per_tile}


def rebuild_point_selection_audit(
    daily_manifest: pd.DataFrame,
    aoi: pd.DataFrame,
    output_path: Path,
    max_radius_deg: float = 0.6,
) -> pd.DataFrame:
    """重建三源空间取点审计，不在内存中重复物化数百万行逐日数据。"""
    candidates: list[dict[str, object]] = []
    for item in daily_manifest.to_dict("records"):
        source, tile = item["source"], item["spatial_tile"]
        with xr.open_dataset(item["daily_path"], engine="h5netcdf") as ds:
            spatial = [
                name
                for name in ("latitude", "longitude")
                if name in ds.coords and ds.sizes.get(name, 0) > 1
            ]
            if spatial:
                ds = ds.sortby(spatial)
            west, east = float(ds.longitude.min()), float(ds.longitude.max())
            south, north = float(ds.latitude.min()), float(ds.latitude.max())
            reefs = aoi.loc[
                aoi.center_lon.between(west, east)
                & aoi.center_lat.between(south, north)
            ]
            for reef in reefs.itertuples(index=False):
                lon, lat = float(reef.center_lon), float(reef.center_lat)
                sub = ds.sel(
                    longitude=slice(lon - max_radius_deg, lon + max_radius_deg),
                    latitude=slice(lat - max_radius_deg, lat + max_radius_deg),
                )
                variables = list(sub.data_vars)
                if (
                    not variables
                    or sub.sizes.get("longitude", 0) == 0
                    or sub.sizes.get("latitude", 0) == 0
                ):
                    continue
                missing_fraction = (
                    sum(sub[name].isnull().mean(dim="time") for name in variables)
                    / len(variables)
                )
                distance = np.hypot(
                    sub.longitude.values[None, :] - lon,
                    sub.latitude.values[:, None] - lat,
                )
                score = missing_fraction.values * 10.0 + distance
                iy, ix = np.unravel_index(int(np.argmin(score)), score.shape)
                candidates.append(
                    {
                        "reef_id": reef.reef_id,
                        "source": source,
                        "spatial_tile": tile,
                        "center_lon": lon,
                        "center_lat": lat,
                        "sampled_lon": float(sub.longitude.values[ix]),
                        "sampled_lat": float(sub.latitude.values[iy]),
                        "valid_fraction": 1.0
                        - float(missing_fraction.values[iy, ix]),
                        "offset_degrees": float(distance[iy, ix]),
                    }
                )
    audit = pd.DataFrame(candidates)
    if audit.empty:
        raise ValueError("三源空间取点审计为空")
    audit = audit.sort_values(
        ["reef_id", "source", "valid_fraction", "offset_degrees"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["reef_id", "source"], keep="first")
    audit = audit.sort_values(["reef_id", "source"]).reset_index(drop=True)
    audit.to_csv(output_path, index=False, encoding="utf-8-sig")
    return audit


def main() -> None:
    proj = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-raw",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-src",
        type=Path,
        required=True,
    )
    parser.add_argument("--quiet-minutes", type=int, default=3)
    parser.add_argument(
        "--active-tiles",
        nargs="*",
        default=["165_10", "145_-20", "150_-20", "55_-20", "70_0"],
        help="正在下载的 tile；其最近写入的文件本轮跳过",
    )
    parser.add_argument("--skip-copy", action="store_true")
    args = parser.parse_args()

    env_dir = proj / "data" / "environment"
    raw_era5 = env_dir / "raw" / "era5"
    harm = env_dir / "harmonized"
    daily_dir = harm / "daily" / "era5"
    reef_csv = harm / "daily" / "reef_daily_features.csv"
    manifest_csv = harm / "daily_manifest.csv"
    audit_dir = env_dir / "logs"
    audit_dir.mkdir(parents=True, exist_ok=True)

    copy_summary = {"copied": 0, "skipped_recent": 0, "per_tile": {}}
    if not args.skip_copy:
        copy_summary = copy_new_files(
            args.train_raw, raw_era5, set(args.active_tiles), args.quiet_minutes
        )
        print(f"copied={copy_summary['copied']} skipped_recent={copy_summary['skipped_recent']}")

    # 找出 raw 文件数与 manifest 不一致的 tile，删除其过期 daily 缓存
    manifest = pd.read_csv(manifest_csv)
    stale_tiles = []
    for tile_dir in sorted(raw_era5.iterdir()):
        if not tile_dir.is_dir():
            continue
        n_now = len(list(tile_dir.glob("*.nc")))
        row = manifest[(manifest.source == "era5") & (manifest.spatial_tile == tile_dir.name)]
        n_old = int(row.raw_file_count.iloc[0]) if len(row) else -1
        if n_now != n_old:
            stale_tiles.append(tile_dir.name)
            daily = daily_dir / f"{tile_dir.name}.nc"
            if daily.exists():
                daily.unlink()
    print(f"stale_tiles={stale_tiles}")

    sys.path.insert(0, str(args.train_src))
    from sand_cay_forecast.environment_harmonization import (
        build_daily_fields,
        build_reef_daily_fields,
    )

    new_manifest = build_daily_fields(env_dir / "raw", harm, ("era5",))
    merged = pd.concat(
        [manifest.loc[manifest.source != "era5"], new_manifest], ignore_index=True, sort=False
    )
    merged.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    audit_path = harm / "daily" / "point_selection_audit.csv"
    if audit_path.exists():
        shutil.copy2(
            audit_path,
            audit_dir / f"point_selection_audit.backup_{datetime.now():%Y%m%d_%H%M%S}.csv",
        )
    era5_manifest = merged.loc[merged.source == "era5"]
    aoi = pd.read_csv(env_dir / "spatial_index" / "reef_environment_aoi.csv")
    era5_daily = build_reef_daily_fields(era5_manifest, aoi, harm)
    point_audit = rebuild_point_selection_audit(merged, aoi, audit_path)

    if reef_csv.exists():
        backup = audit_dir / f"reef_daily_features.backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
        shutil.copy2(reef_csv, backup)
        existing = pd.read_csv(reef_csv, low_memory=False)
        retained = existing.loc[existing.source != "era5"]
        result = pd.concat([retained, era5_daily], ignore_index=True, sort=False)
    else:
        result = era5_daily
        backup = None
    result = result.sort_values(["reef_id", "source", "time"]).reset_index(drop=True)
    result.to_csv(reef_csv, index=False, encoding="utf-8-sig")

    era5_rows = result.loc[result.source == "era5"]
    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "copy": copy_summary,
        "stale_tiles_rebuilt": stale_tiles,
        "era5_raw_files": int(sum(1 for _ in raw_era5.rglob("*.nc"))),
        "era5_daily_rows": int(len(era5_rows)),
        "era5_reefs": int(era5_rows.reef_id.nunique()),
        "era5_time_range": [str(era5_rows.time.min()), str(era5_rows.time.max())],
        "duplicate_reef_time": int(era5_rows.duplicated(["reef_id", "time"]).sum()),
        "point_audit_rows": int(len(point_audit)),
        "point_audit_by_source": {
            str(key): int(value)
            for key, value in point_audit["source"].value_counts().items()
        },
        "backup": str(backup) if backup else None,
        "active_tiles_skipped_recent": sorted(set(args.active_tiles)),
    }
    out = audit_dir / f"wind_sync_audit_{datetime.now():%Y%m%d}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
