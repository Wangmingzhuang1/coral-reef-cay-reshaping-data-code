"""交叉核对 GE 与 Sentinel-2 的标注记录、原图、掩膜、特征和沙洲登记。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_image(path: Path, flags: int) -> np.ndarray | None:
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buffer, flags) if buffer.size else None


def load_records(record_dir: Path, sensor: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record_path in sorted(record_dir.glob("*_sam_record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        image_path = Path(str(record.get("image_path", "")))
        mask_path = Path(str(record.get("mask_path", "")))
        has_mask = mask_path.is_file()
        has_image = image_path.is_file()
        shape_status = "not_checked"
        if has_image and has_mask:
            image = read_image(image_path, cv2.IMREAD_COLOR)
            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                shape_status = "unreadable"
            elif image.shape[:2] != mask.shape[:2]:
                shape_status = "shape_mismatch"
            elif not (mask > 0).any():
                shape_status = "empty_mask"
            else:
                shape_status = "ok"
        rows.append(
            {
                "sensor": sensor,
                "image_id": str(record.get("image_id", "")),
                "reef_id": str(record.get("reef_id", "")),
                "sand_cay_id": str(record.get("sand_cay_id", "")),
                "date": str(record.get("date", "")),
                "annotation_status": str(record.get("annotation_status", "")),
                "annotation_geometry": str(record.get("annotation_geometry", "")),
                "object_status": str(record.get("object_status", "")),
                "quality_grade": str(record.get("quality_grade", "")),
                "has_image": has_image,
                "has_mask": has_mask,
                "shape_status": shape_status,
            }
        )
    return rows


def audit_sensor(
    dataset_root: Path,
    sensor: str,
    records: pd.DataFrame,
    feature_path: Path,
    metadata_path: Path,
    registry_ids: set[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    features = pd.read_csv(feature_path, dtype=str).fillna("")
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    qsat_features = features.iloc[0:0].copy()
    if sensor == "sentinel2":
        qsat_features = features[features["image_id"].str.startswith("qsat__", na=False)].copy()
        features = features[~features["image_id"].str.startswith("qsat__", na=False)].copy()
    record_ids = set(records["image_id"])
    mask_record_ids = set(records.loc[records["has_mask"], "image_id"])
    feature_ids = set(features["image_id"])
    metadata_ids = set(metadata["image_id"])
    feature_cays = set(features["sand_cay_id"])
    record_cays = set(records["sand_cay_id"])
    report = {
        "sensor": sensor,
        "record_rows": int(len(records)),
        "record_reefs": int(records["reef_id"].nunique()),
        "record_sand_cays": int(records["sand_cay_id"].nunique()),
        "record_status_counts": dict(Counter(records["annotation_status"])),
        "record_geometry_counts": dict(Counter(records["annotation_geometry"])),
        "record_object_status_counts": dict(Counter(records["object_status"])),
        "records_with_existing_image": int(records["has_image"].sum()),
        "records_with_existing_mask": int(records["has_mask"].sum()),
        "records_without_existing_mask": int((~records["has_mask"]).sum()),
        "mask_records_image_mask_shape_ok": int((records["shape_status"] == "ok").sum()),
        "mask_records_image_mask_shape_issues": int(
            (records["has_mask"] & records["shape_status"].ne("ok")).sum()
        ),
        "feature_rows": int(len(features)),
        "qsat_2017_feature_rows_excluded_from_sentinel2": int(len(qsat_features)),
        "qsat_2017_feature_ids": sorted(qsat_features["image_id"].tolist()),
        "feature_reefs": int(features["reef_id"].nunique()),
        "feature_sand_cays": int(features["sand_cay_id"].nunique()),
        "feature_ids_missing_record": sorted(feature_ids - record_ids),
        "feature_ids_missing_existing_mask_record": sorted(feature_ids - mask_record_ids),
        "existing_mask_record_ids_missing_feature": sorted(mask_record_ids - feature_ids),
        "feature_ids_missing_metadata": sorted(feature_ids - metadata_ids),
        "feature_sand_cays_missing_registry": sorted(feature_cays - registry_ids),
        "record_sand_cays_missing_registry": sorted(record_cays - registry_ids),
        "feature_duplicate_image_ids": int(features["image_id"].duplicated().sum()),
        "feature_duplicate_cay_date": int(features.duplicated(["sand_cay_id", "date"]).sum()),
        "metadata_rows": int(len(metadata)),
        "metadata_feature_reference_frame_missing": int(
            metadata.loc[metadata["image_id"].isin(feature_ids), "reference_frame_id"].eq("").sum()
        ),
    }
    detail = records.copy()
    detail["in_feature_table"] = detail["image_id"].isin(feature_ids)
    detail["in_metadata_table"] = detail["image_id"].isin(metadata_ids)
    detail["in_registry"] = detail["sand_cay_id"].isin(registry_ids)
    return report, detail


def main() -> None:
    research_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=research_root / "data" / "source_dataset"
    )
    parser.add_argument("--output-dir", type=Path, default=research_root / "outputs")
    args = parser.parse_args()

    registry = pd.read_csv(args.dataset_root / "data" / "metadata" / "cay_registry.csv", dtype=str)
    registry_ids = set(registry["sand_cay_id"].dropna())
    specifications = [
        {
            "sensor": "google_earth",
            "record_dir": args.dataset_root / "outputs" / "sam_records",
            "feature_path": args.dataset_root / "outputs" / "derived_features" / "mask_features.csv",
            "metadata_path": args.dataset_root / "data" / "metadata" / "image_metadata_template.csv",
        },
        {
            "sensor": "sentinel2",
            "record_dir": args.dataset_root / "outputs" / "sentinel2_annotations" / "sam_records",
            "feature_path": args.dataset_root / "outputs" / "sentinel2_annotations" / "derived_features" / "mask_features.csv",
            "metadata_path": args.dataset_root / "data" / "metadata" / "sentinel2_annotation_metadata.csv",
        },
    ]
    reports: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for spec in specifications:
        records = pd.DataFrame(load_records(spec["record_dir"], spec["sensor"]))
        report, detail = audit_sensor(
            args.dataset_root,
            spec["sensor"],
            records,
            spec["feature_path"],
            spec["metadata_path"],
            registry_ids,
        )
        reports.append(report)
        details.append(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "GE_Sentinel标注核查.json"
    detail_path = args.output_dir / "GE_Sentinel标注核查明细.csv"
    summary = {
        "registry_sand_cays": int(len(registry_ids)),
        "reports": reports,
        "semantic_warning": "本核查验证数据链路完整性，不等价于掩膜边界的地貌语义完全正确；语义正确性仍需按优先级抽样复核原始影像。",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat(details, ignore_index=True).to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()
