"""从已分割沙洲掩膜中提取颜色组成，并与人工植被状态进行一致性检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VEGETATION_STATE_ORDER = {
    "none": 0,
    "sparse": 1,
    "partial": 2,
    "dominant": 3,
}

# 像元级规则使用归一化 RGB 与 HSV，输出时完整记录阈值。
GREEN_HUE_MIN_DEG = 45.0
GREEN_HUE_MAX_DEG = 165.0
GREEN_MIN_SATURATION = 0.12
GREEN_MIN_VALUE = 0.08
GREEN_MIN_EXCESS = 0.025
GREEN_MIN_RATIO = 0.345

SUBSTRATE_MIN_VALUE = 0.55
SUBSTRATE_MAX_SATURATION = 0.28

# 暗色/橄榄褐色异常表面只是相对于灰白裸沙和绿色植被的光谱差异，不能解释为
# 生物结皮、藻类、湿沙、阴影或任何确定的表面物质。它保留为一个可检验的
# 过渡候选状态，必须结合人工审计、时间序列和形态移动性后再决定其意义。
OLIVE_ANOMALY_HUE_MIN_DEG = 15.0
OLIVE_ANOMALY_HUE_MAX_DEG = 120.0
OLIVE_ANOMALY_MIN_SATURATION = 0.04
OLIVE_ANOMALY_MAX_SATURATION = 0.35
OLIVE_ANOMALY_MIN_VALUE = 0.45
OLIVE_ANOMALY_MAX_VALUE = 0.92
DARK_ANOMALY_MAX_VALUE = 0.45

# 边缘-内部结构：侵蚀半径取等效半径的 15%。
MARGIN_RADIUS_FRACTION = 0.15
MARGIN_MIN_PIXELS = 10

# 影像级植被状态阈值，对应 vegetation_fraction。
SCENE_SPARSE_MIN = 0.03
SCENE_PARTIAL_MIN = 0.15
SCENE_DOMINANT_MIN = 0.45
WHITE_REFERENCE_QUANTILE = 0.90
WHITE_BALANCE_GAIN_MIN = 0.50
WHITE_BALANCE_GAIN_MAX = 2.00


def read_image(path: Path, flags: int) -> np.ndarray | None:
    """兼容 Windows 非 ASCII 路径。"""
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def classify_scene(vegetation_fraction: float) -> str:
    if vegetation_fraction >= SCENE_DOMINANT_MIN:
        return "dominant"
    if vegetation_fraction >= SCENE_PARTIAL_MIN:
        return "partial"
    if vegetation_fraction >= SCENE_SPARSE_MIN:
        return "sparse"
    return "none"


def resolve_record_path(raw_path: str, dataset_root: Path, fallback: Path) -> Path:
    path = Path(raw_path) if raw_path else fallback
    if path.is_absolute():
        parts = list(path.parts)
        try:
            source_index = next(
                index
                for index, part in enumerate(parts)
                if part.lower() == "sand_cay_temporal_dataset_builder"
            )
        except StopIteration:
            return path
        return dataset_root.joinpath(*parts[source_index + 1 :])
    return dataset_root / path


def calibrated_colour_pixels(image_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """全景比例、岸段对照和图件共用同一白平衡与绿色像元规则。"""
    from 构建沙洲观测与变化表 import normalize_cay_mask
    selected = normalize_cay_mask(mask)
    if image_bgr.shape[:2] != selected.shape or not selected.any():
        raise ValueError("颜色判读要求同尺寸影像与非空沙洲主掩膜。")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    raw_pixels = rgb[selected]

    # 以沙洲掩膜内最亮的 10% 像元作为灰白基质参考，校正不同影像的综合色偏。
    # 这不会把灰白基质进一步解释为珊瑚断枝或沙。
    raw_value = raw_pixels.max(axis=1)
    white_cutoff = float(np.quantile(raw_value, WHITE_REFERENCE_QUANTILE))
    white_candidates = raw_pixels[raw_value >= white_cutoff]
    white_reference = np.median(white_candidates, axis=0)
    neutral_level = float(np.mean(white_reference))
    gains = np.clip(
        neutral_level / np.maximum(white_reference, 1e-6),
        WHITE_BALANCE_GAIN_MIN,
        WHITE_BALANCE_GAIN_MAX,
    )
    pixels = np.clip(raw_pixels * gains, 0.0, 1.0)
    hsv_pixels = cv2.cvtColor(
        np.round(pixels.reshape(-1, 1, 3) * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    ).reshape(-1, 3)
    hues = hsv_pixels[:, 0].astype(np.float32) * 2.0

    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    value = pixels.max(axis=1)
    channel_min = pixels.min(axis=1)
    saturation = (value - channel_min) / np.maximum(value, 1e-6)
    channel_sum = red + green + blue
    green_ratio = green / np.maximum(channel_sum, 1e-6)
    green_excess = 2.0 * green - red - blue

    vegetation = (
        (hues >= GREEN_HUE_MIN_DEG)
        & (hues <= GREEN_HUE_MAX_DEG)
        & (saturation >= GREEN_MIN_SATURATION)
        & (value >= GREEN_MIN_VALUE)
        & (green_excess >= GREEN_MIN_EXCESS)
        & (green_ratio >= GREEN_MIN_RATIO)
    )
    return dict(selected=selected, pixels=pixels, hues=hues, value=value,
                saturation=saturation, green_excess=green_excess,
                white_reference=white_reference, gains=gains, vegetation=vegetation)


def vegetation_pixels(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    data = calibrated_colour_pixels(image_bgr, mask)
    result = np.zeros(mask.shape, dtype=bool)
    result[data["selected"]] = data["vegetation"]
    return result


def extract_record(record_path: Path, dataset_root: Path, sensor: str) -> dict[str, object]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    image_id = str(record.get("image_id", record_path.stem))
    fallback_mask_dir = (
        dataset_root / "outputs" / "master_masks"
        if sensor == "google_earth"
        else dataset_root / "outputs" / "sentinel2_annotations" / "master_masks"
    )
    image_path = resolve_record_path(str(record.get("image_path", "")), dataset_root, Path(""))
    mask_path = resolve_record_path(
        str(record.get("mask_path", "")),
        dataset_root,
        fallback_mask_dir / f"{image_id}_mask.png",
    )

    base = {
        "image_id": image_id,
        "reef_id": str(record.get("reef_id", "")),
        "sand_cay_id": str(record.get("sand_cay_id", "")),
        "date": str(record.get("date", "")),
        "sensor": sensor,
        "manual_vegetation_state": str(record.get("vegetation_state", "")),
        "development_stage": str(record.get("development_stage", "")),
        "surface_cover": str(record.get("surface_cover", "")),
        "quality_grade": str(record.get("quality_grade", "")),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "status": "ok",
        "note": "",
    }

    if not image_path.is_file():
        return base | {"status": "missing_image", "note": "原始影像不存在"}
    if not mask_path.is_file():
        return base | {"status": "missing_mask", "note": "掩膜不存在"}

    image_bgr = read_image(image_path, cv2.IMREAD_COLOR)
    mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        return base | {"status": "unreadable_image", "note": "原始影像无法读取"}
    if mask is None:
        return base | {"status": "unreadable_mask", "note": "掩膜无法读取"}
    if image_bgr.shape[:2] != mask.shape[:2]:
        return base | {
            "status": "shape_mismatch",
            "note": f"image={image_bgr.shape[:2]}, mask={mask.shape[:2]}",
        }

    from 构建沙洲观测与变化表 import normalize_cay_mask
    mask = normalize_cay_mask(mask)
    selected = mask
    pixel_count = int(selected.sum())
    if pixel_count == 0:
        return base | {"status": "empty_mask", "note": "掩膜内没有像元"}

    colour = calibrated_colour_pixels(image_bgr, mask)
    pixels, hues = colour["pixels"], colour["hues"]
    value, saturation, green_excess = colour["value"], colour["saturation"], colour["green_excess"]
    white_reference, gains = colour["white_reference"], colour["gains"]
    vegetation = colour["vegetation"]
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    light_substrate = (
        (~vegetation)
        & (value >= SUBSTRATE_MIN_VALUE)
        & (saturation <= SUBSTRATE_MAX_SATURATION)
    )
    dark_or_olive_anomaly = (
        (~vegetation)
        & (~light_substrate)
        & (
            (
                (hues >= OLIVE_ANOMALY_HUE_MIN_DEG)
                & (hues <= OLIVE_ANOMALY_HUE_MAX_DEG)
                & (saturation >= OLIVE_ANOMALY_MIN_SATURATION)
                & (saturation <= OLIVE_ANOMALY_MAX_SATURATION)
                & (value >= OLIVE_ANOMALY_MIN_VALUE)
                & (value <= OLIVE_ANOMALY_MAX_VALUE)
            )
            | (value < DARK_ANOMALY_MAX_VALUE)
        )
    )
    unclassified = ~(vegetation | light_substrate | dark_or_olive_anomaly)

    mask_binary = (mask > 0).astype(np.uint8)
    veg_2d = np.zeros(mask.shape, dtype=bool)
    veg_2d[selected] = vegetation
    substrate_2d = np.zeros(mask.shape, dtype=bool)
    substrate_2d[selected] = light_substrate
    anomaly_2d = np.zeros(mask.shape, dtype=bool)
    anomaly_2d[selected] = dark_or_olive_anomaly
    radius = max(2, int(round(MARGIN_RADIUS_FRACTION * np.sqrt(pixel_count / np.pi))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    interior = cv2.erode(mask_binary, kernel) > 0
    margin = (mask_binary > 0) & ~interior
    if int(margin.sum()) >= MARGIN_MIN_PIXELS and int(interior.sum()) >= MARGIN_MIN_PIXELS:
        interior_vegetation_fraction = float(veg_2d[interior].mean())
        margin_substrate_fraction = float(substrate_2d[margin].mean())
        margin_vegetation_fraction = float(veg_2d[margin].mean())
        interior_dark_olive_anomaly_fraction = float(anomaly_2d[interior].mean())
        margin_dark_olive_anomaly_fraction = float(anomaly_2d[margin].mean())
        core_margin_structure = bool(
            interior_vegetation_fraction >= 0.45 and margin_substrate_fraction >= 0.40
        )
    else:
        interior_vegetation_fraction = np.nan
        margin_substrate_fraction = np.nan
        margin_vegetation_fraction = np.nan
        interior_dark_olive_anomaly_fraction = np.nan
        margin_dark_olive_anomaly_fraction = np.nan
        core_margin_structure = False

    vegetation_fraction = float(vegetation.mean())
    substrate_fraction = float(light_substrate.mean())
    dark_olive_anomaly_fraction = float(dark_or_olive_anomaly.mean())
    unclassified_fraction = float(unclassified.mean())
    spectral_composition_sum = (
        vegetation_fraction
        + substrate_fraction
        + dark_olive_anomaly_fraction
        + unclassified_fraction
    )

    return base | {
        "pixel_count": pixel_count,
        "vegetation_fraction": vegetation_fraction,
        "light_substrate_fraction": substrate_fraction,
        "dark_olive_anomaly_fraction": dark_olive_anomaly_fraction,
        "unclassified_fraction": unclassified_fraction,
        "spectral_composition_sum": spectral_composition_sum,
        "interior_vegetation_fraction": interior_vegetation_fraction,
        "margin_substrate_fraction": margin_substrate_fraction,
        "margin_vegetation_fraction": margin_vegetation_fraction,
        "interior_dark_olive_anomaly_fraction": interior_dark_olive_anomaly_fraction,
        "margin_dark_olive_anomaly_fraction": margin_dark_olive_anomaly_fraction,
        "core_margin_structure": core_margin_structure,
        "automatic_vegetation_state": classify_scene(vegetation_fraction),
        "mean_red": float(red.mean()),
        "mean_green": float(green.mean()),
        "mean_blue": float(blue.mean()),
        "median_value": float(np.median(value)),
        "median_saturation": float(np.median(saturation)),
        "median_green_excess": float(np.median(green_excess)),
        "white_reference_red": float(white_reference[0]),
        "white_reference_green": float(white_reference[1]),
        "white_reference_blue": float(white_reference[2]),
        "white_balance_gain_red": float(gains[0]),
        "white_balance_gain_green": float(gains[1]),
        "white_balance_gain_blue": float(gains[2]),
        "green_hue_min_deg": GREEN_HUE_MIN_DEG,
        "green_hue_max_deg": GREEN_HUE_MAX_DEG,
        "green_min_saturation": GREEN_MIN_SATURATION,
        "green_min_value": GREEN_MIN_VALUE,
        "green_min_excess": GREEN_MIN_EXCESS,
        "green_min_ratio": GREEN_MIN_RATIO,
        "substrate_min_value": SUBSTRATE_MIN_VALUE,
        "substrate_max_saturation": SUBSTRATE_MAX_SATURATION,
        "olive_anomaly_hue_min_deg": OLIVE_ANOMALY_HUE_MIN_DEG,
        "olive_anomaly_hue_max_deg": OLIVE_ANOMALY_HUE_MAX_DEG,
        "olive_anomaly_min_saturation": OLIVE_ANOMALY_MIN_SATURATION,
        "olive_anomaly_max_saturation": OLIVE_ANOMALY_MAX_SATURATION,
        "olive_anomaly_min_value": OLIVE_ANOMALY_MIN_VALUE,
        "olive_anomaly_max_value": OLIVE_ANOMALY_MAX_VALUE,
        "dark_anomaly_max_value": DARK_ANOMALY_MAX_VALUE,
        "margin_radius_fraction": MARGIN_RADIUS_FRACTION,
        "scene_sparse_min": SCENE_SPARSE_MIN,
        "scene_partial_min": SCENE_PARTIAL_MIN,
        "scene_dominant_min": SCENE_DOMINANT_MIN,
        "white_reference_quantile": WHITE_REFERENCE_QUANTILE,
    }


def weighted_kappa(manual: pd.Series, automatic: pd.Series) -> float:
    pairs = pd.DataFrame({"manual": manual, "automatic": automatic}).dropna()
    if pairs.empty:
        return float("nan")
    labels = list(VEGETATION_STATE_ORDER)
    matrix = pd.crosstab(pairs["manual"], pairs["automatic"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    observed = matrix.to_numpy(dtype=float)
    total = observed.sum()
    if total == 0:
        return float("nan")
    observed /= total
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    positions = np.arange(len(labels), dtype=float)
    weights = ((positions[:, None] - positions[None, :]) / (len(labels) - 1)) ** 2
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else float("nan")


def build_summary(data: pd.DataFrame) -> dict[str, object]:
    ok = data[data["status"] == "ok"].copy()
    valid = ok[
        ok["manual_vegetation_state"].isin(VEGETATION_STATE_ORDER)
        & ok["automatic_vegetation_state"].isin(VEGETATION_STATE_ORDER)
    ].copy()
    def evaluate(frame: pd.DataFrame) -> dict[str, object]:
        if frame.empty:
            return {"n": 0}
        manual_code = frame["manual_vegetation_state"].map(VEGETATION_STATE_ORDER)
        automatic_code = frame["automatic_vegetation_state"].map(VEGETATION_STATE_ORDER)
        confusion = (
            pd.crosstab(
                frame["manual_vegetation_state"],
                frame["automatic_vegetation_state"],
            )
            .reindex(
                index=list(VEGETATION_STATE_ORDER),
                columns=list(VEGETATION_STATE_ORDER),
                fill_value=0,
            )
            .to_dict(orient="index")
        )
        return {
            "n": int(len(frame)),
            "exact_accuracy": float((manual_code == automatic_code).mean()),
            "within_one_class_accuracy": float((manual_code.sub(automatic_code).abs() <= 1).mean()),
            "quadratic_weighted_kappa": weighted_kappa(
                frame["manual_vegetation_state"], frame["automatic_vegetation_state"]
            ),
            "mean_vegetation_fraction_by_manual_state": {
                str(key): float(value)
                for key, value in frame.groupby("manual_vegetation_state")[
                    "vegetation_fraction"
                ].mean().items()
            },
            "confusion_matrix": confusion,
        }

    overall_evaluation = evaluate(valid)

    return {
        "n_records": int(len(data)),
        "n_ok": int(len(ok)),
        "status_counts": {str(k): int(v) for k, v in data["status"].value_counts().items()},
        "sensor_counts": {str(k): int(v) for k, v in ok["sensor"].value_counts().items()},
        "automatic_state_counts": {
            str(k): int(v) for k, v in ok["automatic_vegetation_state"].value_counts().items()
        },
        "manual_state_counts": {
            str(k): int(v) for k, v in valid["manual_vegetation_state"].value_counts().items()
        },
        "evaluation": overall_evaluation,
        "evaluation_by_sensor": {
            str(sensor): evaluate(group)
            for sensor, group in valid.groupby("sensor")
        },
        "spectral_composition_check": {
            "n_ok": int(len(ok)),
            "min_sum": float(ok["spectral_composition_sum"].min()),
            "max_sum": float(ok["spectral_composition_sum"].max()),
            "all_sums_within_1e_minus_6": bool(
                (ok["spectral_composition_sum"] - 1.0).abs().le(1e-6).all()
            ),
        },
        "interpretation": {
            "light_substrate": "高亮度、低饱和度的灰白色裸露基质；颜色不能区分珊瑚断枝与沙",
            "vegetation": "满足绿色色相、饱和度和绿色过量指数阈值的绿色植被信号；不是物种、覆盖度或生物量",
            "dark_olive_anomaly": "相对灰白裸沙更暗或呈橄榄褐色的光谱异常；不能解释为结皮、藻类、湿沙或阴影",
            "unclassified": "不符合绿色植被、灰白基质或暗色异常规则的残余像元",
            "core_margin_structure": "内部植被比例≥0.45 且边缘裸沙比例≥0.40 的植被核心+裸沙边缘结构",
        },
    }


def main() -> None:
    default_dataset = Path(__file__).resolve().parent / "data" / "source_dataset"
    default_output = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()

    sources = [
        (args.dataset_root / "outputs" / "sam_records", "google_earth"),
        (args.dataset_root / "outputs" / "sentinel2_annotations" / "sam_records", "sentinel2"),
    ]
    rows: list[dict[str, object]] = []
    for record_dir, sensor in sources:
        for record_path in sorted(record_dir.glob("*_sam_record.json")):
            rows.append(extract_record(record_path, args.dataset_root, sensor))

    data = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "沙洲颜色组成.csv"
    summary_path = args.output_dir / "颜色判别核查.json"
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(build_summary(data), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"颜色组成: {csv_path}")
    print(f"核查结果: {summary_path}")


if __name__ == "__main__":
    main()
