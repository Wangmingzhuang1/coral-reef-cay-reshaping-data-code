from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from PIL import Image
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression


RANDOM_SEED = 20260914
METRICS = (
    "vegetation_fraction",
    "sand_cay_area_m2",
    "sand_cay_centroid_x_norm",
    "sand_cay_centroid_y_norm",
    "sand_cay_compactness",
    "sand_cay_elongation",
    "sand_cay_major_axis_angle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检验 Sentinel-2 沙洲轮廓和表面状态的跨年季节重复性。"
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("outputs/沙洲观测主表.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def fdr_bh(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    if not ok.any():
        return pd.Series(result, index=p_values.index)
    valid = values[ok]
    order = np.argsort(valid)
    ranked = valid[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[np.flatnonzero(ok)] = restored
    return pd.Series(result, index=p_values.index)


def paired_wilcoxon(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) == 0 or np.allclose(values, 0):
        return np.nan
    return float(wilcoxon(values).pvalue)


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    n_bootstrap: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    cay_means = frame.groupby(cluster)[value].mean().dropna().to_numpy()
    if len(cay_means) < 2:
        return np.nan, np.nan
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        boot[i] = rng.choice(cay_means, size=len(cay_means), replace=True).mean()
    return tuple(np.quantile(boot, [0.025, 0.975]).tolist())


def load_observations(path: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    frame = pd.read_csv(path)
    frame = frame[
        frame["sensor"].eq("sentinel2") & ~frame["quality_grade"].eq("C")
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["year"] = frame["date"].dt.year
    frame["doy"] = frame["date"].dt.dayofyear
    frame["quarter"] = frame["date"].dt.quarter
    frame["time_years"] = (
        (frame["date"] - frame["date"].min()).dt.days / 365.2425
    )
    frame = frame[frame["mask_path"].map(lambda value: Path(value).is_file())].copy()
    masks = {
        int(index): (np.asarray(Image.open(path)) > 0)
        for index, path in frame["mask_path"].items()
    }
    return frame, masks


def build_pairwise_table(
    frame: pd.DataFrame, masks: dict[int, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cay, group in frame.groupby("sand_cay_id"):
        indices = group.sort_values("date").index.to_list()
        for left_position, left_index in enumerate(indices):
            left = frame.loc[left_index]
            left_mask = masks[int(left_index)]
            for right_index in indices[left_position + 1 :]:
                right = frame.loc[right_index]
                gap_days = int((right["date"] - left["date"]).days)
                if gap_days < 30 or gap_days > 800:
                    continue
                right_mask = masks[int(right_index)]
                if left_mask.shape != right_mask.shape:
                    continue
                intersection = int(np.logical_and(left_mask, right_mask).sum())
                union = int(np.logical_or(left_mask, right_mask).sum())
                area_sum = int(left_mask.sum() + right_mask.sum())
                if union == 0 or area_sum == 0:
                    continue
                phase_days = abs(int(left["doy"]) - int(right["doy"]))
                phase_days = min(phase_days, 365 - phase_days)
                rows.append(
                    {
                        "sand_cay_id": cay,
                        "left_image_id": left["image_id"],
                        "right_image_id": right["image_id"],
                        "gap_days": gap_days,
                        "gap_years": gap_days / 365.2425,
                        "phase_days": phase_days,
                        "phase_distance": 1
                        - np.cos(2 * np.pi * phase_days / 365.2425),
                        "same_quarter": bool(left["quarter"] == right["quarter"]),
                        "same_calendar_year": bool(left["year"] == right["year"]),
                        "iou": intersection / union,
                        "dice": 2 * intersection / area_sum,
                        "symmetric_difference_fraction": 1 - intersection / union,
                    }
                )
    return pd.DataFrame(rows)


def fit_pairwise_model(pairwise: pd.DataFrame) -> dict[str, float | int]:
    model = smf.ols(
        "iou ~ phase_distance + gap_years + I(gap_years ** 2) + C(sand_cay_id)",
        data=pairwise,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pairwise["sand_cay_id"]},
    )
    coefficient = float(model.params["phase_distance"])
    standard_error = float(model.bse["phase_distance"])
    same_year = pairwise[
        ~pairwise["same_calendar_year"]
        & pairwise["phase_days"].le(45)
        & pairwise["gap_days"].between(300, 430)
    ]
    within_year = pairwise[
        pairwise["same_calendar_year"] & pairwise["phase_days"].ge(60)
    ]
    opposite = pairwise[
        ~pairwise["same_calendar_year"]
        & pairwise["phase_days"].ge(120)
        & pairwise["gap_days"].le(800)
    ]
    return {
        "n_pairs": int(len(pairwise)),
        "n_cays": int(pairwise["sand_cay_id"].nunique()),
        "phase_distance_coefficient": coefficient,
        "cluster_robust_se": standard_error,
        "ci95_low": coefficient - 1.96 * standard_error,
        "ci95_high": coefficient + 1.96 * standard_error,
        "two_sided_p_value": float(model.pvalues["phase_distance"]),
        "model_r_squared": float(model.rsquared),
        "same_season_cross_year_n": int(len(same_year)),
        "same_season_cross_year_median_iou": float(same_year["iou"].median()),
        "different_season_within_year_n": int(len(within_year)),
        "different_season_within_year_median_iou": float(within_year["iou"].median()),
        "opposite_season_cross_year_n": int(len(opposite)),
        "opposite_season_cross_year_median_iou": float(opposite["iou"].median()),
    }


def opposite_quarter(quarter: int) -> int:
    return ((quarter + 1) % 4) + 1


def build_template_validation(
    frame: pd.DataFrame,
    masks: dict[int, np.ndarray],
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for index, target_row in frame.iterrows():
        pool = frame[
            frame["sand_cay_id"].eq(target_row["sand_cay_id"])
            & frame["year"].ne(target_row["year"])
        ]
        same = pool[pool["quarter"].eq(target_row["quarter"])]
        opposite = pool[
            pool["quarter"].eq(opposite_quarter(int(target_row["quarter"])))
        ]
        target = masks[int(index)].astype(float)
        same_masks = [
            masks[int(candidate)].astype(float)
            for candidate in same.index
            if masks[int(candidate)].shape == target.shape
        ]
        all_masks = [
            masks[int(candidate)].astype(float)
            for candidate in pool.index
            if masks[int(candidate)].shape == target.shape
        ]
        opposite_masks = [
            masks[int(candidate)].astype(float)
            for candidate in opposite.index
            if masks[int(candidate)].shape == target.shape
        ]
        if len(same_masks) < 2 or len(all_masks) < 6 or len(opposite_masks) < 2:
            continue
        same_probability = np.mean(same_masks, axis=0)
        all_probability = np.mean(all_masks, axis=0)
        opposite_probability = np.mean(opposite_masks, axis=0)
        dynamic_band = (all_probability > 0.05) & (all_probability < 0.95)
        if int(dynamic_band.sum()) < 10:
            continue
        brier_same = float(
            np.mean((target[dynamic_band] - same_probability[dynamic_band]) ** 2)
        )
        brier_all = float(
            np.mean((target[dynamic_band] - all_probability[dynamic_band]) ** 2)
        )
        brier_opposite = float(
            np.mean(
                (target[dynamic_band] - opposite_probability[dynamic_band]) ** 2
            )
        )
        rows.append(
            {
                "image_id": target_row["image_id"],
                "sand_cay_id": target_row["sand_cay_id"],
                "year": int(target_row["year"]),
                "quarter": int(target_row["quarter"]),
                "n_same_quarter_training": len(same_masks),
                "n_all_quarter_training": len(all_masks),
                "n_opposite_quarter_training": len(opposite_masks),
                "dynamic_band_pixels": int(dynamic_band.sum()),
                "brier_same_quarter": brier_same,
                "brier_all_quarters": brier_all,
                "brier_opposite_quarter": brier_opposite,
                "gain_same_vs_all": brier_all - brier_same,
                "gain_same_vs_opposite": brier_opposite - brier_same,
            }
        )
    validation = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for contrast in ("gain_same_vs_all", "gain_same_vs_opposite"):
        cay_values = validation.groupby("sand_cay_id")[contrast].mean()
        low, high = cluster_bootstrap_ci(
            validation, contrast, "sand_cay_id", n_bootstrap
        )
        summaries.append(
            {
                "contrast": contrast,
                "n_images": int(len(validation)),
                "n_cays": int(len(cay_values)),
                "mean_gain": float(cay_values.mean()),
                "median_gain": float(cay_values.median()),
                "ci95_low": low,
                "ci95_high": high,
                "positive_cays": int(cay_values.gt(0).sum()),
                "wilcoxon_p_value": paired_wilcoxon(cay_values),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["fdr_q_value"] = fdr_bh(summary["wilcoxon_p_value"])
    return validation, summary


def harmonic_errors(
    train: pd.DataFrame, test: pd.DataFrame, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    baseline_train = train[["time_years"]].to_numpy()
    baseline_test = test[["time_years"]].to_numpy()
    seasonal_train = np.column_stack(
        [
            train["time_years"],
            np.sin(2 * np.pi * train["doy"] / 365.2425),
            np.cos(2 * np.pi * train["doy"] / 365.2425),
        ]
    )
    seasonal_test = np.column_stack(
        [
            test["time_years"],
            np.sin(2 * np.pi * test["doy"] / 365.2425),
            np.cos(2 * np.pi * test["doy"] / 365.2425),
        ]
    )
    if metric == "sand_cay_major_axis_angle":
        doubled = np.deg2rad(2 * train[metric].to_numpy())
        target_train = np.column_stack([np.cos(doubled), np.sin(doubled)])
        target_angle = np.deg2rad(2 * test[metric].to_numpy())
        target_test = np.column_stack([np.cos(target_angle), np.sin(target_angle)])
        baseline_prediction = LinearRegression().fit(
            baseline_train, target_train
        ).predict(baseline_test)
        seasonal_prediction = LinearRegression().fit(
            seasonal_train, target_train
        ).predict(seasonal_test)
        baseline_error = np.sqrt(
            np.sum((target_test - baseline_prediction) ** 2, axis=1)
        )
        seasonal_error = np.sqrt(
            np.sum((target_test - seasonal_prediction) ** 2, axis=1)
        )
        return baseline_error, seasonal_error
    target_train = train[metric].to_numpy()
    target_test = test[metric].to_numpy()
    baseline_prediction = LinearRegression().fit(
        baseline_train, target_train
    ).predict(baseline_test)
    seasonal_prediction = LinearRegression().fit(
        seasonal_train, target_train
    ).predict(seasonal_test)
    scale = float(
        train[metric].quantile(0.75) - train[metric].quantile(0.25)
    )
    if not np.isfinite(scale) or scale <= 0:
        scale = float(train[metric].std())
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (
        np.abs(target_test - baseline_prediction) / scale,
        np.abs(target_test - seasonal_prediction) / scale,
    )


def build_harmonic_validation(
    frame: pd.DataFrame, n_bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        data = frame[
            ["image_id", "sand_cay_id", "year", "doy", "time_years", metric]
        ].dropna()
        for cay, group in data.groupby("sand_cay_id"):
            if group["year"].nunique() < 4 or len(group) < 12:
                continue
            for held_out_year, test in group.groupby("year"):
                train = group[group["year"].ne(held_out_year)]
                if len(train) < 9:
                    continue
                baseline_error, seasonal_error = harmonic_errors(
                    train, test, metric
                )
                for (_, test_row), base, seasonal in zip(
                    test.iterrows(), baseline_error, seasonal_error
                ):
                    rows.append(
                        {
                            "metric": metric,
                            "image_id": test_row["image_id"],
                            "sand_cay_id": cay,
                            "held_out_year": int(held_out_year),
                            "baseline_error": float(base),
                            "seasonal_error": float(seasonal),
                            "error_reduction": float(base - seasonal),
                        }
                    )
    validation = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for metric, group in validation.groupby("metric", sort=False):
        cay_values = group.groupby("sand_cay_id")["error_reduction"].mean()
        low, high = cluster_bootstrap_ci(
            group, "error_reduction", "sand_cay_id", n_bootstrap
        )
        summaries.append(
            {
                "metric": metric,
                "n_observations": int(len(group)),
                "n_cays": int(len(cay_values)),
                "mean_error_reduction": float(cay_values.mean()),
                "median_error_reduction": float(cay_values.median()),
                "ci95_low": low,
                "ci95_high": high,
                "positive_cays": int(cay_values.gt(0).sum()),
                "wilcoxon_p_value": paired_wilcoxon(cay_values),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["fdr_q_value"] = fdr_bh(summary["wilcoxon_p_value"])
    return validation, summary


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_summary(
    pairwise_summary: dict[str, float | int],
    template_summary: pd.DataFrame,
    harmonic_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    ax = axes[0]
    coefficient = float(pairwise_summary["phase_distance_coefficient"])
    low = float(pairwise_summary["ci95_low"])
    high = float(pairwise_summary["ci95_high"])
    ax.errorbar(
        coefficient,
        0,
        xerr=[[coefficient - low], [high - coefficient]],
        fmt="o",
        color="#18766c",
        capsize=3,
    )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set(
        yticks=[0],
        yticklabels=["Annual phase distance"],
        xlabel="Adjusted change in mask IoU",
        title="a  Pairwise seasonal recurrence",
    )

    ax = axes[1]
    template_labels = {
        "gain_same_vs_all": "Same vs all seasons",
        "gain_same_vs_opposite": "Same vs opposite season",
    }
    y = np.arange(len(template_summary))[::-1]
    for position, (_, row) in zip(y, template_summary.iterrows()):
        ax.errorbar(
            row["mean_gain"],
            position,
            xerr=[
                [row["mean_gain"] - row["ci95_low"]],
                [row["ci95_high"] - row["mean_gain"]],
            ],
            fmt="o",
            color="#b85c38" if row["fdr_q_value"] < 0.05 else "#888888",
            capsize=3,
        )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set(
        yticks=y,
        yticklabels=[
            template_labels[value] for value in template_summary["contrast"]
        ],
        xlabel="Brier error reduction",
        title="b  Leave-one-year-out masks",
    )

    ax = axes[2]
    labels = {
        "vegetation_fraction": "Vegetation proxy",
        "sand_cay_area_m2": "Area",
        "sand_cay_centroid_x_norm": "Centroid east–west",
        "sand_cay_centroid_y_norm": "Centroid north–south",
        "sand_cay_compactness": "Compactness",
        "sand_cay_elongation": "Elongation",
        "sand_cay_major_axis_angle": "Major-axis orientation",
    }
    harmonic = harmonic_summary.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(harmonic))
    for position, (_, row) in zip(y, harmonic.iterrows()):
        ax.errorbar(
            row["mean_error_reduction"],
            position,
            xerr=[
                [row["mean_error_reduction"] - row["ci95_low"]],
                [row["ci95_high"] - row["mean_error_reduction"]],
            ],
            fmt="o",
            color="#18766c" if row["fdr_q_value"] < 0.05 else "#888888",
            capsize=2,
        )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set(
        yticks=y,
        yticklabels=[labels[value] for value in harmonic["metric"]],
        xlabel="Cross-validated error reduction",
        title="c  Annual harmonic",
    )
    for panel in axes:
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(output_dir / "Sentinel2季节形态重复性.png", dpi=400)
    fig.savefig(output_dir / "Sentinel2季节形态重复性.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_columns = pd.read_csv(
        args.observations, usecols=["sensor", "quality_grade"]
    )
    excluded_quality_c = int(
        (
            quality_columns["sensor"].eq("sentinel2")
            & quality_columns["quality_grade"].eq("C")
        ).sum()
    )
    observations, masks = load_observations(args.observations)
    pairwise = build_pairwise_table(observations, masks)
    pairwise_summary = fit_pairwise_model(pairwise)
    template, template_summary = build_template_validation(
        observations, masks, args.bootstrap
    )
    harmonic, harmonic_summary = build_harmonic_validation(
        observations, args.bootstrap
    )
    pairwise.to_csv(
        args.output_dir / "Sentinel2季节形态配对.csv",
        index=False,
        encoding="utf-8-sig",
    )
    template.to_csv(
        args.output_dir / "Sentinel2季节模板交叉验证.csv",
        index=False,
        encoding="utf-8-sig",
    )
    template_summary.to_csv(
        args.output_dir / "Sentinel2季节模板检验汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    harmonic.to_csv(
        args.output_dir / "Sentinel2季节谐波交叉验证.csv",
        index=False,
        encoding="utf-8-sig",
    )
    harmonic_summary.to_csv(
        args.output_dir / "Sentinel2季节谐波检验汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_summary(
        pairwise_summary,
        template_summary,
        harmonic_summary,
        args.output_dir,
    )
    audit = {
        "analysis_contract": {
            "scientific_question": (
                "Do Sentinel-2 sand-cay masks and surface metrics contain a "
                "repeatable annual component beyond cay identity and secular trend?"
            ),
            "sample_unit": (
                "pair of masks within a fixed reference frame, or one held-out "
                "image-year for cross-validation"
            ),
            "season_definition": (
                "continuous circular day-of-year phase for the main pairwise and "
                "harmonic tests; calendar quarter only for template construction"
            ),
            "prediction_boundary": (
                "all templates and harmonic models exclude the target calendar year"
            ),
            "inference_boundary": (
                "observational recurrence, not proof of a seasonal hydrodynamic "
                "mechanism or vegetation phenology"
            ),
        },
        "input": {
            "observations": str(args.observations),
            "n_observations": int(len(observations)),
            "n_cays": int(observations["sand_cay_id"].nunique()),
            "n_reefs": int(observations["reef_id"].nunique()),
            "date_start": observations["date"].min().date().isoformat(),
            "date_end": observations["date"].max().date().isoformat(),
            "excluded_quality_c": excluded_quality_c,
        },
        "pairwise_model": pairwise_summary,
        "template_tests": template_summary.to_dict(orient="records"),
        "harmonic_tests": harmonic_summary.to_dict(orient="records"),
        "known_limitations": [
            "Tide elevation could not be included because the local FES2022b grid is absent.",
            "Quarterly scene selection and segmentation uncertainty may contribute to apparent seasonality.",
            "The RGB-derived vegetation fraction is a surface-colour proxy, not a field-validated phenology measure.",
            "A significant annual component does not identify waves, currents, wind, tide, or vegetation as its cause.",
        ],
    }
    with (args.output_dir / "Sentinel2季节形态重复性核查.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
