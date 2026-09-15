"""在已审校的 Sentinel-2 表面状态序列上拟合三状态连续时间马尔可夫模型。

模型输入是同一沙洲相邻的真实拍摄日期，而非年度聚合状态。该脚本只估计
低覆盖、局部覆盖和优势覆盖之间的观测窗口内转移；它不把状态解释为完整的
成岛发育阶段，也不加入 ENSO 或台风协变量。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
INPUT = OUTPUT / "植被状态审校主表.csv"
STATES = ("low_cover", "partial_cover", "dominant_cover")
STATE_TO_INDEX = {state: index for index, state in enumerate(STATES)}
PRIMARY_MIN_DAYS = 30
PRIMARY_MAX_DAYS = 365
BOOTSTRAPS = 100
RNG_SEED = 20260913


def make_pairs(
    observations: pd.DataFrame,
    *,
    min_days: int,
    max_days: int,
    confirmed_only: bool,
) -> pd.DataFrame:
    """返回沙洲内相邻、且处于指定观测窗口的状态对。"""
    data = observations.copy()
    if confirmed_only:
        data = data.loc[data["cover_state_status_3"].eq("confirmed")].copy()
    data = data.loc[data["analysis_cover_state_3"].isin(STATES)].copy()
    data = data.sort_values(["sand_cay_id", "date", "image_id"])
    data = data.drop_duplicates(["sand_cay_id", "date"], keep="last")
    data["next_state"] = data.groupby("sand_cay_id")["analysis_cover_state_3"].shift(-1)
    data["next_date"] = data.groupby("sand_cay_id")["date"].shift(-1)
    pairs = data.loc[data["next_state"].notna()].copy()
    pairs["interval_days"] = (pairs["next_date"] - pairs["date"]).dt.days
    pairs = pairs.loc[pairs["interval_days"].between(min_days, max_days)].copy()
    pairs["from_index"] = pairs["analysis_cover_state_3"].map(STATE_TO_INDEX).astype(int)
    pairs["to_index"] = pairs["next_state"].map(STATE_TO_INDEX).astype(int)
    pairs["interval_years"] = pairs["interval_days"] / 365.2425
    return pairs


def theta_to_q(theta: np.ndarray) -> np.ndarray:
    """将六个无约束对数速率转为三状态生成矩阵（单位：年^-1）。"""
    rates = np.exp(np.asarray(theta, dtype=float))
    q = np.zeros((3, 3), dtype=float)
    cursor = 0
    for source in range(3):
        for target in range(3):
            if source == target:
                continue
            q[source, target] = rates[cursor]
            cursor += 1
        q[source, source] = -q[source].sum()
    return q


def initial_theta(pairs: pd.DataFrame) -> np.ndarray:
    """以暴露时间中的观测变化率作为优化起点。"""
    values: list[float] = []
    for source in range(3):
        exposure = pairs.loc[pairs["from_index"].eq(source), "interval_years"].sum()
        for target in range(3):
            if source == target:
                continue
            count = int(
                ((pairs["from_index"] == source) & (pairs["to_index"] == target)).sum()
            )
            values.append(max((count + 0.25) / max(exposure, 1e-6), 1e-5))
    return np.log(values)


def collapse_likelihood_terms(pairs: pd.DataFrame) -> pd.DataFrame:
    """合并相同起点、终点和观测间隔，避免重复计算相同矩阵指数。"""
    return (
        pairs.groupby(["from_index", "to_index", "interval_years"], as_index=False)
        .size()
        .rename(columns={"size": "n_pairs"})
    )


def negative_log_likelihood(theta: np.ndarray, terms: pd.DataFrame) -> float:
    q = theta_to_q(theta)
    total = 0.0
    for row in terms.itertuples(index=False):
        probability = expm(q * float(row.interval_years))[int(row.from_index), int(row.to_index)]
        total -= int(row.n_pairs) * np.log(max(float(probability), 1e-12))
    return total


def fit_ctmc(pairs: pd.DataFrame) -> tuple[np.ndarray, float]:
    if pairs.empty or pairs["from_index"].nunique() < 3:
        raise ValueError("三种起始状态均需有可用相邻观测对，当前数据不足。")
    terms = collapse_likelihood_terms(pairs)
    result = minimize(
        negative_log_likelihood,
        initial_theta(pairs),
        args=(terms,),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"连续时间模型未收敛：{result.message}")
    return theta_to_q(result.x), float(result.fun)


def model_rows(q: np.ndarray, variant: str, n_pairs: int, n_cays: int, nll: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, source_name in enumerate(STATES):
        exit_rate = -q[source, source]
        rows.append(
            {
                "variant": variant,
                "record_type": "state_summary",
                "from_state": source_name,
                "to_state": "",
                "rate_per_year": exit_rate,
                "mean_holding_years": 1.0 / exit_rate,
                "probability_1_year": np.nan,
                "probability_3_years": np.nan,
                "probability_5_years": np.nan,
                "n_pairs": n_pairs,
                "n_cays": n_cays,
                "negative_log_likelihood": nll,
            }
        )
        for target, target_name in enumerate(STATES):
            if source == target:
                continue
            rows.append(
                {
                    "variant": variant,
                    "record_type": "transition_rate",
                    "from_state": source_name,
                    "to_state": target_name,
                    "rate_per_year": q[source, target],
                    "mean_holding_years": np.nan,
                    "probability_1_year": expm(q * 1.0)[source, target],
                    "probability_3_years": expm(q * 3.0)[source, target],
                    "probability_5_years": expm(q * 5.0)[source, target],
                    "n_pairs": n_pairs,
                    "n_cays": n_cays,
                    "negative_log_likelihood": nll,
                }
            )
    return rows


def bootstrap_primary(pairs: pd.DataFrame) -> pd.DataFrame:
    """按沙洲重抽样，避免把同一沙洲的重复观测当作独立样本。"""
    rng = np.random.default_rng(RNG_SEED)
    cays = pairs["sand_cay_id"].drop_duplicates().to_numpy()
    samples: list[dict[str, object]] = []
    for replicate in range(BOOTSTRAPS):
        selected = rng.choice(cays, size=len(cays), replace=True)
        chunks = []
        for draw_index, cay in enumerate(selected):
            chunk = pairs.loc[pairs["sand_cay_id"].eq(cay)].copy()
            chunk["bootstrap_cay"] = f"draw_{draw_index}"
            chunks.append(chunk)
        sample = pd.concat(chunks, ignore_index=True)
        try:
            q, _ = fit_ctmc(sample)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        for source, source_name in enumerate(STATES):
            samples.append(
                {
                    "replicate": replicate,
                    "metric": "state_summary",
                    "from_state": source_name,
                    "to_state": "",
                    "value": 1.0 / (-q[source, source]),
                }
            )
            for target, target_name in enumerate(STATES):
                if source == target:
                    continue
                samples.append(
                    {
                        "replicate": replicate,
                        "metric": "transition_rate",
                        "from_state": source_name,
                        "to_state": target_name,
                        "value": q[source, target],
                    }
                )
    return pd.DataFrame(samples)


def calibration_table(q: np.ndarray, pairs: pd.DataFrame) -> pd.DataFrame:
    """按观测间隔分箱比较实测频数与模型期望频数，仅用于拟合诊断。"""
    data = pairs.copy()
    bins = [30, 90, 180, 366]
    labels = ["30-89 d", "90-179 d", "180-365 d"]
    data["interval_bin"] = pd.cut(
        data["interval_days"], bins=bins, labels=labels, right=False
    )
    rows: list[dict[str, object]] = []
    for interval_bin, group in data.groupby("interval_bin", observed=True):
        for source, source_name in enumerate(STATES):
            origin = group.loc[group["from_index"].eq(source)]
            if origin.empty:
                continue
            for target, target_name in enumerate(STATES):
                observed = int(origin["to_index"].eq(target).sum())
                expected = float(
                    sum(
                        expm(q * interval)[source, target]
                        for interval in origin["interval_years"]
                    )
                )
                rows.append(
                    {
                        "interval_bin": str(interval_bin),
                        "from_state": source_name,
                        "to_state": target_name,
                        "n_origin_pairs": int(len(origin)),
                        "observed_pairs": observed,
                        "expected_pairs": expected,
                        "observed_minus_expected": observed - expected,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_csv(INPUT, encoding="utf-8-sig", parse_dates=["date"])
    s2 = raw.loc[raw["sensor"].eq("sentinel2")].copy()
    variants = {
        "primary_30_365d": dict(min_days=30, max_days=365, confirmed_only=False),
        "sensitivity_confirmed_30_365d": dict(min_days=30, max_days=365, confirmed_only=True),
        "sensitivity_90_365d": dict(min_days=90, max_days=365, confirmed_only=False),
    }
    all_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    fitted: dict[str, tuple[np.ndarray, pd.DataFrame, float]] = {}
    for name, options in variants.items():
        pairs = make_pairs(s2, **options)
        q, nll = fit_ctmc(pairs)
        fitted[name] = (q, pairs, nll)
        all_rows.extend(model_rows(q, name, len(pairs), pairs["sand_cay_id"].nunique(), nll))
        diagnostics.append(
            {
                "variant": name,
                "n_pairs": int(len(pairs)),
                "n_cays": int(pairs["sand_cay_id"].nunique()),
                "n_state_changes": int((pairs["from_index"] != pairs["to_index"]).sum()),
                "n_change_cays": int(pairs.loc[pairs["from_index"] != pairs["to_index"], "sand_cay_id"].nunique()),
                "interval_median_days": float(pairs["interval_days"].median()),
                "interval_p10_days": float(pairs["interval_days"].quantile(0.10)),
                "interval_p90_days": float(pairs["interval_days"].quantile(0.90)),
                "negative_log_likelihood": nll,
            }
        )
    primary_q, primary_pairs, primary_nll = fitted["primary_30_365d"]
    boot = bootstrap_primary(primary_pairs)
    estimates = pd.DataFrame(all_rows)
    if not boot.empty:
        intervals = (
            boot.groupby(["metric", "from_state", "to_state"], dropna=False)["value"]
            .agg(ci_low=lambda s: s.quantile(0.025), ci_high=lambda s: s.quantile(0.975), n_bootstrap="size")
            .reset_index()
        )
        estimates = estimates.merge(
            intervals,
            how="left",
            left_on=["record_type", "from_state", "to_state"],
            right_on=["metric", "from_state", "to_state"],
        ).drop(columns="metric")
    else:
        estimates["ci_low"] = np.nan
        estimates["ci_high"] = np.nan
        estimates["n_bootstrap"] = 0

    pair_counts = (
        primary_pairs.groupby(["analysis_cover_state_3", "next_state"], dropna=False)
        .agg(n_pairs=("sand_cay_id", "size"), median_interval_days=("interval_days", "median"), n_cays=("sand_cay_id", "nunique"))
        .reset_index()
        .rename(columns={"analysis_cover_state_3": "from_state", "next_state": "to_state"})
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(OUTPUT / "Sentinel2三状态连续时间模型估计.csv", index=False, encoding="utf-8-sig")
    pair_counts.to_csv(OUTPUT / "Sentinel2三状态连续时间观测转移计数.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "Sentinel2三状态连续时间模型诊断.csv", index=False, encoding="utf-8-sig")
    calibration_table(primary_q, primary_pairs).to_csv(
        OUTPUT / "Sentinel2三状态连续时间模型校准.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "model": "three-state continuous-time Markov chain; panel-observation likelihood",
        "states": {
            "low_cover": "人工 none 或 sparse，且与颜色代理相容",
            "partial_cover": "人工 partial，且与颜色代理相容",
            "dominant_cover": "人工 dominant，且与颜色代理相容",
        },
        "primary_window_days": [PRIMARY_MIN_DAYS, PRIMARY_MAX_DAYS],
        "primary_n_pairs": int(len(primary_pairs)),
        "primary_n_cays": int(primary_pairs["sand_cay_id"].nunique()),
        "primary_n_state_changes": int((primary_pairs["from_index"] != primary_pairs["to_index"]).sum()),
        "left_censoring": "每条沙洲序列首次观测之前的状态历时未知。",
        "right_censoring": "每条沙洲序列末次观测之后的状态历时未知。",
        "measurement_error_handling": "主分析排除人工标签与颜色代理冲突的记录；仅完全一致标签和更严格间隔窗口为敏感性分析。",
        "not_estimated": [
            "隐马尔可夫观测误差层",
            "半马尔可夫非指数停留时间",
            "ENSO 或台风对转移率的协变量效应",
            "从水下沉积体到成岛的完整发育时间",
        ],
        "bootstrap": {
            "unit": "sand_cay_id",
            "replicates_requested": BOOTSTRAPS,
            "replicates_completed": int(boot["replicate"].nunique()) if not boot.empty else 0,
            "seed": RNG_SEED,
        },
        "primary_negative_log_likelihood": primary_nll,
    }
    (OUTPUT / "Sentinel2三状态连续时间模型核查.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
