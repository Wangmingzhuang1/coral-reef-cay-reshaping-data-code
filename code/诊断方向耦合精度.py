"""诊断：沿轴 vs 横轴方向耦合的置信区间宽度来源（残差噪声 vs 预测量变异）。

仅诊断用：比较两个 within-cay 单变量模型的残差标准差、沙洲内预测量标准差与 R²，
解释为何横轴浪系数 CI 更宽。不改变任何主结果。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent


def main() -> None:
    analysis = pd.read_csv(ROOT / "outputs" / "已成洲沙洲定量分析样本.csv", low_memory=False)
    decomposition = pd.read_csv(
        ROOT / "outputs" / "流场波浪沿轴横轴分解.csv", low_memory=False
    )
    columns = [
        "transition_id",
        "wave_vector_along_mean",
        "wave_vector_cross_mean",
        "centroid_along_shift_normalized_per_year",
        "centroid_cross_shift_normalized_per_year",
    ]
    frame = analysis.merge(decomposition[columns], on="transition_id", how="inner")
    frame = frame.dropna()
    grouped = frame.groupby("sand_cay_id")
    for name, outcome, predictor in [
        (
            "along",
            "centroid_along_shift_normalized_per_year",
            "wave_vector_along_mean",
        ),
        (
            "cross",
            "centroid_cross_shift_normalized_per_year",
            "wave_vector_cross_mean",
        ),
    ]:
        y = frame[outcome] - grouped[outcome].transform("mean")
        x = frame[predictor] - grouped[predictor].transform("mean")
        informative = x.abs().gt(0) & y.notna()
        fitted = sm.OLS(y.loc[informative], x.loc[informative]).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame.loc[informative, "sand_cay_id"]},
        )
        ci_low, ci_high = fitted.conf_int().iloc[0].astype(float)
        print(
            f"{name}: n={int(informative.sum())} cays={int(frame.loc[informative, 'sand_cay_id'].nunique())} "
            f"coef={float(fitted.params.iloc[0]):.3f} ci=[{ci_low:.3f},{ci_high:.3f}] "
            f"width={ci_high - ci_low:.3f} resid_sd={float(np.std(fitted.resid, ddof=1)):.3f} "
            f"within_pred_sd={float(x.loc[informative].std()):.3f} r2={float(fitted.rsquared):.3f}"
        )


if __name__ == "__main__":
    main()
