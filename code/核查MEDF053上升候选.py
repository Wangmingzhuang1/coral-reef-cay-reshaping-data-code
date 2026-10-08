"""Data-level per-scene diagnostics for the MEDF_053 increasing candidate."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "medf053_audit.txt"


def fit(frame: pd.DataFrame) -> float:
    y = np.log1p(frame["centroid_norm_speed_per_year"].to_numpy())
    x = frame["midpoint_year"].to_numpy()
    li = np.log(frame["time_interval_days"].to_numpy())
    month = pd.to_datetime(frame["midpoint"]).dt.month.to_numpy()
    design = np.column_stack([x - x.mean(), li - li.mean(), np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)])
    y = y - y.mean()
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0])


def main() -> None:
    sample = pd.read_csv(ROOT / "全球意义" / "outputs" / "区域趋势分析样本.csv", low_memory=False)
    part = sample.loc[sample["reef_id"].eq("MEDF_053") & sample["specification"].eq("primary_30_400d_normalized")].copy()
    part["midpoint"] = pd.to_datetime(part["midpoint"])
    part = part.sort_values("midpoint").reset_index(drop=True)
    lines = [f"n_intervals={len(part)} span_years={part.midpoint_year.max()-part.midpoint_year.min():.2f}"]
    full = fit(part)
    lines.append(f"full_slope_log_per_year={full:.4f} (log1p units per decade={full*10:.4f})")
    influence = []
    for i in range(len(part)):
        beta_i = fit(part.drop(index=i))
        influence.append((abs(beta_i - full), part.loc[i, "midpoint"].date(), beta_i))
    influence.sort(reverse=True)
    lines.append("top leave-one-out influences (drop date -> log1p slope per decade):")
    for delta, date, beta_i in influence[:4]:
        lines.append(f"  drop {date}: slope={beta_i*10:.4f} (delta={delta:.4f})")
    lines.append("scene table (date, interval_days, area_m2, norm_speed, month):")
    seen = []
    for row in part.itertuples():
        seen.append(f"  {row.midpoint.date()}  {row.time_interval_days:.0f}d  area={row.area_t_m2:.0f}  speed={row.centroid_norm_speed_per_year:.3f}  m{row.midpoint.month:02d}")
    lines.extend(seen)
    early = part.loc[part.midpoint_year.lt(part.midpoint_year.median())]
    late = part.loc[part.midpoint_year.ge(part.midpoint_year.median())]
    lines.append(f"month mean early={early.midpoint.dt.month.mean():.1f} late={late.midpoint.dt.month.mean():.1f}")
    lines.append(f"interval days median early={early.time_interval_days.median():.0f} late={late.time_interval_days.median():.0f}")
    lines.append(f"area median early={early.area_t_m2.median():.0f} late={late.area_t_m2.median():.0f} ratio={late.area_t_m2.median()/early.area_t_m2.median():.2f}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
