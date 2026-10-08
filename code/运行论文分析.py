"""按依赖顺序运行论文一的现有分析，最后核验结果；不新增模型。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from 构建沙洲观测与变化表 import select_analysis_observations

ROOT = Path(__file__).resolve().parent
ARCHIVE = "额外分析_区域与气候响应/归档_裸沙质心轨迹周期与突发"
STAGES = [
    ("颜色与掩膜口径", "提取沙洲颜色组成.py", []),
    ("观测和变化区间", "构建沙洲观测与变化表.py", []),
    ("植被标签审校", "审校植被状态.py", []),
    ("长期面积与质心轨迹", "分析沙洲面积轨迹.py", []),
    ("区域与类型描述", "分析沙洲类型与区域差异.py", []),
    ("年度轮廓和留一年验证", "分析Sentinel2季节形态重复性.py", []),
    ("同季与反季质心分离", "分析季节相位质心往复.py", []),
    ("风浪流与观测区间对齐", "对齐流场波浪与沙洲变化.py", []),
    ("矢量方向主模型", "分析已成洲沙洲形态响应.py", []),
    ("植被与岸段移动性", "分析植被形态台风关系.py", []),
    ("覆盖状态转换", "拟合Sentinel2三状态连续时间模型.py", []),
    ("严格台风事件对照", "分析台风事件前后沙洲响应.py", []),
    ("ENSO滞后关联", "分析ENSO与沙洲面积.py", []),
    ("已缓存热应激补充结果", "额外分析_区域与气候响应/分析白化脉冲与沙洲面积.py", ["--offline", "--reuse-aggregated"]),
    ("跨源重塑指标", "全球意义/分析全球形态重塑谱系.py", []),
    ("尺寸几何与跨源描述", "全球意义/分析全球普适性与可迁移性.py", []),
    ("区域长期趋势", "全球意义/分析区域质心移动趋势.py", []),
    ("质心年度与区间变化", f"{ARCHIVE}/分析裸沙质心轨迹周期与突发.py", []),
    ("高波浪日区间关联", f"{ARCHIVE}/分析裸沙质心轨迹风浪对照.py", []),
    ("组间差异与必要敏感性", f"{ARCHIVE}/分析裸沙主文升级检验.py", []),
    ("边界判读必要敏感性", "稳健性扩展检验.py", []),
    ("区域及研究地图", "额外分析_区域与气候响应/绘制研究区域与区域结果图.py", []),
    ("尺寸与区域趋势图", "全球意义/绘制全球综合图.py", []),
    ("当前主文与补充图", "绘制当前结果图.py", []),
]


def run_command(command: list[str]):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        print(result.stdout[-4000:], flush=True)
        print(result.stderr[-4000:], flush=True)
        raise RuntimeError(f"步骤失败，已停止以防旧结果混入：{command}")
    return result


def validate_current_results() -> dict:
    obs = pd.read_csv(ROOT / "outputs/沙洲观测主表.csv")
    selected = select_analysis_observations(obs)
    keys = ["sensor", "sand_cay_id", "reference_frame_id", "date"]
    assert not selected.duplicated(keys).any(), "同一分析时点被重复计数"
    assert not obs.duplicated(["sensor", "image_id"]).any(), "主表影像身份重复"
    assert selected.groupby("reference_frame_id").pixel_size_m.nunique().le(1).all(), "固定框像素尺寸不一致"
    assert np.allclose(selected.pixel_count, selected.sand_cay_area_pixels), "颜色与形态掩膜口径不一致"
    intervals = pd.read_csv(ROOT / "outputs/沙洲变化区间.csv")
    assert not intervals.transition_id.duplicated().any(), "变化区间身份重复"
    balance = intervals.area_change_m2 - intervals.deposition_area_m2 + intervals.erosion_area_m2
    balance_error = float(np.nanmax(np.abs(balance)))
    assert balance_error < 1e-5, "原始同源面积收支不一致"
    assert intervals.time_interval_days.gt(0).all(), "存在零或负时间间隔"
    assert intervals.quality_grade_t.isin(["A", "B"]).all() and intervals.quality_grade_t1.isin(["A", "B"]).all(), "低质量观测进入变化分析"
    direction = pd.read_csv(ROOT / "outputs/流场波浪沿轴横轴分解.csv")
    expected = selected.assign(time_t=pd.to_datetime(selected.date)).set_index(["sensor", "sand_cay_id", "reference_frame_id", "time_t"]).sand_cay_major_axis_angle
    probe = direction.copy()
    probe["time_t"] = pd.to_datetime(probe.time_t)
    target = expected.reindex(pd.MultiIndex.from_frame(probe[["sensor", "sand_cay_id", "reference_frame_id", "time_t"]]))
    assert np.allclose(probe.sand_cay_major_axis_angle, target.to_numpy()), "方向轴与起始影像源/参考框不匹配"
    models = pd.read_csv(ROOT / "outputs/强迫植被沙洲内模型.csv")
    valid = models.dropna(subset=["coefficient", "ci95_low", "ci95_high", "fdr_q_value"])
    assert valid.fdr_q_value.between(0, 1).all(), "FDR数值越界"
    assert (valid.ci95_low.le(valid.coefficient) & valid.ci95_high.ge(valid.coefficient)).all(), "置信区间与系数不一致"
    state = pd.read_csv(ROOT / "outputs/Sentinel2三状态连续时间模型估计.csv")
    means = state[state.record_type.eq("state_summary")]
    assert means.mean_holding_years.gt(0).all(), "状态平均停留时间非法"
    assert np.allclose(means.rate_per_year * means.mean_holding_years, 1), "状态退出率与平均停留时间单位不一致"
    global_intervals = pd.read_csv(ROOT / "全球意义/outputs/全球形态重塑区间汇总.csv")
    paths = pd.read_csv(ROOT / "outputs/沙洲面积主轨迹分类.csv")
    audit = {"status": "passed", "原始观测": len(obs), "分析观测": len(selected),
             "同源变化区间": len(intervals), "原始收支最大误差_m2": balance_error,
             "长期主序列": len(paths), "跨源描述区间": len(global_intervals),
             "主要分析": "原矢量固定效应、分响应跨源校准、年度留出、植被关联及状态模型"}
    audit["输入指纹"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in ["outputs/沙洲观测主表.csv", "outputs/沙洲变化区间.csv"]}
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只运行独立数值测试和现有结果一致性核查")
    parser.add_argument("--start-at", type=int, default=1, help="从指定步骤继续已完成的分析，编号从1开始")
    args = parser.parse_args()
    run_command([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"])
    completed = []
    if not args.check_only:
        for index, (label, script, extra) in enumerate(STAGES, 1):
            if index < args.start_at:
                continue
            print(f"[{index}/{len(STAGES)}] {label}", flush=True)
            begin = time.monotonic()
            run_command([sys.executable, "-X", "utf8", script, *extra])
            completed.append({"步骤": index, "名称": label, "秒": round(time.monotonic() - begin, 2)})
            print(f"完成：{label}", flush=True)
    audit = validate_current_results()
    audit["完成步骤"] = completed
    audit["核查时间_UTC"] = datetime.now(timezone.utc).isoformat()
    target = ROOT / "outputs/研究方法核查.json"
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
