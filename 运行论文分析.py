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

sys.path.insert(0,str(Path(__file__).resolve().parent/"代码/公共"))
from 项目路径 import configure_imports
configure_imports()
import os
import numpy as np
import pandas as pd
from 构建沙洲观测与变化表 import select_analysis_observations

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("颜色与掩膜口径", "代码/数据准备/提取沙洲颜色组成.py", []),
    ("观测和变化区间", "代码/公共/构建沙洲观测与变化表.py", []),
    ("植被标签审校", "代码/数据准备/审校植被状态.py", []),
    ("长期面积与质心轨迹", "代码/地貌轨迹/分析沙洲面积轨迹.py", []),
    ("区域与类型描述", "代码/地貌轨迹/分析沙洲类型与区域差异.py", []),
    ("同对象紧致度与年度对照", "代码/控制验证/核查紧致度与质心证据.py", ["--compactness-only"]),
    ("年度轮廓和留一年验证", "代码/时间结构/分析Sentinel2季节形态重复性.py", []),
    ("同季与反季质心分离", "代码/时间结构/分析季节相位质心往复.py", []),
    ("风浪流与观测区间对齐", "代码/数据准备/对齐流场波浪与沙洲变化.py", []),
    ("矢量方向主模型", "代码/控制验证/分析已成洲沙洲形态响应.py", []),
    ("植被与岸段移动性", "代码/表面状态/分析植被形态台风关系.py", []),
    ("覆盖状态转换", "代码/表面状态/拟合Sentinel2三状态连续时间模型.py", []),
    ("严格台风事件对照", "代码/补充分析/分析台风事件前后沙洲响应.py", []),
    ("ENSO滞后关联", "代码/补充分析/分析ENSO与沙洲面积.py", []),
    ("已缓存热应激补充结果", "代码/补充分析/分析白化脉冲与沙洲面积.py", ["--offline", "--reuse-aggregated"]),
    ("区域ENSO交互补充检验", "代码/补充分析/分析区域ENSO滞后响应.py", []),
    ("跨源重塑指标", "代码/地貌轨迹/校准跨源形态区间.py", []),
    ("尺寸几何与跨源描述", "代码/控制验证/检验尺寸几何与跨源一致性.py", []),
    ("区域长期趋势", "代码/补充分析/筛查区域质心趋势.py", []),
    ("高波浪日区间关联", "代码/时间结构/分析裸沙质心轨迹风浪对照.py", []),
    ("质心年度与区间变化", "代码/时间结构/分析裸沙质心轨迹周期与突发.py", []),
    ("组间差异与必要敏感性", "代码/时间结构/分析裸沙主文升级检验.py", []),
    ("边界判读必要敏感性", "代码/控制验证/稳健性扩展检验.py", []),
    ("真实图谱、路径与完整状态转换", "代码/绘图/构建重构图谱.py", []),
    ("双轴区间证据表", "代码/地貌轨迹/构建双轴证据表.py", []),
    ("空间轨迹、转换前后与参考框核查", "代码/控制验证/核查空间轨迹与覆盖转换.py", []),
    ("长期主序列区域客观汇总", "代码/地貌轨迹/汇总长期序列区域特征.py", []),
    ("质心采样、误差与匹配支持", "代码/控制验证/核查紧致度与质心证据.py", ["--spatial-only"]),
    ("审稿测量、原始网格及预测复核", "代码/控制验证/复核审稿主要问题.py", []),
    ("单一原始网格年度主图", "代码/绘图/绘制当前结果图.py", []),
    ("环境与覆盖主图整合", "代码/绘图/整合主图证据.py", []),
    ("按面板重组结果与验证图", "代码/绘图/重组结果与验证图.py", []),
    ("长期形状边界与季节支持审查", "代码/控制验证/审查长期形状指标.py", []),
    ("伸长率补充图与真实轮廓", "代码/绘图/绘制长期形状补充图.py", []),
    ("当前图号归位与历史图封存", "代码/绘图/整理当前图件.py", []),
    ("Word文档同步", "代码/文稿/生成中文论文.py", []),
    ("原件与修改对照核验", "代码/文稿/整理修改对照.py", []),
    ("实际交付与图源指纹核验", "代码/核验/核验当前交付.py", []),
]


def run_command(command: list[str]):
    result = subprocess.run(command, cwd=ROOT, env={**os.environ,"PYTHONPATH":os.pathsep.join(str(x) for x in (ROOT/"代码").iterdir() if x.is_dir())}, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    global_intervals = pd.read_csv(ROOT / "outputs/尺度与跨源/全球形态重塑区间汇总.csv")
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
    parser.add_argument("--build-only", action="store_true", help="使用已核查结果重绘图件并同步Word，不重复拟合模型")
    parser.add_argument("--with-climate", action="store_true", help="额外重算ENSO、DHW及区域趋势")
    parser.add_argument("--shape-audit", action="store_true", help="仅复算93条主序列形状与边界情景，输出候选审计；不修改正文或主图")
    parser.add_argument("--p0-audit",action="store_true",help="复算同对象紧致度、年度同折对照、质心误差与匹配支持，不重写Word")
    args = parser.parse_args()
    if args.p0_audit:
        run_command([sys.executable,"-X","utf8","代码/控制验证/核查紧致度与质心证据.py"])
        print("P0证据核查完成：outputs/P0证据核查",flush=True)
        return
    if args.shape_audit:
        run_command([sys.executable,"-X","utf8","代码/控制验证/审查长期形状指标.py"])
        print("形状审计完成：shape_audit.md；候选结果：outputs/形状审查",flush=True)
        return
    run_command([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"])
    completed = []
    if not args.check_only:
        for index, (label, script, extra) in enumerate(STAGES, 1):
            if args.build_only and not any(x in script for x in ["绘图/", "文稿/", "核验/", "核查空间轨迹与覆盖转换.py", "汇总长期序列区域特征.py", "审查长期形状指标.py"]):
                continue
            if not args.with_climate and any(x in label for x in ["ENSO", "热应激", "区域长期趋势"]):
                continue
            if index < args.start_at:
                continue
            print(f"[{index}/{len(STAGES)}] {label}", flush=True)
            begin = time.monotonic()
            run_command([sys.executable, "-X", "utf8", script, *extra])
            completed.append({"步骤": index, "名称": label, "秒": round(time.monotonic() - begin, 2)})
            print(f"完成：{label}", flush=True)
    audit = validate_current_results()
    if args.check_only and (ROOT / "outputs/figures/当前图件索引.json").exists():
        run_command([sys.executable, "-X", "utf8", "代码/核验/核验当前交付.py"])
    audit["完成步骤"] = completed
    audit["核查时间_UTC"] = datetime.now(timezone.utc).isoformat()
    target = ROOT / "outputs/研究方法核查.json"
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
