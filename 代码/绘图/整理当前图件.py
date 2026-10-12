"""将绘图中间文件归位到唯一图号；旧图封存，不覆盖原始证据。"""

import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()
from pathlib import Path
import hashlib, json, shutil

ROOT = PROJECT_ROOT
OUT = ROOT / 'outputs/figures'
ARCHIVE = ROOT.parent / '归档/未采用图件'
FIGURES = [
    ('1', '图1_四区真实演变', 'outputs/figures/图1_四区真实演变', '研究区位置及四区真实个体的面积、边界和位置变化'),
    ('2', '图2_长期轨迹与空间调整', 'outputs/figures/主图/图2_长期轨迹与空间调整', '连续活动包络、净位移、累计路径与首末交换；同地区可比记录对照'),
    ('5', '图3_表面状态双向转换', 'outputs/figures/图3_表面状态双向转换', '同沙洲可观测状态双向变化及描述性联合分布'),
    ('3', '图4_年度结构', 'outputs/figures/figure_2_seasonal_recurrence', '年度轮廓重复和留一年验证'),
    ('4', '图5_环境与移动性联系', 'outputs/figures/图5_环境与移动性联系', '背景方向与高波浪暴露的观测关联'),
    ('6', '图6_覆盖与移动性', 'outputs/figures/图6_覆盖与移动性', '沙洲级、同影像及区间级覆盖移动性联系'),
    ('S1', '图S1_样本来源与跨源控制', 'outputs/figures/补充图/图S1_样本来源与跨源控制', '样本来源、尺寸与样本内跨源校准'),
    ('S2', '图S2_轨迹阈值与采样支持', 'outputs/figures/补充图/图S2_轨迹阈值与采样支持', '轨迹阈值、采样支持与NH_015背景配准筛查'),
    ('S3', '图S3_年度预测对照', 'outputs/figures/补充图/图S3_年度预测对照', '年度模板基线及等训练数量'),
    ('S4', '图S4_方向关联敏感性', 'outputs/figures/补充图/图S4_方向关联敏感性', '方向模型窗口、边界和沙洲剔除'),
    ('S5', '图S5_状态判读与模型校准', 'outputs/figures/补充图/图S5_状态判读与模型校准', '严格标签、状态模型校准与转换前后独立窗口支持及描述性对照'),
    ('S6', '图S6_尺寸几何与可检测性', 'outputs/figures/补充图/图S6_尺寸几何与可检测性', '尺寸几何比例及可检测性'),
    ('S7', '图S7_风场与台风替代检验', 'outputs/figures/补充图/图S7_风场与台风替代检验', '风场增量与匹配事件对照'),
    ('S8', '图S8_长期伸长率与边界敏感性', 'outputs/figures/补充图/图S8_长期伸长率与边界敏感性', '伸长率早晚差异、边界情景与季节支持；真实轮廓说明'),
]

def main():
    FIGURES.sort(key=lambda x: (x[0].startswith("S"), int(x[0].lstrip("S"))))
    for label, name, legacy, role in FIGURES:
        folder = '补充图' if label.startswith('S') else '主图'
        for ext in ['png', 'pdf', 'svg', 'tiff']:
            assert (ROOT/(legacy+'.'+ext)).exists() or (OUT/folder/(name+'.'+ext)).exists(), f'缺少图{label}的{ext}，未移动任何图件'
    current_names = {name for _,name,_,_ in FIGURES}
    for folder in ['主图','补充图']:
        for file in (OUT/folder).iterdir():
            if file.is_file() and file.stem not in current_names:
                target=ROOT.parent/'归档/未采用图件/方案v2调整前'/folder/file.name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(file),str(target))
    entries = []
    for label, name, legacy, role in FIGURES:
        folder = '补充图' if label.startswith('S') else '主图'
        exports = {}
        for ext in ['png', 'pdf', 'svg', 'tiff']:
            source = ROOT / (legacy + '.' + ext)
            target = OUT / folder / (name + '.' + ext)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists() and source != target:
                source.replace(target)
            assert target.exists(), f'缺少当前图件：{target}'
            exports[ext] = {'path': target.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        entries.append({'图号': label, '用途': role, '原绘图位置': legacy, '文件': exports})
    # 仅清理明确的绘图输出目录，不进入源数据或链接。
    excluded = []
    for directory in [OUT, OUT/'中间图', ROOT/'outputs']:
        if not directory.exists():
            continue
        for file in list(directory.iterdir()):
            if file.name in ['当前图件索引.json', '当前图件索引.md']:
                continue
            if directory == ROOT/'outputs' and file.suffix.lower() == '.json':
                continue
            if not file.is_file() or file.suffix.lower() not in ['.png', '.pdf', '.svg', '.tiff', '.json']:
                continue
            target = ARCHIVE / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() == file.read_bytes():
                file.unlink()
            else:
                if target.exists():
                    target = target.with_name(target.stem + '_' + hashlib.sha256(file.read_bytes()).hexdigest()[:10] + target.suffix)
                shutil.move(str(file), str(target))
            excluded.append(str(file.relative_to(ROOT)))
    manifest = {'主图数': sum(not x['图号'].startswith('S') for x in entries), '补充图数': sum(x['图号'].startswith('S') for x in entries), '图件': entries, '说明': '唯一当前图件清单；source_data保留原计算字段名，原绘图文件名仅作来源映射。'}
    (OUT/'当前图件索引.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 当前图件索引', '', '正文和Word只采用以下图件。原绘图名称与现行图号不能互换。PNG供浏览，PDF/SVG供排版，TIFF供投稿导出。', '', '| 图号 | 主要用途 | 当前PNG |', '| --- | --- | --- |']
    for entry in entries:
        p = ROOT / entry['文件']['png']['path']
        lines.append(f"| {entry['图号']} | {entry['用途']} | [{p.name}](<{p.as_posix()}>) |")
    (OUT/'当前图件索引.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'current_figures': len(entries), 'archived_intermediates': len(excluded)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
