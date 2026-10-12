"""定向复核图6b网格资格及主要案例背景筛查，不改写观测或拟合新机制。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '公共'))
from 项目路径 import ROOT
import argparse
import json
import zipfile
import numpy as np
import pandas as pd
from 原始网格核验 import single_grid_cays

OUT = ROOT / 'outputs/审稿问题复核'


def sector():
    from 分析植被形态台风关系 import prepare, vegetation_sector_contrast, exclude_reefs
    from 构建沙洲观测与变化表 import select_analysis_observations
    with zipfile.ZipFile(ROOT.parent / '归档/定向完善前.zip') as archive:
        prior = pd.read_csv(archive.open('outputs/植被扇区对照.csv'))
    eligible = single_grid_cays()
    audit = prior[['transition_id', 'sand_cay_id']].copy()
    audit['single_original_grid_sequence'] = audit.sand_cay_id.isin(eligible)
    audit['decision'] = np.where(audit.single_original_grid_sequence, 'retain', 'exclude_mixed_grid_sequence')
    audit.to_csv(OUT / '图6b原始网格样本审计.csv', index=False, encoding='utf-8-sig')
    intervals = exclude_reefs(pd.read_csv(ROOT / 'outputs/沙洲变化区间.csv'))
    valid, _ = prepare(intervals)
    obs = select_analysis_observations(pd.read_csv(ROOT / 'outputs/沙洲观测主表.csv'))
    frame, stats = vegetation_sector_contrast(valid, pd.read_csv(ROOT / 'outputs/沙洲颜色组成.csv'), obs)
    expected = prior[prior.sand_cay_id.isin(eligible)]
    assert set(frame.transition_id) == set(expected.transition_id)
    for name in ['mobility_difference', 'retreat_difference']:
        x = frame.set_index('transition_id')[name].sort_index()
        y = expected.set_index('transition_id')[name].sort_index()
        assert np.allclose(x, y, atol=1e-12)
    frame.to_csv(ROOT / 'outputs/植被扇区对照.csv', index=False, encoding='utf-8-sig')
    stats.to_csv(ROOT / 'outputs/植被扇区对照统计.csv', index=False, encoding='utf-8-sig')
    summary_path = ROOT / 'outputs/植被形态台风分析核查.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary['sector_contrasts'] = stats.to_dict('records')
    summary['sector_grid_policy'] = 'single_original_grid_sequence; 4 mixed-grid sequences excluded'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(stats.to_json(orient='records', force_ascii=False), flush=True)
    return {'before_intervals': len(prior), 'after_intervals': len(frame),
            'after_cays': frame.sand_cay_id.nunique(), 'excluded_intervals': len(prior)-len(frame),
            'preserved_retained_values': True, 'scope': 'single-grid metadata eligibility, not independent registration accuracy'}


def background():
    from 核查空间轨迹与覆盖转换 import track, read_mask, registration_background_screen
    source = pd.read_csv(ROOT / 'outputs/figures/source_data/表A_沙洲级面积路径状态.csv')
    # 一次预先固定的物理尺度方案；候选礁面纹理不是独立测量控制点。
    targets = ['MEDF_008_cay_001', 'MEDF_022_cay_001', 'GBR_039_cay_001',
               'MEDF_015_cay_001', 'OTHER_008_cay_001']
    rows, support = [], []
    for cay in targets:
        record = source[source.sand_cay_id.eq(cay) & source.sensor.eq('sentinel2')].iloc[0]
        g = track(record)
        masks = [read_mask(p) for p in g.resolved_mask_path]
        coords = g[['sand_cay_centroid_x', 'sand_cay_centroid_y']].to_numpy()
        for label, options in [('historical', {}), ('physical_scale',
                dict(buffer_m=20, edge_m=10, compact_features=True))]:
            frame = registration_background_screen(g, coords, masks, **options)
            frame['sand_cay_id'], frame['specification'] = cay, label
            rows.append(frame)
            support.append(dict(sand_cay_id=cay, specification=label, n_pairs=len(frame),
                n_accepted=int(frame.screen_accepted.sum()),
                background_fraction=float(frame.background_fraction.iloc[0]),
                median_features=float(frame.n_features_t.median())))
        print(cay, support[-2:], flush=True)
    pd.concat(rows).to_csv(OUT / '关键案例背景测量筛查.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(support).to_csv(OUT / '关键案例背景测量支持.csv', index=False, encoding='utf-8-sig')
    return {'fixed_buffer_m': 20, 'fixed_edge_m': 10, 'support': support,
            'scope': 'candidate background matching only; no independent controls, no mask correction; acceptance thresholds unchanged'}


def lag():
    from 分析植被形态台风关系 import sector_causal_tests, exclude_reefs
    from 构建沙洲观测与变化表 import select_analysis_observations
    intervals = exclude_reefs(pd.read_csv(ROOT / 'outputs/沙洲变化区间.csv'))
    obs = select_analysis_observations(pd.read_csv(ROOT / 'outputs/沙洲观测主表.csv'))
    result = sector_causal_tests(intervals, obs, pd.read_csv(ROOT / 'outputs/沙洲颜色组成.csv'))
    result.to_csv(ROOT / 'outputs/植被扇区因果滞后检验.csv', index=False, encoding='utf-8-sig')
    summary_path = ROOT / 'outputs/植被形态台风分析核查.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary['adjacent_sector_associations'] = result.fillna('').to_dict('records')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result.to_json(orient='records', force_ascii=False), flush=True)
    return result.to_dict('records')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['all', 'sector', 'background', 'lag'], default='all')
    args = parser.parse_args()
    destination = OUT / '定向完善复核.json'
    result = json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {}
    for name, fn in [('sector', sector), ('background', background), ('lag', lag)]:
        if args.part in ['all', name]:
            result[name] = fn()
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
