"""汇总同一套长期主序列的区域事实；不校准、不检验区域机制差异。"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '公共'))
from 项目路径 import ROOT

SOURCE = ROOT / 'outputs/轨迹与转换核查/连续空间轨迹与首末边界交换.csv'
OUT = ROOT / 'outputs/轨迹与转换核查'
REGIONS = ['Great Barrier Reef', 'Marshall Islands', 'Maldives', 'South China Sea', 'Lakshadweep']
METRICS = ['span_years', 'n_observations', 'start_area_m2', 'centroid_envelope_normalized',
           'centroid_net_normalized', 'centroid_path_normalized', 'endpoint_exchange_fraction']


def summarize(frame):
    rows = []
    for region in REGIONS:
        group = frame[frame.region.eq(region)]
        if group.empty:
            continue
        row = {'region': region, 'n_primary': len(group), 'n_sentinel2': int(group.sensor.eq('sentinel2').sum()),
               'n_google_earth': int(group.sensor.eq('google_earth').sum()),
               'n_near_stable': int(group.net_area_bin.eq('Near-stable area').sum()),
               'n_net_gain': int(group.net_area_bin.eq('Net gain').sum()),
               'n_net_loss': int(group.net_area_bin.eq('Net loss').sum()),
               'start_date': group.start_date.min(), 'end_date': group.end_date.max()}
        for metric in METRICS:
            values = group[metric].dropna()
            row[metric + '_n'] = len(values)
            for label, value in [('q25', values.quantile(.25)), ('median', values.median()),
                                 ('q75', values.quantile(.75)), ('min', values.min()), ('max', values.max())]:
                row[metric + '_' + label] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    frame = pd.read_csv(SOURCE)
    assert len(frame) == 93 and frame.sand_cay_id.nunique() == 93
    assert set(frame.region) == set(REGIONS)
    expected = frame.relative_change.map(lambda x: 'Net gain' if x >= .25 else ('Net loss' if x <= -.25 else 'Near-stable area'))
    assert expected.eq(frame.net_area_bin).all()
    regional = summarize(frame)
    regional.to_csv(OUT / '长期主序列区域汇总.csv', index=False, encoding='utf-8-sig')
    by_source = []
    for sensor, group in frame.groupby('sensor'):
        table = summarize(group)
        table.insert(1, 'sensor', sensor)
        by_source.append(table)
    pd.concat(by_source, ignore_index=True).to_csv(OUT / '长期主序列区域分源汇总.csv', index=False, encoding='utf-8-sig')
    provenance = {'input': str(SOURCE.relative_to(ROOT)), 'input_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  'n_primary': len(frame), 'n_four_focus_regions': int(frame.region.ne('Lakshadweep').sum()),
                  'unit': 'one selected same-source fixed-frame primary series per cay',
                  'quantiles': 'pandas linear; cay-level distribution; no regional significance test',
                  'area_bins': 'first-two/last-three area medians, relative change thresholds +/-25%; outcome bins, not geomorphic types',
                  'position_scope': 'stored positions, uncorrected for registration; cumulative path depends on sampling',
                  'endpoint_exchange_scope': 'comparable first/last masks; fraction of first-scene area; not early/late median area balance'}
    (OUT / '区域汇总来源.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(regional[['region', 'n_primary', 'n_near_stable', 'n_net_gain', 'n_net_loss']].to_string(index=False))


if __name__ == '__main__':
    main()
