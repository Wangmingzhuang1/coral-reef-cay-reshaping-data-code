"""以原始栅格元数据确定序列资格，不将统一数组尺寸视为配准精度。"""
import pandas as pd
from 项目路径 import ROOT


def single_grid_cays():
    detail = pd.read_csv(ROOT / 'outputs/审稿问题复核/原始红光近红外逐景诊断.csv')
    required = ['crs', 'grid', 'raster_width', 'raster_height']
    if detail[required].isna().any().any() or detail.image_id.duplicated().any():
        raise ValueError('原始网格逐景记录缺失或重复，不能确定序列资格')
    signatures = detail[required].astype(str).agg('|'.join, axis=1)
    counts = signatures.groupby(detail.sand_cay_id).nunique()
    return set(counts[counts.eq(1)].index)
