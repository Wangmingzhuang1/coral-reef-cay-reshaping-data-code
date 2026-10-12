"""审查93条主序列的形状定义、早晚窗口与边界敏感性；不分类、不拟合新模型。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'公共'))
from 项目路径 import ROOT
from 构建沙洲观测与变化表 import select_analysis_observations,read_mask,perturb_mask
import json,hashlib,itertools
import numpy as np
import pandas as pd
import cv2
from skimage.measure import regionprops,perimeter_crofton
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'outputs/形状审查';OUT.mkdir(parents=True,exist_ok=True)
OBS=select_analysis_observations(pd.read_csv(ROOT.parent/'归档/重构前/第一阶段审计/有效观测与图像掩膜清单.csv'))
OBS.date=pd.to_datetime(OBS.date)
PRIMARY=pd.read_csv(ROOT/'outputs/轨迹与转换核查/连续空间轨迹与首末边界交换.csv')
PRIMARY=PRIMARY.drop(columns=['Index','_1'],errors='ignore')

def save(d,name):d.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def metrics(mask):
    if mask is None or not mask.any():return dict(A=np.nan,P=np.nan,C=np.nan,E=np.nan,C_crofton=np.nan,major=np.nan,minor=np.nan)
    yy,xx=np.where(mask);mask=mask[max(0,yy.min()-4):yy.max()+5,max(0,xx.min()-4):xx.max()+5]
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),connectivity=8)
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])); primary=labels==largest
    p=regionprops(primary.astype('uint8'))[0];pc=perimeter_crofton(primary,directions=4)
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 真实存储口径：最大连通域面积/轴，但周长为原掩膜全部外轮廓长度之和。
    perimeter=sum(cv2.arcLength(c,True) for c in contours)
    primary_contours,_=cv2.findContours(primary.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pp=sum(cv2.arcLength(c,True) for c in primary_contours)
    return dict(A=float(p.area),P=float(perimeter),C=float(4*np.pi*p.area/perimeter**2) if perimeter>0 else np.nan,E=float(p.axis_major_length/p.axis_minor_length) if p.axis_minor_length>0 else np.nan,C_crofton=float(4*np.pi*p.area/pc**2),C_primary=float(4*np.pi*p.area/pp**2) if pp>0 else np.nan,components=len(stats)-1,major=float(p.axis_major_length),minor=float(p.axis_minor_length))

def window_scenario_range(values, early, late, key):
    # 同一日期只取一个扰动；共享早晚日期使用同一变量。
    indices=list(values)
    shifts=[]
    for choice in itertools.product(range(3),repeat=len(indices)):
        selected={i:values[i][j][key] for i,j in zip(indices,choice)}
        shifts.append(np.median([selected[i] for i in late])-np.median([selected[i] for i in early]))
    return float(np.nanmin(shifts)),float(np.nanmax(shifts))

def run():
    traces=[];rows=[];variants=[]
    for p in PRIMARY.itertuples(index=False):
        g=OBS[(OBS.sand_cay_id==p.sand_cay_id)&(OBS.sensor==p.sensor)&(OBS.reference_frame_id==p.reference_frame_id)].sort_values('date').reset_index(drop=True)
        assert len(g)==p.n_observations
        C=4*np.pi*g.sand_cay_area_pixels/g.sand_cay_perimeter_pixels**2;E=g.sand_cay_major_axis_length/g.sand_cay_minor_axis_length
        assert np.allclose(C,g.sand_cay_compactness) and np.allclose(E,g.sand_cay_elongation)
        idx=list(dict.fromkeys([0,1,len(g)-3,len(g)-2,len(g)-1]));v={}
        for i in idx:
            x=g.iloc[i];m=cv2.imdecode(np.fromfile(x.resolved_mask_path,dtype='uint8'),cv2.IMREAD_GRAYSCALE)>0
            v[i]=[metrics(perturb_mask(m,k)) for k in [-1,0,1]]
            base=v[i][1]
            traces.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,reference_frame_id=p.reference_frame_id,date=str(x.date.date()),image_id=x.image_id,quality=x.quality_grade,mask_path=x.resolved_mask_path,pixel_size_m=x.pixel_size_m,stored_C=x.sand_cay_compactness,recomputed_C=base['C'],stored_E=x.sand_cay_elongation,recomputed_E=base['E'],stored_P=x.sand_cay_perimeter_pixels,recomputed_P=base['P'],stored_minor_pixels=x.sand_cay_minor_axis_length,recomputed_minor_pixels=base['minor'],C_crofton=base['C_crofton'],C_primary=base['C_primary'],components=base['components']))
            for k,z in zip([-1,0,1],v[i]):variants.append(dict(sand_cay_id=p.sand_cay_id,date=str(x.date.date()),perturbation_pixels=k,**z))
        row=p._asdict();row.update(shape_observations=len(g),C_gt1_observations=int((C>1).sum()),invalid_observations=int((~np.isfinite(C)|~np.isfinite(E)|(C<=0)|(E<1)).sum()),median_C=float(C.median()),median_E=float(E.median()),C_early=float(C.iloc[:2].median()),C_late=float(C.iloc[-3:].median()),E_early=float(E.iloc[:2].median()),E_late=float(E.iloc[-3:].median()),anchor_overlap=len(g)<5,minimum_minor_pixels=float(g.sand_cay_minor_axis_length.min()))
        row.update(window_centroid_metrics(g))
        row.update(delta_C=row['C_late']-row['C_early'],delta_E=row['E_late']-row['E_early'],delta_log_area=float(np.log(p.end_area_m2/p.start_area_m2)))
        angles=2*np.pi*g.date.dt.dayofyear.to_numpy()/365.2425
        phi0=np.angle(np.mean(np.exp(1j*angles[:2])));phi1=np.angle(np.mean(np.exp(1j*angles[-3:])))
        row['early_late_season_gap_days']=float(abs(np.angle(np.exp(1j*(phi1-phi0))))*365.2425/(2*np.pi))
        day=g.date.dt.dayofyear.to_numpy(dtype=float)
        gap=np.abs(day[:2,None]-day[None,-3:]);gap=np.minimum(gap,365.2425-gap)
        row['maximum_cross_window_season_gap_days']=float(gap.max())
        row['anchor_C_in_range']=bool(C.iloc[idx].between(0,1).all())
        for key in ['C','E','C_crofton']:
            low,high=window_scenario_range(v,[0,1],list(range(len(g)-3,len(g))),key)
            row['delta_'+key+'_scenario_min']=float(low);row['delta_'+key+'_scenario_max']=float(high)
            row[key+'_direction_preserved_1px']=bool(low>0 or high<0)
        row['C_methods_same_direction']=bool(np.sign(row['delta_C'])==np.sign(row['delta_C_crofton_scenario_min'])==np.sign(row['delta_C_crofton_scenario_max']))
        rows.append(row)
        print(p.sand_cay_id,flush=True)
    d=pd.DataFrame(rows);tr=pd.DataFrame(traces);save(d,'主序列形状分布与窗口变化');save(tr,'早晚掩膜来源与算法复核');save(pd.DataFrame(variants),'逐景一像素边界情景')
    reproduction={k:float(np.max(np.abs(tr['stored_'+k]-tr['recomputed_'+k]))) for k in ['C','E','P']}
    assert reproduction['C']<1e-10 and reproduction['E']<1e-10 and reproduction['P']<1e-8
    groups=[]
    for (region,sensor),g in d.groupby(['region','sensor']):
        groups.append(dict(region=region,sensor=sensor,n_cays=len(g),n_reefs=g.reef_id.nunique(),n_observations=int(g.shape_observations.sum()),median_C=float(g.median_C.median()),median_E=float(g.median_E.median()),median_delta_C=float(g.delta_C.median()),median_delta_E=float(g.delta_E.median()),C_direction_preserved=int(g.C_direction_preserved_1px.sum()),E_direction_preserved=int(g.E_direction_preserved_1px.sum()),C_invalid_anchor=int((~g.anchor_C_in_range).sum()),overlap=int(g.anchor_overlap.sum()),median_minor_pixels=float(g.minimum_minor_pixels.median())))
    save(pd.DataFrame(groups),'地区与来源支持')
    correlations=[]
    for sensor,g in d.groupby('sensor'):
        for shape in ['delta_C','delta_E']:
            for other in ['delta_log_area','centroid_net_normalized','centroid_envelope_normalized','endpoint_exchange_fraction']:
                z=g[[shape,other]].dropna();rho=float(spearmanr(z[shape].abs(),z[other].abs()).statistic)
                correlations.append(dict(sensor=sensor,shape_change=shape,comparison=other,n_cays=len(z),rho_absolute_magnitudes=rho,scope='descriptive; no p-value, adjusted association or coupling claim'))
    save(pd.DataFrame(correlations),'形状与既有变化的描述性重叠')
    usable=d[~d.anchor_overlap & d.anchor_C_in_range & d.invalid_observations.eq(0)]
    summary=dict(primary=len(d),n_observations=int(d.shape_observations.sum()),early_late_masks=len(tr),source_counts=d.groupby('sensor').size().to_dict(),original_metric_reproduction_max_abs_error=reproduction,C_gt1_selected_observations=int(d.C_gt1_observations.sum()),C_gt1_cays=int(d.C_gt1_observations.gt(0).sum()),invalid_observations=int(d.invalid_observations.sum()),anchor_overlap=int(d.anchor_overlap.sum()),C_anchor_out_of_range=int((~d.anchor_C_in_range).sum()),usable_nonoverlap_C=len(usable),C_preserved_in_usable=int(usable.C_direction_preserved_1px.sum()),C_preserved_both_perimeters=int((usable.C_direction_preserved_1px & usable.C_methods_same_direction).sum()),E_preserved_nonoverlap=int(d.loc[~d.anchor_overlap,'E_direction_preserved_1px'].sum()),season_matched_45d=int((~d.anchor_overlap & d.early_late_season_gap_days.le(45)).sum()),boundary_scenario='all combinations of -1/0/+1 pixel morphological boundary perturbations for observed early-two/late-three masks; not empirical uncertainty intervals; no registration, tide or colour calibration',classification='not attempted; no independent or physical thresholds supplied; low C does not uniquely identify curvature',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256((ROOT/'outputs/沙洲观测主表.csv').read_bytes()).hexdigest())
    summary['all_anchor_pairs_matched_45d']=int((~d.anchor_overlap & d.maximum_cross_window_season_gap_days.le(45)).sum())
    summary['all_anchor_pairs_matched_90d']=int((~d.anchor_overlap & d.maximum_cross_window_season_gap_days.le(90)).sum())
    summary['compactness_scope']='C columns below reproduce legacy stored definition; current annual C uses primary Crofton, see P0 audit'
    summary['seasonal_scope']='early_late_season_gap_days compares circular mean phase only; it is not full image-wise seasonal matching'
    (OUT/'形状核查.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    plot(d)
    outline_cases(d)
    report(d,tr,summary)
    seasonal_control(d)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def plot(d):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(9.2,3.0),layout='constrained')
    for sensor,color,marker in [('sentinel2','#22536D','o'),('google_earth','#BC7137','^')]:
        g=d[d.sensor.eq(sensor)];axes[0].scatter(g.median_E,g.median_C,color=color,marker=marker,s=15,alpha=.7,label=f'{sensor}; n={len(g)}')
        for ax,key in zip(axes[1:],['C','E']):
            ax.scatter(g.delta_log_area,g['delta_'+key],color=color,marker=marker,s=15,alpha=.7)
            z=g[~g.anchor_overlap & (g.anchor_C_in_range if key=='C' else True) & g[key+'_direction_preserved_1px']]
            ax.scatter(z.delta_log_area,z['delta_'+key],facecolors='none',edgecolors='black',s=35,linewidth=.7)
    axes[0].set(xlabel='Median elongation (L/W)',ylabel='Median compactness (4πA/P²)');axes[0].legend(fontsize=6,frameon=False)
    for ax,key in zip(axes[1:],['C','E']):ax.axhline(0,color='#888888',lw=.7);ax.axvline(0,color='#888888',lw=.7);ax.set(xlabel='Long-term log area ratio',ylabel=f'Late − early {key}')
    fig.suptitle('Candidate only: continuous shape; rings retain direction under ±1-pixel boundary scenarios',fontsize=8)
    for ax,label in zip(axes,'abc'):ax.text(-.13,1.02,label,transform=ax.transAxes,fontweight='bold')
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('形状候选图.'+ext),dpi=300)
    plt.close(fig)

def outline_cases(d):
    candidates=d[~d.anchor_overlap & d.E_direction_preserved_1px & d.early_late_season_gap_days.le(45) & d.minimum_minor_pixels.ge(20)]
    chosen=candidates.assign(abs_change=candidates.delta_E.abs()).sort_values(['sensor','abs_change'],ascending=[True,False]).groupby('sensor').head(2)
    save(chosen,'轮廓候选案例及选择条件')
    fig,axes=plt.subplots(1,len(chosen),figsize=(9.2,2.8),layout='constrained')
    for ax,p in zip(np.atleast_1d(axes),chosen.itertuples()):
        g=OBS[(OBS.sand_cay_id==p.sand_cay_id)&(OBS.sensor==p.sensor)&(OBS.reference_frame_id==p.reference_frame_id)].sort_values('date').reset_index(drop=True)
        radius=np.sqrt(p.start_area_m2/np.pi)
        for i in [0,1,len(g)-3,len(g)-2,len(g)-1]:
            x=g.iloc[i];mask=read_mask(x.resolved_mask_path);y,z=np.where(mask);cx=z.mean();cy=y.mean()
            cs,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                xy=c[:,0,:];xy=np.vstack([xy,xy[0]]);ax.plot((xy[:,0]-cx)*x.pixel_size_m/radius,-(xy[:,1]-cy)*x.pixel_size_m/radius,color='#22536D' if i<2 else '#BC7137',alpha=.55,lw=.7)
        ax.set_aspect('equal');ax.set(xlabel='Centroid-centred E-W (r)',ylabel='Centroid-centred N-S (r)',title=p.sand_cay_id+'\n'+p.sensor)
        ax.text(.03,.04,f'E: {p.E_early:.2f} → {p.E_late:.2f}',transform=ax.transAxes,fontsize=7)
    fig.suptitle('Real masks: early two (blue), late three (orange); each outline recentered independently',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('真实轮廓候选图.'+ext),dpi=300)
    plt.close(fig)

def report(d,tr,s):
    support=[]
    for sensor,g in d.groupby('sensor'):
        support.append(f"| {sensor} | {len(g)} | {int(g.shape_observations.sum())} | {int((~g.anchor_overlap).sum())} | {int((~g.anchor_overlap & g.E_direction_preserved_1px).sum())} | {int((~g.anchor_overlap & g.early_late_season_gap_days.le(45)).sum())} |")
    text=f'''# 形状指标审计与局部整合

结论：不建立面积—形状综合指数，不建立“紧凑—狭长—弯曲/不规则”三分类。建议形状沿连续指标作有限补充，优先伸长率；紧致度在正式作为新结果前须解决周长与面积对象口径。当前为B档有限整合，不达到C档主图替换门槛。方法2.2、结果3.2和讨论4.1局部补充ΔE，补充S8承接完整结果；标题、摘要和主图保持本次整合前版本。

## 指标究竟表示什么

面积A回答大小，紧致度C=4πA/P²回答轮廓相对于面积的紧凑程度，伸长率E=L/W回答轴向比例。C含面积不代表同时保留绝对大小：按比例放大的相似轮廓具有相同C/E。使用(A,C,E)可解释坐标，不任意加权。高E和低C可以同时出现；低C既可能来自狭长、凹陷、曲折，也可能来自边界像素和碎片，因此不能直接等同“弯曲型”。凸实度可补充凹陷信息，但当前没有独立证据要求追加该指标，更不能把凹陷唯一解释为弯曲。

## 数据和实际算法

继承当前93条主序列，包含{s['n_observations']}次A/B观测：S2 56条、GE 37条。同源、同沙洲、同框、同日去重。完整93条均有有限正值C/E，无缺失补值。取最初2期和末3期分别求中位数，得到ΔC、ΔE及长期log面积比；3条仅4次观测，窗口共享一景，标记后不进入不重叠窗口支持计数。

462个早晚唯一日期的真实掩膜逐景复算精确复现存储值：C最大误差{s['original_metric_reproduction_max_abs_error']['C']:.2g}，E最大误差{s['original_metric_reproduction_max_abs_error']['E']:.2g}。E的长短轴可由最大连通域二阶矩等效椭圆复现，不是外接矩形长宽。P可由OpenCV原二值掩膜全部外轮廓长度之和复现；A和轴却对应最大连通域。9景、8个沙洲有多个连通域，主连通域周长对照与原C最大差为{float((tr.stored_C-tr.C_primary).abs().max()):.3f}。原始特征生成脚本尚未在当前目录定位，这一算法结论来自逐景复现；不私自改写已有列。

2115次主序列观测中69次C>1，涉及9个沙洲；4条序列的早晚锚点含越界C，不能按理想连续轮廓阈值分类。像素外轮廓周长对小目标的偏差及碎片口径均须保留。Crofton周长仅作为算法敏感性，并非“真值”。算法定义参见[scikit-image区域属性](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)，像素周长估计差异参见[官方周长比较示例](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_perimeters.html)。

## 边界情景与长期变化

每个真实日期分别使用原项目十字结构元进行−1、0、+1像素形态学边界变化，穷举早2/晚3窗口的组合（共享日期使用同一扰动）。这是有限边界情景，不是实测误差置信区间，不涵盖潮位、季节、配准形变或所有判读偏差。平移本身不改变C/E，但影像变形、水位导致的可见轮廓和季节构型仍可影响它们。

| 来源 | 序列 | 观测 | 不重叠窗口 | ΔE保持方向（±1像素） | 早晚季节相位差≤45d |
|---|---|---|---|---|---|
{chr(10).join(support)}

90条不重叠窗口中53条ΔE在全部边界情景下仍保持方向；早晚相位差≤45d的23条中16条保持方向。这说明存在值得核查的几何变化，但不能等同53条已排除所有测量与季节误差。紧致度排除共享窗口和锚点越界后剩86条，45条在原口径边界情景下保持方向，39条同时通过主连通域Crofton口径的方向对照；混合周长口径未因此成为已校准分类指标。

原空间展示对MEDF_008/022的ΔE均未超出本边界情景的零变化范围，不能顺势给这对案例添加“不同形状路径”的强结论。NH_015虽然ΔE保持方向，已有配准筛查也不能被形状情景取代。

## 增量、分类和与覆盖状态的联系

两来源中C与E的沙洲级中位数Spearman相关分别约−0.86与−0.89；它们大量共享形状信息，不值得扩展成并列主发现。按来源描述|ΔE|与|长期log面积变化|的相关约0.30（GE）、0.26（S2），与净位移约0.63、0.34；这些有限相关说明指标不完全重复，但不是控制尺寸、季节后的独立增量证明。全部相关仅作描述，不提供新增P值或筛选显著结果。

没有预定的测量/物理分类阈值，也未验证稳定自然分组，因此不命名三个客观类别、不按三分位切分。现有图4已经支持几何指标的年度预测信息；它不能代替长期起止形变证据。形状与可见覆盖是否存在系统性联系尚未由本次审计解决，不能把不同沙洲的覆盖均值关联当作同沙洲状态变化响应；保留为需要同区间、同来源与误差控制的独立待检验问题，不扩展交叉分组或回归搜索。

## 最小整合决策

阶段1及候选结果复核后，已将连续ΔE按B档局部整合，不增正文形状类型或新主图。主图1–6及现有空间轨迹叙事保留。补充S8以一张形状图和数据表承接结果，3.2只写有量值、明确“边界情景保持方向”的短段；讨论4.1只补几何构型与质心平移是不同对象的一句。主图2下排是标准化区间的四区比较，93条早晚窗口ΔE口径不同，当前不替换g，不增加h/i。C需先确定主体轮廓周长口径、保留原字段并评估既有年度结果对口径的敏感性；不因本轮探索擅自撤销已验证年度结果。

候选图和完整可追溯表位于outputs/形状审查。真实轮廓候选按不重叠、ΔE保持方向、早晚相位差≤45d、最窄期短轴≥20像素筛选，每来源取|ΔE|最大的两例，仅用于说明指标；各轮廓独立移至自身质心，明确抑制平移，不能用来判断原位置变化。该20像素条件是展示筛查，不是证明可检测性的阈值。候选图未登记为现行补充图，避免与已交付图号混淆。

复算：运行论文分析.py --shape-audit。该选项只审查形状，不调用旧形状—质心回归，不重写Word或主图。原值、输入指纹、掩膜来源、全部情景和代码指纹均可追溯。
'''
    text += '\n## 平均季节相位与逐景匹配复核\n\n前述23条仅指窗口平均相位接近，不是同季影像完全匹配。要求早2与晚3的全部六个跨窗日期对相位差≤45d时，90条不重叠序列中无合格记录；放宽至90d仅1条。故53/90作为几何差异在边界情景下的方向稳定性报告，不能解释为已排除季节的长期趋势。S8将两种季节条件分开报告。原候选图仍留作审计，现行图仅采用S8，不作为新增分类或生态耦合结果。\n'
    text += '\n## 现行紧致度口径更新\n\n上述C复现、越界与候选图属于原字段审计。P0核查已统一现行紧致度为同一最大连通域面积与四方向Crofton周长，原字段保留。相同留出年份下原口径增益6.3%，主体外轮廓6.7%，主体Crofton约6.0%，三者均通过原七指标FDR。图4采用现行Crofton结果；定义与逐景来源及原口径对照见补充S3和outputs/P0证据核查。并未据此建立形状分类。\n'
    (ROOT/'shape_audit.md').write_text(text,encoding='utf-8')


def window_centroid_metrics(g):
    # 同一参考框内，以分量中位数构造代表位置；并非某一真实影像的质心。
    assert g.reference_frame_id.nunique()==1
    scale=g.pixel_size_m.to_numpy(dtype=float)
    assert np.allclose(scale,scale[0]), '同框尺度不一致，不能合并质心'
    xy=g[['sand_cay_centroid_x','sand_cay_centroid_y']].to_numpy(dtype=float)*scale[:,None]
    xy[:,1]*=-1
    early=np.median(xy[:2],axis=0);late=np.median(xy[-3:],axis=0)
    radius=float(np.sqrt(g.sand_cay_area_m2.iloc[:2].median()/np.pi))
    return dict(window_centroid_early_east_m=float(early[0]),window_centroid_early_north_m=float(early[1]),window_centroid_late_east_m=float(late[0]),window_centroid_late_north_m=float(late[1]),window_centroid_net_normalized=float(np.linalg.norm(late-early)/radius),window_initial_radius_m=radius)

def seasonal_control(original):
    candidates=[];rows=[];sources=[]
    for p in PRIMARY.itertuples(index=False):
        g=OBS[(OBS.sand_cay_id==p.sand_cay_id)&(OBS.sensor==p.sensor)&(OBS.reference_frame_id==p.reference_frame_id)].sort_values('date').reset_index(drop=True)
        eligible=[]
        for quarter in range(1,5):
            h=g[g.date.dt.quarter.eq(quarter)].reset_index(drop=True)
            if len(h)<5:
                candidates.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,quarter=quarter,n_observations=len(h),eligible=False,reason='fewer than five observations'))
                continue
            early=h.iloc[:2];late=h.iloc[-3:]
            gap=float((late.date.median()-early.date.median()).total_seconds()/86400/365.2425)
            day=h.date.dt.dayofyear.to_numpy(dtype=float)
            distances=np.abs(day[:2,None]-day[None,-3:]);distances=np.minimum(distances,365.2425-distances)
            maximum=float(distances.max());ok=gap>=3 and maximum<=90
            candidates.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,quarter=quarter,n_observations=len(h),window_separation_years=gap,maximum_cross_window_season_gap_days=maximum,eligible=ok,reason='eligible' if ok else 'span below three years or phase gap above 90 days'))
            if ok:eligible.append((gap,len(h),-quarter,quarter,h))
        if not eligible:continue
        gap,n,_,quarter,h=max(eligible,key=lambda x:x[:3])
        anchors=pd.concat([h.iloc[:2],h.iloc[-3:]],ignore_index=True)
        assert anchors.image_id.nunique()==5 and anchors.date.nunique()==5
        E=anchors.sand_cay_elongation;C=anchors.sand_cay_compactness
        row=dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,reference_frame_id=p.reference_frame_id,region=p.region,quarter=quarter,n_quarter_observations=n,window_separation_years=gap,early_dates='|'.join(anchors.date.iloc[:2].dt.strftime('%Y-%m-%d')),late_dates='|'.join(anchors.date.iloc[-3:].dt.strftime('%Y-%m-%d')),early_area_m2=float(anchors.sand_cay_area_m2.iloc[:2].median()),late_area_m2=float(anchors.sand_cay_area_m2.iloc[-3:].median()),E_early=float(E.iloc[:2].median()),E_late=float(E.iloc[-3:].median()),delta_E=float(E.iloc[-3:].median()-E.iloc[:2].median()),delta_C=float(C.iloc[-3:].median()-C.iloc[:2].median()),minimum_anchor_minor_pixels=float(anchors.sand_cay_minor_axis_length.min()))
        row['delta_log_area']=float(np.log(row['late_area_m2']/row['early_area_m2']))
        row.update(window_centroid_metrics(anchors))
        day=anchors.date.dt.dayofyear.to_numpy(dtype=float);distances=np.abs(day[:2,None]-day[None,-3:]);row['maximum_cross_window_season_gap_days']=float(np.minimum(distances,365.2425-distances).max())
        values=[]
        for i,x in anchors.iterrows():
            mask=cv2.imdecode(np.fromfile(x.resolved_mask_path,dtype='uint8'),cv2.IMREAD_GRAYSCALE)>0
            v=[metrics(perturb_mask(mask,k))['E'] for k in [-1,0,1]]
            assert np.isclose(v[1],x.sand_cay_elongation,atol=1e-10)
            values.append(v)
            sources.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,reference_frame_id=p.reference_frame_id,quarter=quarter,window='early' if i<2 else 'late',date=str(x.date.date()),image_id=x.image_id,mask_path=x.resolved_mask_path,pixel_size_m=x.pixel_size_m,quality_grade=x.quality_grade,E_original=v[1],E_eroded=v[0],E_dilated=v[2]))
        low,high=window_scenario_range({i:[{'E':z} for z in v] for i,v in enumerate(values)},[0,1],[2,3,4],'E')
        row['delta_E_scenario_min']=low;row['delta_E_scenario_max']=high;row['E_direction_preserved_1px']=bool(low>0 or high<0)
        old=original[(original.sand_cay_id==p.sand_cay_id)&(original.sensor==p.sensor)&(original.reference_frame_id==p.reference_frame_id)].iloc[0]
        row['original_delta_E']=float(old.delta_E);row['original_E_direction_preserved_1px']=bool(old.E_direction_preserved_1px);row['same_direction_as_original']=bool(np.sign(old.delta_E)==np.sign(row['delta_E']))
        rows.append(row)
    d=pd.DataFrame(rows);save(pd.DataFrame(candidates),'同季度候选窗口');save(d,'同季度早晚形状位置');save(pd.DataFrame(sources),'同季度逐景来源与边界情景')
    summary=dict(n_selected=len(d),n_source_images=len(sources),n_preserved=int(d.E_direction_preserved_1px.sum()),n_same_sign=int(d.same_direction_as_original.sum()),n_both_preserved=int((d.E_direction_preserved_1px & d.original_E_direction_preserved_1px).sum()),n_both_preserved_same_sign=int((d.E_direction_preserved_1px & d.original_E_direction_preserved_1px & d.same_direction_as_original).sum()),by_source={sensor:dict(n=len(g),preserved=int(g.E_direction_preserved_1px.sum()),same_sign=int(g.same_direction_as_original.sum()),median_window_separation_years=float(g.window_separation_years.median())) for sensor,g in d.groupby('sensor')},selection='calendar quarter; >=5 dates; early 2 / late 3; median time separation >=3 years; all six annual phase gaps <=90 days; choose longest median separation, then most quarter observations, then lowest quarter; no outcome selection',scope='coarse seasonal sensitivity comparison, not full tide, registration or environmental control; windows and time spans change')
    (OUT/'窗口与同季对照核查.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (ROOT/'shape_audit.md').open('a',encoding='utf-8') as f:
        f.write('\n## 窗口一致的位置与同季度敏感性对照\n\n早2/晚3质心分别按东西、南北分量取中位数，净距离用该窗口早期面积等效半径归一化；代表位置不是某景影像的真实质心。原首末单景净位移保留，不替换累计路径或活动包络。\n\n固定日历季度，每季度至少5景、早晚中位日期相隔至少3年、全部六个跨窗相位差≤90天，按时间跨度最大、观测数最多、季度号最小选择一组。选择不使用变化大小。共'+str(len(d))+'条、'+str(len(sources))+'景，'+str(summary['n_preserved'])+'条ΔE在边界情景下保持方向；与原窗口同符号'+str(summary['n_same_sign'])+'条；两套窗口均保持方向且同符号'+str(summary['n_both_preserved_same_sign'])+'条。完整来源分层见窗口与同季对照核查.json。季度对照同时改变日期与跨度，不能将不一致全部归因于季节，也不能以一致证明长期单调趋势。\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
