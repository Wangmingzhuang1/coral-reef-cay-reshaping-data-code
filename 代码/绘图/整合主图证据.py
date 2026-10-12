"""重组现有已复算图源，不改变模型与校正族。"""

import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()
from pathlib import Path
import sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from 绘制当前结果图 import C,clean,tag,save
ROOT = PROJECT_ROOT;OUT=ROOT/'outputs';SRC=OUT/'figures/source_data'
sys.path.insert(0,str(ROOT/"代码/时间结构"))
from 分析裸沙质心轨迹周期与突发 import load_observations,build_steps,panel_tail
from 分析裸沙质心轨迹风浪对照 import load_forcing_inputs,build_usable,fit_forcing_models,panel_storm
def environment_figure(usable, models):
    main=pd.read_csv(OUT/'审稿问题复核/单一原始网格方向模型.csv')
    main=main[main.specification.eq('within_cay_fixed_effects')].copy()
    main=main[((main.outcome=='along_shift') & main.term.isin(['current_along_mean_z','wave_vector_along_mean_z'])) | ((main.outcome=='cross_shift') & main.term.isin(['current_cross_mean_z','wave_vector_cross_mean_z']))].copy()
    main=main.set_index(['outcome','term']).loc[[('along_shift','current_along_mean_z'),('along_shift','wave_vector_along_mean_z'),('cross_shift','current_cross_mean_z'),('cross_shift','wave_vector_cross_mean_z')]].reset_index()
    main.to_csv(SRC/'figure_3_directional_models.csv',index=False,encoding='utf-8-sig')
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'width_ratios':[1,1.15]})
    y=np.arange(len(main))[::-1];a.errorbar(main.coefficient,y,xerr=np.vstack([main.coefficient-main.ci95_low,main.ci95_high-main.coefficient]),fmt='o',color=C['navy'],capsize=2,ms=4,lw=.9)
    a.axvline(0,color='grey',ls=':',lw=.7);a.set(yticks=y,yticklabels=['Along current','Along wave drift','Cross current','Cross wave drift'],xlabel='Standardized coefficient (95% CI)');a.set_title('Direction correspondence',fontsize=9);clean(a);tag(a,'a')
    for yy,r in zip(y,main.itertuples()):a.text(r.ci95_high+.025,yy,f'q={r.fdr_q_value:.3g}',fontsize=6,va='center')
    panel_storm(b,usable,models,'b  High-wave exposure and bare-cay mobility');fig.subplots_adjust(left=.16,right=.98,bottom=.22,top=.86,wspace=.72);save(fig,'图5_环境与移动性联系');plt.close(fig)


def coverage_figure(steps):
    fig,(a,b,c)=plt.subplots(1,3,figsize=(7.2,3.0),gridspec_kw={'width_ratios':[1.1,1,1]})
    cays=pd.read_csv(SRC/'figure5_vegetation_stability_cays.csv')
    cays=cays[~(cays.sensor.eq('sentinel2')&cays.sand_cay_id.isin(excluded))]
    for sensor,col,label in [('sentinel2',C['teal'],'S2'),('google_earth',C['orange'],'GE')]:
        d=cays[cays.sensor.eq(sensor)].dropna(subset=['vegetation_fraction_median','centroid_path_normalized'])
        assert d.centroid_path_normalized.gt(0).all()
        a.scatter(d.vegetation_fraction_median,d.centroid_path_normalized,s=15,alpha=.7,
                  label=f'{label}; n={len(d)}',color=col,edgecolor='white',linewidth=.3)
    cays.to_csv(SRC/'主图6_覆盖路径个体点.csv',index=False,encoding='utf-8-sig')
    rho=pd.read_csv(OUT/'审稿问题复核/单一原始网格覆盖路径秩相关.csv');rho=rho[rho.outcome.eq('centroid_path_normalized')]
    rho.to_csv(SRC/'主图6_覆盖与累计路径统计.csv',index=False,encoding='utf-8-sig')
    statistics=[]
    for sensor,label in [('sentinel2','S2'),('google_earth','GE')]:
        row=rho[rho.sensor.eq(sensor)].iloc[0];statistics.append(f'{label}: ρ = {row.spearman_rho:.3f}; q = {row.fdr_q_value:.3g}')
    a.text(.97,.96,'\n'.join(statistics),transform=a.transAxes,ha='right',va='top',fontsize=5.7)
    a.set(xlabel='Median green fraction',ylabel='Cumulative path (radii)',yscale='log');a.legend(frameon=False,fontsize=6.5,loc='lower left');a.set_title('Unadjusted between-cay association',fontsize=8);tag(a,'a')
    sector=pd.read_csv(OUT/'植被扇区对照统计.csv');sector=sector[sector.stratum.isin(['all','background'])];y=np.arange(len(sector))[::-1]
    b.errorbar(sector.mean_mobility_difference,y,xerr=np.vstack([sector.mean_mobility_difference-sector.bootstrap_ci_low,sector.bootstrap_ci_high-sector.mean_mobility_difference]),fmt='o',color=C['teal'],capsize=2,ms=4);b.axvline(0,color='grey',ls=':',lw=.7);b.set(yticks=y,yticklabels=[f'{label} (n={int(n)})' for label,n in zip(['All','Background'],sector.n_transitions)],xlabel='Covered minus low-cover mobility');b.set_title('Within-interval contrast',fontsize=9);tag(b,'b')
    panel_tail(c,steps,'c  Interval mobility distribution')
    tail=pd.read_csv(OUT/'审稿问题复核/单一原始网格高移动速率对照.csv').iloc[0]
    tail.to_frame().T.to_csv(SRC/'主图6_高移动速率占比.csv',index=False,encoding='utf-8-sig')
    steps[['reef_id','sand_cay_id','time_t','time_t1','stratum','speed_norm']].to_csv(SRC/'主图6_移动速率区间.csv',index=False,encoding='utf-8-sig')
    c.axvline(tail.pooled_p90_speed_radii_per_year,color=C['grey'],ls=':',lw=.8)
    c.text(.97,.98,f'Pooled p90 = {tail.pooled_p90_speed_radii_per_year:.2f} r/year\nBare: {100*tail.bare_median:.1f}% ({int(tail.n_bare_high_speed)}/{int(tail.n_bare_intervals)})\nCovered: {100*tail.vegetated_median:.1f}% ({int(tail.n_vegetated_high_speed)}/{int(tail.n_vegetated_intervals)})',transform=c.transAxes,ha='right',va='top',fontsize=5.6)
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.83,wspace=.67);save(fig,'图6_覆盖与移动性');plt.close(fig)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=['all','environment','coverage'],default='all')
    args=parser.parse_args()
    global excluded
    grids=pd.read_csv(OUT/'审稿问题复核/Sentinel2原始栅格网格核验.csv')
    excluded=set(grids.loc[grids.n_grids.gt(1),'sand_cay_id'])
    obs=load_observations();obs=obs[~obs.sand_cay_id.isin(excluded)]
    steps=build_steps(obs, grouping='cover')
    if args.part in ['all','environment']:
        forcing,daily=load_forcing_inputs()
        usable,_=build_usable(obs,steps,forcing,daily)
        models=fit_forcing_models(usable)
        environment_figure(usable,models)
    if args.part in ['all','coverage']:
        coverage_figure(steps)
    print('现行主图完成：'+args.part)

if __name__=='__main__': main()
