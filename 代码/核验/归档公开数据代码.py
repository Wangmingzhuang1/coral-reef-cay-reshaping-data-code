"""归档当前公开代码与派生表；不打包文稿、审稿记录或第三方影像。"""
from pathlib import Path
import csv,hashlib,importlib.metadata,io,json,re,sys,zipfile

ROOT=Path(__file__).resolve().parents[2]
REPOSITORY='https://github.com/Wangmingzhuang1/coral-reef-cay-reshaping-data-code'
VERSION='v2.0.1'
TITLE='珊瑚礁沙洲时空演变的年度重复性与植被相关移动性差异'

def public_payload():
    files=set(ROOT.glob('代码/**/*.py'))|set(ROOT.glob('tests/**/*.py'))
    files.update(ROOT.glob('outputs/**/*.csv'))
    files.update(p for p in ROOT.glob('outputs/**/*.json')
                 if '引用核验' not in p.parts and p.name not in
                 {'当前图件索引.json','方法同步核查.json','公开归档核验.json'})
    files.update(p for p in (ROOT/'额外分析_空间时间覆盖联合规律').glob('*')
                 if p.suffix in {'.py','.csv','.json'})
    files.update(ROOT/p for p in ['运行论文分析.py','分析排除清单.csv'])
    payload={};changed=set()
    for p in sorted(files):
        name=p.relative_to(ROOT).as_posix();old=p.read_bytes()
        content=old.decode('utf-8-sig')
        for prefix,replacement in [(str(ROOT)+'\\',''),(ROOT.as_posix()+'/',''),
                                   (str(ROOT.parent)+'\\','../'),(ROOT.parent.as_posix()+'/','../')]:
            content=content.replace(prefix,replacement).replace(prefix.replace('\\','\\\\'),replacement)
        content=re.sub(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\r\n",]+[\\/]+','{USER_HOME}/',content)
        if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{30,})',content):
            raise ValueError('疑似凭据，停止公开归档：'+name)
        data=content.encode('utf-8-sig' if p.suffix=='.csv' else 'utf-8')
        if data!=old:changed.add(name)
        payload[name]=data
    versions={}
    for pkg in ['numpy','pandas','scipy','statsmodels','rasterio','opencv-python','scikit-image',
                'matplotlib','python-docx','Pillow','PyMuPDF']:
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:continue
    payload['依赖版本.json']=json.dumps({'python':sys.version,'packages':versions},ensure_ascii=False,indent=2).encode('utf-8')
    payload['requirements.txt']='\n'.join(k+'=='+v for k,v in versions.items()).encode('utf-8')+b'\n'
    payload['DATA_LICENSE']='Derived measurement tables and source data: CC BY 4.0.\nhttps://creativecommons.org/licenses/by/4.0/\nCode: MIT; see LICENSE.\nThird-party imagery and masks are not included.\n'.encode('utf-8')
    payload['归档元数据.json']=json.dumps({'title':TITLE,'version':VERSION,'date':'2026-10-12',
        'repository':REPOSITORY,'release':REPOSITORY+'/releases/tag/'+VERSION,
        'code_license':'MIT','derived_tables_license':'CC-BY-4.0',
        'contents':['analysis code','derived measurement and statistical tables','figure source data'],
        'excluded':['manuscripts','reviewer correspondence','third-party original imagery','per-date masks'],
        'doi':None},ensure_ascii=False,indent=2).encode('utf-8')
    payload['README.md']=f'''# {TITLE}

This repository archives the current analysis code, derived measurement/statistical tables,
and figure source data for a multi-temporal study of coral-reef sand cays.

Current archive: **{VERSION}, 12 October 2026**. Repository: {REPOSITORY}
Versioned record: {REPOSITORY}/releases/tag/{VERSION}

## Scope and primary results

- Long-term area analysis: 93 records, including 63 near-stable, 17 net-gain and 13 net-loss records.
- Comparable position analysis: 90 records, including 60 near-stable records.
- Annual geometry reconstruction: 52 cays; five position/shape indicators show corrected support.
- Environmental direction models: 760 intervals from 46 cays.
- Direction sensitivity checks use the same eligibility: 760 intervals for 90–365 d and ±1 pixel,
  373 intervals for 120–365 d, and 46 leave-one-cay-out fits. The 60–365 d window has the
  same effective sample as 90–365 d. All four directions remain positive; corrected support
  varies with the window and boundary setting.
- Vegetation–path correlations: 53 Sentinel-2 and 37 Google Earth records, analysed separately.
- Within-interval shoreline contrasts: 1,012 comparisons from 62 cays.
- High-speed distribution under the accepted initial-cover labels: 521 bare intervals (94 high-speed)
  and 889 vegetated intervals (47 high-speed); 18.0% versus 5.3%. The threshold is the pooled p90.
- Bare-interval high-wave model: 497 intervals from 34 cays, with reef-clustered uncertainty.

Earlier exploratory analyses and sensitivity results are retained as supporting records.
The current main-result tables are mapped below; older English figure-number filenames reflect
earlier figure arrangements and do not define the current main-figure numbering.

## Files and main-figure mapping

| Current figure / analysis | Current source tables |
| --- | --- |
| Fig. 1: area outcomes and actual examples | `outputs/figures/source_data/表A_沙洲级面积路径状态.csv`; `四区图谱位置面积序列.csv` in the same source-data folder |
| Fig. 2: position and boundary differences | `outputs/审稿问题复核/原始网格一致长期位置描述.csv`; `outputs/figures/source_data/主图2_连续空间轨迹.csv` |
| Fig. 3: annual geometry | `outputs/审稿问题复核/单一原始网格留一年汇总.csv`; `outputs/审稿问题复核/单一原始网格留一年逐景对照.csv`; `outputs/figures/source_data/figure_2_harmonic_relative_error_reduction.csv` |
| Fig. 4: wave/current direction and exposure | `outputs/审稿问题复核/单一原始网格方向模型.csv`; `单一原始网格高波浪日模型.csv`; `覆盖分组风浪分析时段.csv` in the same folder |
| Fig. 5: observed cover-state changes | `outputs/figures/source_data/六向状态转换与保持.csv`; `表面状态相邻观测对.csv` in the same source-data folder |
| Fig. 6: vegetation and mobility | `outputs/figures/source_data/主图6_覆盖与累计路径统计.csv`; `outputs/植被扇区对照.csv`; `outputs/植被扇区对照统计.csv`; `outputs/审稿问题复核/单一原始网格高移动速率对照.csv` |

`代码/` contains Python modules grouped by scientific task. `outputs/` contains source measurements,
derived tables and all retained statistical comparisons. `额外分析_空间时间覆盖联合规律/` contains
the secondary joint-analysis code and results. CSV files use UTF-8 with BOM for Windows Excel.
`文件清单.csv` records every packaged file's SHA-256 and flags path-sanitized copies.

## Reproduce checks from the public archive

Download the release asset `data-code.zip` and extract it into a new directory. Open that directory
as the working directory, install Python 3.11 or later and the packages in `requirements.txt`, then run:

```text
python 代码/控制验证/复核审稿主要问题.py --check-derived
```

This command checks the past-only training dates, two seven-response BH families, 1,490 state pairs,
90 position-comparable records, shoreline-contrast eligibility, the accepted vegetation grouping,
the high-speed threshold/counts, the high-wave model family, the current direction sensitivity
samples and figure-source agreement, and every manifest SHA-256.
The files can also be read directly on GitHub; use the ZIP asset when byte-for-byte checks are required.

## Observation definitions and access to original inputs

Coordinates are image-based planform positions; geometry units and normalizations are named in
the table columns and defined by the analysis functions. `sand_cay_id`, `reef_id`, `sensor`,
`reference_frame_id`, dates and observation IDs link the measurement and comparison tables.
Green fraction describes visible vegetation inferred from rendered RGB images. Accepted initial
cover `none` defines bare intervals; `sparse`, `partial` and `dominant` define vegetated intervals.
This definition differs from the historical `stage` grouping retained for older exploratory analyses.

The public archive contains derived-data checks; recomputing masks, spectral diagnostics,
background alignment or the full measurement pipeline additionally requires original imagery,
per-date masks, annotation metadata and environmental input records. Those files are not in this archive.
The source-data layout expected by the scripts is `../data/source_dataset` relative to the project root.
Manuscript-generation and local delivery-audit scripts also require local manuscript/frozen-archive files.

Original imagery and environment sources:
- Sentinel-2: [Copernicus Data Space](https://dataspace.copernicus.eu/).
- Google Earth historical imagery: accessed through Google Earth under its applicable terms.
- ERA5: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
- GLORYS12V1 and WAVERYS: [Copernicus Marine Service](https://marine.copernicus.eu/).
- IBTrACS: [NOAA NCEI](https://www.ncei.noaa.gov/products/international-best-track-archive).

No third-party original imagery, per-date masks, manuscript files or reviewer correspondence are published.
Personal absolute paths in packaged copies are replaced with relative paths or `{{USER_HOME}}`;
the authoritative local measurement files remain unchanged.

## Licences and citation

Code: MIT (`LICENSE`). Author-derived tables: CC BY 4.0 (`DATA_LICENSE`).
Third-party data remain subject to their original providers' terms.

To cite this archive, include the study title above, the version {VERSION}, the date 12 October 2026,
and {REPOSITORY}/releases/tag/{VERSION}. No DOI is assigned to this release.
This release supersedes the previous repository layout; prior code/data remain recoverable in GitHub history.
'''.encode('utf-8')
    # 确认README所标识的现行表确实在包内；不能以旧图号推定文件。
    for required in ['outputs/审稿问题复核/单一原始网格留一年汇总.csv',
                     'outputs/审稿问题复核/单一原始网格方向模型.csv']:
        if required not in payload:raise FileNotFoundError(required)
    return payload,changed

def main():
    payload,changed=public_payload()
    stage=ROOT/'public_release';stage.mkdir(exist_ok=True)
    # 许可沿用现有公开仓库；不得凭空改换许可或作者名单。
    license_path=stage/'LICENSE'
    if not license_path.exists():raise FileNotFoundError('先取得现有GitHub LICENSE')
    payload['LICENSE']=license_path.read_bytes()
    for n,b in payload.items():
        dest=stage/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    # 清除上轮不再纳入的公开副本，归档根以当前清单为准。
    for old in stage.rglob('*'):
        if old.is_file() and old.relative_to(stage).as_posix() not in payload and old.name!='文件清单.csv':
            assert stage.resolve() in old.resolve().parents
            old.unlink()
    rows=io.StringIO(newline='');writer=csv.writer(rows)
    writer.writerow(['relative_path','size_bytes','sha256','packaged_copy_changed'])
    for n,b in sorted(payload.items()):writer.writerow([n,len(b),hashlib.sha256(b).hexdigest(),n in changed])
    payload['文件清单.csv']=rows.getvalue().encode('utf-8-sig')
    (stage/'文件清单.csv').write_bytes(payload['文件清单.csv'])
    archive=ROOT/'公开数据代码.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in sorted(payload.items()):z.writestr(n,b)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in csv.DictReader(io.StringIO(z.read('文件清单.csv').decode('utf-8-sig'))):
            b=z.read(r['relative_path']);assert len(b)==int(r['size_bytes'])
            assert hashlib.sha256(b).hexdigest()==r['sha256']
        assert not any(n.endswith(('.docx','.png','.pdf','.svg','.tiff')) for n in z.namelist())
    print(json.dumps({'files':len(payload),'archive':str(archive),'size_mb':round(archive.stat().st_size/1024**2,2),
        'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'all_manifest_hashes_verified':True},ensure_ascii=False))

if __name__=='__main__':main()
