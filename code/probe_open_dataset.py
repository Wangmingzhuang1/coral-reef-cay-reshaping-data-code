import copernicusmarine as cm
ds = cm.open_dataset(dataset_id='cmems_obs-ins_glo_phybgcwav_mynrt_na_irr')
vars_ = [str(v) for v in ds.data_vars]
lines = ['VARS ' + ','.join(sorted(vars_))]
lines.append('DIMS ' + str(dict(ds.sizes)))
open('outputs/tide_gauge_open_dataset_probe.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('open probe done')