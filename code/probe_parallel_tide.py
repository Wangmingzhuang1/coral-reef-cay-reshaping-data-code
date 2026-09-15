import time
from concurrent.futures import ThreadPoolExecutor
import copernicusmarine as cm

REEFS = [
    ("OTHER_008", 11.553051, 165.248617),
    ("MEDF_053", 2.314732, 72.982990),
    ("GBR_020", -13.44, 143.98),
    ("MEDF_006", 5.862948, 73.207655),
]

def one(reef):
    name, lat, lon = reef
    t0 = time.time()
    try:
        df = cm.read_dataframe(
            dataset_id="cmems_obs-ins_glo_phy-ssh_my_na_PT1H",
            variables=["SLEV"],
            start_datetime="2025-06-01",
            end_datetime="2025-06-08",
            minimum_latitude=lat - 2.0,
            maximum_latitude=lat + 2.0,
            minimum_longitude=lon - 2.0,
            maximum_longitude=lon + 2.0,
            disable_progress_bar=True,
        )
        return f"{name} rows={len(df)} secs={round(time.time()-t0,1)}"
    except Exception as exc:
        return f"{name} failed {type(exc).__name__} {str(exc)[:120]} secs={round(time.time()-t0,1)}"

t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(one, REEFS))
open("outputs/tide_gauge_parallel_test.txt", "w", encoding="utf-8").write("\n".join(results) + f"\ntotal_secs={round(time.time()-t0,1)}\n")
print("parallel test done")