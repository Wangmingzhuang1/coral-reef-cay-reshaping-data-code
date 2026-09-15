"""SFTP 传输速度对照测试（前台运行）。"""

import os
import time

import paramiko

t0 = time.time()
transport = paramiko.Transport(("ftp-access.aviso.altimetry.fr", 2221))
transport.connect(
    username=os.environ["AVISO_USER"], password=os.environ["AVISO_PASS"]
)
sftp = paramiko.SFTPClient.from_transport(transport)
remote = "/auxiliary/tide_model/fes2022b/ocean_tide_extrapolated/mask_fes2022B.nc.xz"
handle = sftp.open(remote, "rb")
handle.prefetch()
total = 0
out = open(os.path.join(os.environ["TEMP"], "mask_sftp.xz"), "wb")
while True:
    chunk = handle.read(1 << 20)
    if not chunk:
        break
    total += len(chunk)
    out.write(chunk)
out.close()
print("bytes", total, "seconds", round(time.time() - t0, 1))
transport.close()
