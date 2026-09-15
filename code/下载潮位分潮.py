"""AVISO FES2022b 分潮下载（SFTP 断点续传 + 重试）。

凭据从环境变量 AVISO_USER / AVISO_PASS 读取（由启动命令注入，不写入文件）。
用法：python 下载潮位分潮.py --files m2 s2 --dest <dir> --tag W1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import paramiko

HOST = "ftp-access.aviso.altimetry.fr"
PORT = 2221
REMOTE_DIR = "/auxiliary/tide_model/fes2022b/ocean_tide_extrapolated"
MAX_ATTEMPTS = 12


def log(tag: str, message: str, log_path: Path) -> None:
    line = f"{time.strftime('%H:%M:%S')} {tag} {message}"
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def download_one(ftp_path: str, out: Path, tag: str, log_path: Path) -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = out.stat().st_size if out.is_file() else 0
        try:
            transport = paramiko.Transport((HOST, PORT))
            transport.connect(
                username=os.environ["AVISO_USER"],
                password=os.environ["AVISO_PASS"],
            )
            try:
                sftp = paramiko.SFTPClient.from_transport(transport)
                size = sftp.stat(ftp_path).st_size
                if offset >= size:
                    log(tag, f"complete {out.name} {offset}", log_path)
                    return True
                with sftp.open(ftp_path, "rb") as source:
                    source.prefetch()
                    if offset:
                        source.seek(offset)
                    mode = "ab" if offset else "wb"
                    with open(out, mode) as sink:
                        while True:
                            chunk = source.read(1 << 20)
                            if not chunk:
                                break
                            sink.write(chunk)
            finally:
                transport.close()
            actual = out.stat().st_size
            if actual != size:
                log(tag, f"size_mismatch {out.name} {actual}/{size}", log_path)
                continue
            log(tag, f"complete {out.name} {actual}", log_path)
            return True
        except Exception as exc:  # noqa: BLE001 - resume loop
            log(tag, f"attempt {attempt} failed {out.name}: {exc}", log_path)
            time.sleep(5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--tag", default="W")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().with_name("fes_download_py.log"),
    )
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in args.files:
        ftp_path = f"{REMOTE_DIR}/{name}_fes2022.nc.xz"
        out = args.dest / f"{name}_fes2022.nc.xz"
        if not download_one(ftp_path, out, args.tag, args.log):
            ok = False
    with open(args.log, "a", encoding="utf-8") as handle:
        handle.write(f"WORKER_DONE {args.tag} {'ok' if ok else 'failed'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
