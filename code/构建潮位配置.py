"""解压 FES2022b 分潮文件并生成 PyFES 配置 ocean_tide.yaml。

长周期分潮由 PyFES 平衡潮（compute_long_period_equilibrium）补齐，
因此配置只包含已下载的短周期/浅水分潮。
"""

from __future__ import annotations

import argparse
import lzma
from pathlib import Path


REQUIRED = ["m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1"]
OPTIONAL = ["m4", "ms4", "mn4", "n4", "s4", "m6", "m8", "mks2"]


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tide-dir",
        type=Path,
        default=root / "data" / "environment" / "tide" / "fes2022b" / "ocean_tide_extrapolated",
    )
    args = parser.parse_args()
    args.tide_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED if not (args.tide_dir / f"{name}_fes2022.nc.xz").is_file()]
    if missing:
        raise SystemExit(f"missing required archives: {missing}")
    constituents = REQUIRED + [
        name for name in OPTIONAL if (args.tide_dir / f"{name}_fes2022.nc.xz").is_file()
    ]
    extracted = []
    for name in constituents:
        archive = args.tide_dir / f"{name}_fes2022.nc.xz"
        target = args.tide_dir / f"{name}_fes2022.nc"
        if not archive.is_file():
            raise SystemExit(f"missing archive: {archive}")
        if not target.is_file() or target.stat().st_size == 0:
            with lzma.open(archive, "rb") as source, open(target, "wb") as sink:
                while chunk := source.read(1 << 26):
                    sink.write(chunk)
        extracted.append(target)
    lines = ["engine: darwin", "tide:", "  cartesian:", "    longitude: longitude", "    latitude: latitude", "    amplitude: amplitude", "    phase: phase", "    paths:"]
    for name, target in zip(constituents, extracted):
        lines.append(f"      {name}: {target.as_posix()}")
    config_path = args.tide_dir.parent / "ocean_tide.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"config: {config_path}")
    print(f"constituents: {len(extracted)}")


if __name__ == "__main__":
    main()
