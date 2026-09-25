"""CLI entry point: build the stage-1 link dataset and print a summary."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from leosim.config import SelectionConfig, SimConfig

ROOT = Path(__file__).parent


def build_config(args) -> SimConfig:
    selection = SelectionConfig(
        method=args.method,
        n_sats=args.n_sats,
        n_planes=args.planes,
        sats_per_plane=args.per_plane,
        altitude_band_km=(args.alt_min, args.alt_max),
        inclination_deg=args.inclination,
        seed=args.seed,
    )
    cfg = SimConfig(
        tle_path=args.tle,
        duration_s=args.duration_min * 60.0,
        step_s=args.step,
        start_utc=args.start,
        selection=selection,
    )
    return cfg.resolve(ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tle", default="data/starlink.tle")
    p.add_argument("--duration-min", type=float, default=30.0)
    p.add_argument("--step", type=float, default=1.0, help="seconds")
    p.add_argument("--start", default=None, help="ISO UTC; default = median TLE epoch")
    p.add_argument("--method", default="planes", choices=["planes", "first", "random"])
    p.add_argument("--n-sats", type=int, default=100)
    p.add_argument("--planes", type=int, default=20)
    p.add_argument("--per-plane", type=int, default=5)
    p.add_argument("--alt-min", type=float, default=500.0)
    p.add_argument("--alt-max", type=float, default=600.0)
    p.add_argument("--inclination", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    from leosim.simulation import run

    dataset = run(build_config(args), use_cache=not args.no_cache)
    print()
    print(dataset.summary())


if __name__ == "__main__":
    main()
