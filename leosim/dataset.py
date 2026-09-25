"""Result container and on-disk cache."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import SimConfig

_ARRAYS = ("positions", "altitudes", "distance", "visible", "capacity", "latency")


@dataclass
class LinkDataset:
    """Per-timestep link state, indexed [timestep, sat_i, sat_j].

    Attributes
    ----------
    distance : (T, N, N) float, km. Full matrix (symmetric, zero diagonal).
    visible  : (T, N, N) bool.  True iff the pair has line of sight and is in range.
    capacity : (T, N, N) float, bits/s. Zero where not visible.
    latency  : (T, N, N) float, seconds. Zero where not visible.
    positions: (T, N, 3) float, km, Earth-centred inertial (GCRS).
    altitudes: (T, N) float, km, WGS84 geodetic.
    """

    distance: np.ndarray
    visible: np.ndarray
    capacity: np.ndarray
    latency: np.ndarray
    positions: np.ndarray
    altitudes: np.ndarray

    sat_names: list[str]
    norad_ids: list[int]
    plane_index: list[int]
    times_utc_iso: list[str]
    time_offsets_s: np.ndarray

    config: SimConfig
    selection_info: dict

    @property
    def n_steps(self) -> int:
        return self.distance.shape[0]

    @property
    def n_sats(self) -> int:
        return self.distance.shape[1]

    def visible_pairs(self, t: int) -> np.ndarray:
        """(K, 2) array of i<j index pairs visible at timestep ``t``."""
        iu = np.triu_indices(self.n_sats, k=1)
        mask = self.visible[t][iu]
        return np.column_stack([iu[0][mask], iu[1][mask]])

    def degree(self) -> np.ndarray:
        """(T, N) count of visible neighbours per satellite per timestep."""
        return self.visible.sum(axis=2)

    def summary(self) -> str:
        deg = self.degree()
        vis_d = self.distance[self.visible]
        vis_c = self.capacity[self.visible]
        lines = [
            f"LinkDataset: {self.n_steps} steps x {self.n_sats} sats "
            f"({self.config.step_s:g}s step, {self.config.duration_s / 60:g} min)",
            f"  start (UTC)     : {self.times_utc_iso[0]}",
            f"  altitude km     : {self.altitudes.min():.1f} .. {self.altitudes.max():.1f}"
            f"  (geodetic WGS84)",
            f"  visible pairs   : {int(self.visible.sum() // 2)} total link-timesteps",
            f"  degree per sat  : min {deg.min()}, mean {deg.mean():.2f}, max {deg.max()}",
        ]
        if vis_d.size:
            lines += [
                f"  link range km   : {vis_d.min():.1f} .. {vis_d.max():.1f}",
                f"  capacity Mbps   : {vis_c.min() / 1e6:.1f} .. {vis_c.max() / 1e6:.1f}",
                f"  latency ms      : {self.latency[self.visible].min() * 1e3:.3f} .."
                f" {self.latency[self.visible].max() * 1e3:.3f}",
            ]
        return "\n".join(lines)

    # --- persistence ---------------------------------------------------------

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in _ARRAYS:
            np.save(directory / f"{name}.npy", getattr(self, name))
        np.save(directory / "time_offsets_s.npy", self.time_offsets_s)
        meta = {
            "sat_names": self.sat_names,
            "norad_ids": self.norad_ids,
            "plane_index": self.plane_index,
            "times_utc_iso": self.times_utc_iso,
            "config": asdict(self.config),
            "selection_info": self.selection_info,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, directory: Path, config: SimConfig) -> "LinkDataset":
        """Memory-map the arrays so a re-run does not pull ~200 MB resident."""
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        arrays = {
            name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in _ARRAYS
        }
        return cls(
            **arrays,
            time_offsets_s=np.load(directory / "time_offsets_s.npy"),
            sat_names=meta["sat_names"],
            norad_ids=meta["norad_ids"],
            plane_index=meta["plane_index"],
            times_utc_iso=meta["times_utc_iso"],
            config=config,
            selection_info=meta["selection_info"],
        )

    @classmethod
    def is_cached(cls, directory: Path) -> bool:
        directory = Path(directory)
        needed = [f"{n}.npy" for n in _ARRAYS] + ["meta.json", "time_offsets_s.npy"]
        return directory.is_dir() and all((directory / f).exists() for f in needed)
