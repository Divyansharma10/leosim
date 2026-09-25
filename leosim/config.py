"""Configuration dataclasses.

Every physical constant and tunable in the simulator lives here. Nothing in the
compute modules should contain a bare numeric literal for a physical quantity.

Fields marked ASSUMED were not specified in the original requirements; the value
is a documented default, not a measurement. Change them here, not inline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

# --- Universal constants (not tunables) --------------------------------------
SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_J_K = 1.380649e-23

SelectionMethod = Literal["planes", "first", "random"]


@lru_cache(maxsize=1)
def source_digest() -> str:
    """SHA-256 over the source of every module in this package.

    The cache key has to cover the *code* as well as the config and the TLE
    file. Config and TLE alone leave a real hole: edit ``geometry.py``, re-run,
    and the fingerprint is unchanged, so the pre-edit arrays are served from
    disk and the edit appears to have done nothing. That failure is silent and
    it hides exactly the changes most worth checking.

    Files are hashed in sorted order by POSIX-style relative path, with the path
    itself folded in so that renaming a module also changes the digest. Computed
    once per process.
    """
    package_dir = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(
        package_dir.rglob("*.py"), key=lambda p: p.relative_to(package_dir).as_posix()
    ):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(package_dir).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


@dataclass(frozen=True)
class LinkBudgetConfig:
    """Free-space path loss link budget for an RF inter-satellite link.

    No atmospheric attenuation term: ISLs propagate in vacuum. This models an RF
    ISL at Ku band; real Starlink ISLs are optical, but the requested parameter
    set is RF.
    """

    carrier_hz: float = 12.2e9
    bandwidth_hz: float = 100e6

    gain_tx_dbi: float = 40.0
    gain_rx_dbi: float = 40.0

    polarization_loss_db: float = 4.5
    misalignment_loss_db: float = 0.5

    # ASSUMED: not in the original spec. 10 dBW = 10 W.
    tx_power_dbw: float = 10.0

    # ASSUMED: not in the original spec. Typical Ku-band receive chain system
    # noise temperature used in Starlink link-budget literature.
    noise_temperature_k: float = 354.81

    @property
    def total_fixed_loss_db(self) -> float:
        return self.polarization_loss_db + self.misalignment_loss_db

    @property
    def total_gain_dbi(self) -> float:
        return self.gain_tx_dbi + self.gain_rx_dbi


@dataclass(frozen=True)
class GeometryConfig:
    """Earth occlusion model and range limit.

    A spherical Earth is used for the occlusion test (conservative, and what the
    requirements specify). Geodetic WGS84 altitude is used for *reporting and
    validation* instead, because |r| - equatorial_radius understates true
    altitude by up to 21 km at the poles.
    """

    earth_radius_km: float = 6378.0
    atmosphere_km: float = 50.0
    max_range_km: float = 7000.0

    @property
    def blocking_radius_km(self) -> float:
        """Radius the line-of-sight segment must clear."""
        return self.earth_radius_km + self.atmosphere_km


@dataclass(frozen=True)
class SelectionConfig:
    """How to choose the satellite subset from the TLE file.

    The default 'planes' method builds an ``n_planes x sats_per_plane`` grid so
    the subset has both cross-plane and in-plane structure. A pure one-per-plane
    spread would contain no in-plane pairs at all.

    Planes are identified by RAAN gap, so a plane here is a real orbital plane
    and ``sats_per_plane`` really is that plane's sampled population. Forcing a
    requested number of RAAN clusters instead merges ~7 real planes into each
    and thins every one of them, which silently breaks +Grid -- see
    :func:`leosim.tle.group_into_planes`.

    ``altitude_band_km`` matters: as of 2026 the median Starlink mean altitude is
    ~471 km, not 550 km, and the catalog has a decaying tail below 350 km. The
    band filter keeps the subset a coherent operational shell.
    """

    method: SelectionMethod = "planes"
    n_sats: int = 200

    # Both mean real orbital planes. ``n_planes`` planes consecutive in RAAN
    # are chosen from the shell and each is filled with ``sats_per_plane``
    # members spread by mean anomaly, so in-plane neighbours really are
    # adjacent in phase. n_sats should equal the product.
    n_planes: int = 10
    sats_per_plane: int = 20

    # RAAN separation above which two satellites are in different planes.
    # Planes on the 70 degree shell are ~10 deg apart and ~0.4 deg wide, so
    # anything in roughly 1..8 groups identically. Mirrors
    # topology.grid.PLANE_GAP_DEG, which applies the same rule to recovered
    # orbit normals as an independent cross-check.
    plane_gap_deg: float = 3.0

    altitude_band_km: tuple[float, float] = (500.0, 600.0)

    # None => auto-select the inclination shell with the most satellites inside
    # the altitude band. Set explicitly (e.g. 53.2) to pin a specific shell.
    inclination_deg: float | None = None
    inclination_tol_deg: float = 0.5

    # Reject near-decayed or badly eccentric objects.
    max_eccentricity: float = 0.01

    seed: int = 42


@dataclass(frozen=True)
class SimConfig:
    """Top-level simulation configuration."""

    tle_path: str = "data/starlink.tle"

    duration_s: float = 1800.0  # 30 minutes
    step_s: float = 1.0

    # None => median TLE epoch of the selected satellites. SGP4 error grows
    # ~1-3 km/day away from epoch, and catalog epochs span several days, so the
    # median bounds worst-case propagation error.
    start_utc: str | None = None

    link_budget: LinkBudgetConfig = field(default_factory=LinkBudgetConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    # Verify each candidate's propagated geodetic altitude against
    # SelectionConfig.altitude_band_km before selecting, so the band is a
    # guarantee the validation suite can assert. Costs ~1s.
    verify_altitude_band: bool = True

    cache_dir: str = "cache"

    # float32 resolves a 5000 km distance to well under a metre.
    float_dtype: str = "float32"

    # Timesteps processed per block. Peak geometry memory is
    # ~time_chunk * n_sats^2 * 3 * 4 bytes for the difference tensor.
    time_chunk: int = 200

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s / self.step_s))

    def fingerprint(self, tle_bytes: bytes) -> str:
        """Cache key over the full config, the TLE file contents *and* the code.

        Any change to any of the three produces a new key, so a stale cache
        cannot be silently reused. Including the source digest is what makes
        that true for edits to the compute modules themselves -- see
        :func:`source_digest`.
        """
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        h = hashlib.sha256()
        h.update(payload)
        h.update(hashlib.sha256(tle_bytes).digest())
        h.update(bytes.fromhex(source_digest()))
        return h.hexdigest()[:16]

    def resolve(self, root: Path) -> "SimConfig":
        """Return a copy with relative paths resolved against ``root``."""
        import dataclasses

        changes = {}
        if not Path(self.tle_path).is_absolute():
            changes["tle_path"] = str((root / self.tle_path).resolve())
        if not Path(self.cache_dir).is_absolute():
            changes["cache_dir"] = str((root / self.cache_dir).resolve())
        return dataclasses.replace(self, **changes) if changes else self
