"""SGP4 orbit propagation into an Earth-centred inertial frame."""

from __future__ import annotations

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

from .config import SimConfig
from .tle import TleRecord

# Built-in timescale: no network download, no ~/.skyfield cache required.
_TS = load.timescale(builtin=True)


def build_satellites(records: list[TleRecord]) -> list[EarthSatellite]:
    return [EarthSatellite(r.line1, r.line2, r.name, _TS) for r in records]


def resolve_start_time(records: list[TleRecord], cfg: SimConfig):
    """Simulation epoch: explicit config value, else median TLE epoch."""
    if cfg.start_utc is not None:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(cfg.start_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return _TS.from_datetime(dt)

    median_jd = float(np.median([r.epoch_jd for r in records]))
    return _TS.ut1_jd(median_jd)


def propagate(
    records: list[TleRecord], cfg: SimConfig
) -> tuple[np.ndarray, np.ndarray, object]:
    """Propagate all satellites across the simulation window.

    Returns
    -------
    positions : (T, N, 3) float array, km, Earth-centred inertial (GCRS/ICRF).
    altitudes : (T, N) float array, km, WGS84 *geodetic* altitude.
    times     : Skyfield Time array of length T.

    Frame note: SGP4 natively emits TEME; Skyfield applies the TEME->ICRF
    rotation, so these are GCRS coordinates. Every stage-1 quantity (pairwise
    distance, distance from Earth's centre to a chord) is rotation-invariant, so
    the choice between TEME and GCRS cannot change any output. It would matter
    for ground stations or sun vectors, which need ECEF/ITRF.
    """
    sats = build_satellites(records)
    t0 = resolve_start_time(records, cfg)

    offsets_s = np.arange(cfg.n_steps, dtype=np.float64) * cfg.step_s
    times = _TS.tt_jd(t0.tt + offsets_s / 86400.0)

    n_steps, n_sats = cfg.n_steps, len(sats)
    dtype = np.dtype(cfg.float_dtype)
    positions = np.empty((n_steps, n_sats, 3), dtype=dtype)
    altitudes = np.empty((n_steps, n_sats), dtype=dtype)

    for j, sat in enumerate(sats):
        geocentric = sat.at(times)  # vectorised over the whole time array
        positions[:, j, :] = geocentric.position.km.T.astype(dtype)
        altitudes[:, j] = wgs84.height_of(geocentric).km.astype(dtype)

    return positions, altitudes, times


def propagate_single(record: TleRecord, start_time, duration_s: float, step_s: float):
    """Propagate one satellite over an arbitrary window (used by validation)."""
    sat = EarthSatellite(record.line1, record.line2, record.name, _TS)
    offsets_s = np.arange(int(round(duration_s / step_s)), dtype=np.float64) * step_s
    times = _TS.tt_jd(start_time.tt + offsets_s / 86400.0)
    geocentric = sat.at(times)
    return geocentric.position.km.T, offsets_s, times


def refine_by_geodetic_altitude(
    records: list[TleRecord],
    start_time,
    band_km: tuple[float, float],
    probe_minutes: float = 100.0,
    probe_step_s: float = 60.0,
) -> list[TleRecord]:
    """Keep only satellites whose *actual* geodetic altitude stays inside the band.

    Filtering on Brouwer mean altitude alone is not enough. Instantaneous WGS84
    geodetic altitude departs from the mean by two effects that do not cancel:

      * J2 short-period radial oscillation, ~10-15 km peak-to-peak in LEO even
        at essentially zero eccentricity;
      * the geoid itself, whose geocentric radius is ~12 km smaller at 70 deg
        latitude than at the equator, raising geodetic altitude near apex.

    Probing over slightly more than one orbital period costs well under a second
    for a few thousand candidates and turns SelectionConfig.altitude_band_km
    into a guarantee the validation suite can assert against.
    """
    if not records:
        return records

    lo, hi = band_km
    offsets_s = np.arange(0.0, probe_minutes * 60.0, probe_step_s)
    times = _TS.tt_jd(start_time.tt + offsets_s / 86400.0)

    kept = []
    for rec in records:
        sat = EarthSatellite(rec.line1, rec.line2, rec.name, _TS)
        heights = wgs84.height_of(sat.at(times)).km
        if np.isfinite(heights).all() and heights.min() >= lo and heights.max() <= hi:
            kept.append(rec)
    return kept
