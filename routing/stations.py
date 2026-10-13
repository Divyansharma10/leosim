"""Ground stations: where they are, and which satellites they can reach.

This is the first part of the project that is not rotation-invariant. Every
stage-1 quantity is a distance between two orbiting bodies, unchanged by any
rotation about Earth's centre, which is why the TEME/GCRS distinction could be
ignored there. A ground station rotates *with* the Earth, so its position in the
inertial frame depends on the Earth rotation angle at that instant and the frame
now matters. Skyfield is asked for the station's GCRS position at each timestep,
which puts it in the same frame as ``dataset.positions``.

The local vertical is likewise taken from Skyfield rather than assumed radial:
on an oblate Earth the geodetic normal is not parallel to the position vector
except at the equator and the poles, and the difference reaches ~0.19 deg at
45 deg latitude -- the same order as the elevation mask's own effect on how many
satellites a city can see.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from skyfield.api import load, wgs84

from leosim.dataset import LinkDataset
from leosim.linkbudget import capacity_bps, latency_s

from .config import RoutingConfig

_TS = load.timescale(builtin=True)

# Offset used to recover the local geodetic vertical, km. The direction is
# scale-free; 1 km is large enough to be exact in float64 and small enough that
# the two points share a timestep.
_UP_PROBE_KM = 1.0


@dataclass(frozen=True)
class GroundStation:
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0


# The five city pairs of the base paper's Table II, S1 to S5, plus two extras
# used only as validation probes.
CITIES = {
    "New York": GroundStation("New York", 40.7128, -74.0060),
    "San Francisco": GroundStation("San Francisco", 37.7749, -122.4194),
    "Sydney": GroundStation("Sydney", -33.8688, 151.2093),
    "Darwin": GroundStation("Darwin", -12.4634, 130.8456),
    "Miami": GroundStation("Miami", 25.7617, -80.1918),
    "Calgary": GroundStation("Calgary", 51.0447, -114.0719),
    "Phnom Penh": GroundStation("Phnom Penh", 11.5564, 104.9282),
    "Kathmandu": GroundStation("Kathmandu", 27.7172, 85.3240),
    "London": GroundStation("London", 51.5074, -0.1278),
    "Singapore": GroundStation("Singapore", 1.3521, 103.8198),
}

# Table II's pair labels, in the paper's order.
PAPER_PAIRS = [
    ("S1", "Sydney", "Darwin"),
    ("S2", "Miami", "Calgary"),
    ("S3", "New York", "Miami"),
    ("S4", "New York", "San Francisco"),
    ("S5", "Phnom Penh", "Kathmandu"),
]


def great_circle_km(a: GroundStation, b: GroundStation, radius_km: float = 6371.0) -> float:
    """Ground distance along the sphere. The fibre-route reference, not a bound."""
    p1, p2 = np.radians([a.latitude_deg, a.longitude_deg]), np.radians(
        [b.latitude_deg, b.longitude_deg]
    )
    d = 2.0 * np.arcsin(
        np.sqrt(
            np.sin((p2[0] - p1[0]) / 2) ** 2
            + np.cos(p1[0]) * np.cos(p2[0]) * np.sin((p2[1] - p1[1]) / 2) ** 2
        )
    )
    return float(radius_km * d)


def chord_km(a: GroundStation, b: GroundStation, radius_km: float = 6371.0) -> float:
    """Straight-line distance through the Earth between two surface points.

    This, not the great-circle arc, is the rigorous lower bound on the length of
    any path between them: the triangle inequality gives |A..B| >= |A - B| for
    any route whatsoever, while the arc is *longer* than the chord and so cannot
    bound a path from below. The distinction is not academic -- a long enough
    space path legitimately beats the ground arc traversed at c, which is the
    mechanism behind the base paper's "faster than fibre" claim. Asserting the
    arc as a floor would eventually assert that claim is impossible.
    """
    p1, p2 = np.radians([a.latitude_deg, a.longitude_deg]), np.radians(
        [b.latitude_deg, b.longitude_deg]
    )
    d = 2.0 * np.arcsin(
        np.sqrt(
            np.sin((p2[0] - p1[0]) / 2) ** 2
            + np.cos(p1[0]) * np.cos(p2[0]) * np.sin((p2[1] - p1[1]) / 2) ** 2
        )
    )
    return float(2.0 * radius_km * np.sin(d / 2))


@lru_cache(maxsize=None)
def geocentric_xyz_km(station: GroundStation) -> tuple[float, float, float]:
    """Earth-fixed position on the WGS84 ellipsoid, km. Constant in time."""
    site = wgs84.latlon(station.latitude_deg, station.longitude_deg, station.elevation_m)
    return tuple(float(v) for v in site.itrs_xyz.km)


def geodesic_floor_km(a: GroundStation, b: GroundStation) -> float:
    """Arc between a and b on the sphere of radius min(|a|, |b|): the denominator
    of the stretch factor, and a rigorous lower bound on any routed path.

    Why this and not ``great_circle_km`` or ``chord_km``. The triangle inequality
    bounds a path only by the chord. The arc bound comes from elsewhere: radial
    projection onto a sphere of radius r never lengthens a curve lying outside
    it, so any path that stays outside the r-sphere is at least the r-arc
    between the projected endpoints. Every routed path does stay outside a
    sphere of radius min(|a|, |b|): ISLs clear Earth plus 50 km by stage-1
    visibility, and an access link leaves its station above the elevation mask,
    so its distance from the centre grows from the station outward.

    Two details keep the bound exact rather than approximately true. The radius
    is the stations' own geocentric radius -- New York sits at ~6369 km, inside
    the 6371 km sphere ``great_circle_km`` assumes, so that arc would not bound
    the path. And the angle is between the stations' geocentric VECTORS:
    ``great_circle_km`` treats geodetic latitude as spherical, which is a
    different angle on an oblate Earth.
    """
    va, vb = np.asarray(geocentric_xyz_km(a)), np.asarray(geocentric_xyz_km(b))
    angle = np.arctan2(np.linalg.norm(np.cross(va, vb)), float(va @ vb))
    return float(min(np.linalg.norm(va), np.linalg.norm(vb)) * angle)


def dataset_times(dataset: LinkDataset):
    """Skyfield Time array matching the dataset's timesteps.

    Rebuilt from the stored start instant and offsets rather than carried
    through the cache, so it cannot drift out of step with ``time_offsets_s``.
    """
    start = datetime.fromisoformat(dataset.times_utc_iso[0].replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    t0 = _TS.from_datetime(start)
    offsets = np.asarray(dataset.time_offsets_s, dtype=np.float64)
    return _TS.tt_jd(t0.tt + offsets / 86400.0)


def station_geometry(station: GroundStation, times) -> tuple[np.ndarray, np.ndarray]:
    """(T, 3) GCRS position in km and (T, 3) unit geodetic vertical."""
    site = wgs84.latlon(
        station.latitude_deg, station.longitude_deg, station.elevation_m
    )
    above = wgs84.latlon(
        station.latitude_deg,
        station.longitude_deg,
        station.elevation_m + _UP_PROBE_KM * 1000.0,
    )
    pos = site.at(times).position.km.T
    up = above.at(times).position.km.T - pos
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    return pos, up


@dataclass
class GroundAccess:
    """Per-timestep ground-satellite link state for one station.

    Arrays are indexed [timestep, satellite], matching stage 1's convention.
    ``capacity`` and ``latency`` are zero where the link does not exist, also
    matching stage 1.
    """

    station: GroundStation
    visible: np.ndarray  # (T, N) bool
    distance_km: np.ndarray  # (T, N)
    elevation_deg: np.ndarray  # (T, N)
    capacity: np.ndarray  # (T, N) bits/s
    latency: np.ndarray  # (T, N) seconds

    @property
    def n_visible(self) -> np.ndarray:
        """(T,) satellites reachable at each timestep."""
        return self.visible.sum(axis=1)


def ground_link_budget(dataset: LinkDataset, cfg: RoutingConfig):
    """Stage-1 link budget with the ground antenna gain on the receive end.

    Everything else -- carrier, bandwidth, transmit power, losses, noise
    temperature -- is the satellite configuration unchanged, so a ground link
    differs from an ISL only in gain and distance and the two stay comparable.
    """
    return dataclasses.replace(
        dataset.config.link_budget, gain_rx_dbi=cfg.ground_gain_dbi
    )


def compute_access(
    dataset: LinkDataset, station: GroundStation, cfg: RoutingConfig, times=None
) -> GroundAccess:
    """Which satellites ``station`` can reach at each timestep, and how well.

    A satellite counts as reachable when it is above ``min_elevation_deg`` of
    the station's local horizon. No Earth-occlusion test is needed: positive
    elevation already means the line of sight clears the local horizon, and the
    mask is stricter than that.
    """
    if times is None:
        times = dataset_times(dataset)
    gs_pos, gs_up = station_geometry(station, times)

    sat_pos = np.asarray(dataset.positions, dtype=np.float64)  # (T, N, 3)
    los = sat_pos - gs_pos[:, None, :]
    distance = np.linalg.norm(los, axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        sin_elev = np.einsum("tnc,tc->tn", los, gs_up) / distance
    elevation = np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))

    visible = elevation >= cfg.min_elevation_deg

    if cfg.max_access_links is not None:
        # Keep the highest-elevation links; ties broken by index, which is
        # deterministic. argsort descending on elevation, masked to visible.
        ranked = np.argsort(-np.where(visible, elevation, -np.inf), axis=1)
        keep = np.zeros_like(visible)
        rows = np.arange(visible.shape[0])[:, None]
        chosen = ranked[:, : cfg.max_access_links]
        keep[rows, chosen] = True
        visible = visible & keep

    budget = ground_link_budget(dataset, cfg)
    safe = np.where(visible, distance, 1.0)
    capacity = np.where(visible, capacity_bps(safe, budget), 0.0)
    latency = np.where(visible, latency_s(safe), 0.0)

    return GroundAccess(
        station=station,
        visible=visible,
        distance_km=distance,
        elevation_deg=elevation,
        capacity=capacity,
        latency=latency,
    )
