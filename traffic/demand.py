"""Where the demand is, where it exits, and how much arrives each timestep.

Demand is raw population with no market filtering. ``data/README.md`` records
the resulting skew -- GeoNames gives several Chinese cities their municipal
population, so China holds 72 of the 200 rows -- and it is kept deliberately as
a property of the model, not corrected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from routing.stations import GroundStation

from .config import TrafficConfig

HOURS_PER_DAY = 24.0
DEG_PER_HOUR = 15.0


@dataclass(frozen=True)
class DemandSites:
    """User cities and gateways, as ground stations with populations."""

    cities: list[GroundStation]
    population: np.ndarray  # (C,)
    gateways: list[GroundStation]
    gateway_country: list[str]


def load_sites(cfg: TrafficConfig, root: Path) -> DemandSites:
    df = pd.read_csv(root / cfg.cities_path)
    cities = [
        GroundStation(str(r.name), float(r.lat), float(r.lon))
        for r in df.itertuples(index=False)
    ]

    if cfg.gateway_sites == "krishnan8":
        gw = pd.read_csv(root / cfg.gateways_krishnan_path)
        return DemandSites(
            cities=cities,
            population=df["population"].to_numpy(dtype=np.float64),
            gateways=[GroundStation(f"GW-{r.name}", float(r.lat), float(r.lon))
                      for r in gw.itertuples(index=False)],
            gateway_country=gw["country"].tolist(),
        )

    # One gateway per country, at its largest listed city, for the countries
    # with the most listed population. Ties broken by name so the choice is
    # deterministic.
    by_country = df.groupby("country")["population"].sum().sort_values(ascending=False, kind="stable")
    countries = list(by_country.index[: cfg.n_gateways])
    if len(countries) < cfg.n_gateways:
        raise ValueError(f"only {len(countries)} countries listed, need {cfg.n_gateways}")
    gateways = []
    for c in countries:
        row = df[df["country"] == c].sort_values(["population", "name"], ascending=[False, True]).iloc[0]
        gateways.append(GroundStation(f"GW-{row['name']}", float(row["lat"]), float(row["lon"])))

    return DemandSites(
        cities=cities,
        population=df["population"].to_numpy(dtype=np.float64),
        gateways=gateways,
        gateway_country=countries,
    )


def local_solar_hour(utc_hour: np.ndarray | float, longitude_deg: np.ndarray) -> np.ndarray:
    """Local mean solar time in hours, [0, 24)."""
    return np.mod(np.asarray(utc_hour)[..., None] + np.asarray(longitude_deg) / DEG_PER_HOUR, HOURS_PER_DAY)


def diurnal_factor(local_hour: np.ndarray, cfg: TrafficConfig) -> np.ndarray:
    """Raised cosine in [trough, 1], equal to 1 at the peak hour."""
    phase = 2.0 * np.pi * (local_hour - cfg.diurnal_peak_hour) / HOURS_PER_DAY
    floor = cfg.diurnal_trough_fraction
    return floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(phase))


def expected_rate_bps(sites: DemandSites, utc_hours: np.ndarray, cfg: TrafficConfig) -> np.ndarray:
    """(T, C) expected offered load per city in bits/s.

    Scaled so that if every city sat at its diurnal peak together the total
    would be ``peak_total_gbps``. They never do -- the peak moves round the
    globe -- so the realised total is lower and varies with the UTC hour.
    """
    share = sites.population / sites.population.sum()
    lon = np.array([c.longitude_deg for c in sites.cities])
    factor = diurnal_factor(local_solar_hour(utc_hours, lon), cfg)
    return cfg.peak_total_gbps * 1e9 * share[None, :] * factor


def offered_bits(rate_bps: np.ndarray, step_s: float, cfg: TrafficConfig) -> np.ndarray:
    """(T, C) Poisson flow arrivals per step, in bits.

    The flow count is Poisson with mean rate * step / mean_flow_bits, so the
    expected bits per step equal rate * step exactly, whatever the flow size.
    """
    rng = np.random.default_rng(cfg.seed)
    flows = rng.poisson(rate_bps * step_s / cfg.mean_flow_bits)
    return flows * cfg.mean_flow_bits
