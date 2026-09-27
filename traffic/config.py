"""Stage-4 configuration: offered load, gateways and queues.

Same convention as the other three stages: every tunable lives here, fields
marked ASSUMED were not pinned by the base paper or the secondary one, and a
real modelling choice is a flag stamped into ``variant_label``.

None of this exists in the base paper. DoTD's metrics are all load-independent
(project doc, section 3), so a demand-aware extension cannot be measured on
them at all. Stage 4 supplies the axes it needs: delivered throughput, delay
under load, and loss. The traffic model follows the shape Krishnan et al.
(arXiv 2601.13662) describe -- population density modulated by time of day,
feeding Poisson arrivals, with queues at the satellites -- but every number
below is ours; none was read from that paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Which way the modelled traffic flows between a user city and its gateway.
# Internet access traffic is dominated by the download direction, so the
# default loads gateway -> satellites -> city. ASSUMED: neither paper pins it,
# and ISL capacities are per direction, so the choice decides which queues fill.
TrafficDirection = Literal["to_user", "to_gateway"]

# Where the gateways are. Neither is a real operator's gateway map.
# "populous20" -- the largest listed city of each of the `n_gateways` most
#                 populous countries. OURS. Limitation: it puts gateways where
#                 the demand is, so most demand has a gateway close by and
#                 egress is easier than for a real operator, whose gateways
#                 follow licensing and backhaul rather than population.
# "krishnan8"  -- the eight sites named in Fig. 1 of Krishnan et al. (arXiv
#                 2601.13662): LA, London, Dubai, Tokyo, Singapore, Sao Paulo,
#                 Johannesburg, Sydney. Sourced, but only as names on a map.
GatewaySites = Literal["populous20", "krishnan8"]


@dataclass(frozen=True)
class TrafficConfig:
    # --- demand --------------------------------------------------------------
    cities_path: str = "data/cities.csv"

    # Aggregate offered load, summed over every city, at the diurnal peak. The
    # absolute level is a knob, not a reading: results are reported as curves
    # against it rather than at one value. ASSUMED default. On the 187-satellite
    # window the gateway feeder links are the bottleneck and already exceed
    # capacity at 5 Gbps, so 2 Gbps sits in the loaded-but-not-collapsed range.
    peak_total_gbps: float = 2.0

    # Diurnal shape: raised cosine in local SOLAR time (UTC + longitude/15),
    # peak at `diurnal_peak_hour`, floor at `diurnal_trough_fraction` of peak.
    # ASSUMED: evening peak and a 5:1 peak-to-trough ratio are the usual shape
    # for residential access networks; not taken from either paper.
    diurnal_peak_hour: float = 21.0
    diurnal_trough_fraction: float = 0.2

    # Flows arrive as a Poisson process per city; each carries this many bits.
    # Sets the granularity of the randomness only: the expected load is
    # independent of it. ASSUMED: 1 MB.
    mean_flow_bits: float = 8.0e6

    seed: int = 7

    # --- gateways ------------------------------------------------------------
    # Gateways sit at the largest listed city of each of the `n_gateways`
    # countries with the most listed population. ASSUMED: real gateway sites
    # are not published in a usable form. One per country spreads them the way
    # a real operator's are spread; top-N cities would put 7 in China.
    n_gateways: int = 20
    gateway_sites: GatewaySites = "populous20"
    gateways_krishnan_path: str = "data/gateways_krishnan.csv"

    # Gateway (feeder) link budget. None = use the stage-1 link budget with the
    # base paper's ground gain (Table I), i.e. the same 12.2 GHz / 100 MHz
    # budget as a user terminal -- the Phase 2 baseline as first built.
    # Real gateway links run in other bands with far wider bandwidth, so that
    # default understates feeder capacity; setting these tests how much the
    # "gateways are the bottleneck" result depends on it. Krishnan et al. model
    # this link separately (their B vs B^ISL) but give no values, and neither
    # does the base paper.
    #
    # Sourced value for feeder_bandwidth_hz: 2.1e9. Starlink gateway uplinks
    # (gateway -> satellite, the direction "to_user" traffic loads) use Ka-band
    # 27.5-29.1 + 29.5-30.0 GHz = 2.1 GHz, per the FCC gateway authorisations
    # (Banning CA; Lynnwood WA STA, 2025) as quoted in search results -- the FCC
    # documents themselves could not be opened from here. Pachler et al. 2021
    # (MIT, "An Updated Comparison of Four LEO Constellation Systems", from the
    # FCC filings as of Jan 2021) corroborates the 27.5-30.0 GHz envelope and
    # says each satellite "can use the full frequency range in both
    # polarizations" to a gateway, up to four satellites per gateway at once.
    # Their "2.1 GHz uplink / 1.3 GHz downlink" sentence describes OneWeb, NOT
    # Starlink; do not cite it for Starlink.
    #
    # Only bandwidth is overridden. Carrier stays 12.2 GHz and ground gain
    # 33.2 dBi, so a 2.1 GHz feeder is a hybrid: a real Ka link has ~7 dB more
    # path loss (28 vs 12.2 GHz) but a much larger antenna gain (1.5 m dish),
    # which on balance raises SNR. The bandwidth-only case therefore likely
    # UNDERSTATES a real feeder, and the 4-satellite-per-gateway limit is not
    # applied (unlimited here). Transmit power and noise temperature: still the
    # stage-1 ASSUMED values; no source found.
    feeder_bandwidth_hz: float | None = None
    feeder_tx_power_dbw: float | None = None
    feeder_noise_temperature_k: float | None = None

    direction: TrafficDirection = "to_user"

    # --- queues --------------------------------------------------------------
    # Buffer per directed link, as seconds of that link's capacity. Arrivals
    # beyond it are dropped. ASSUMED: 50 ms is a common rule-of-thumb buffer.
    buffer_s: float = 0.05

    def __post_init__(self) -> None:
        if self.peak_total_gbps < 0:
            raise ValueError("peak_total_gbps must be >= 0")
        if not 0.0 <= self.diurnal_trough_fraction <= 1.0:
            raise ValueError("diurnal_trough_fraction must be in [0, 1]")
        if self.mean_flow_bits <= 0 or self.buffer_s <= 0:
            raise ValueError("mean_flow_bits and buffer_s must be > 0")
        if self.n_gateways < 1:
            raise ValueError("n_gateways must be >= 1")

    @property
    def variant_label(self) -> str:
        sites = (f"populous{self.n_gateways}" if self.gateway_sites == "populous20"
                 else self.gateway_sites)
        def show(v, unit):
            return "stage1" if v is None else f"{v:g}{unit}"
        feeder = "stage1" if self.feeder_is_default else (
            f"B={show(self.feeder_bandwidth_hz, 'Hz')},P={show(self.feeder_tx_power_dbw, 'dBW')},"
            f"T={show(self.feeder_noise_temperature_k, 'K')}"
        )
        return (
            f"load={self.peak_total_gbps:g}Gbps, direction={self.direction}, "
            f"gateways={sites}, feeder={feeder}, buffer={self.buffer_s * 1e3:g}ms"
        )

    @property
    def feeder_is_default(self) -> bool:
        return (self.feeder_bandwidth_hz is None and self.feeder_tx_power_dbw is None
                and self.feeder_noise_temperature_k is None)
