"""Stage-1 validation suite.

Every check prints PASS/FAIL with the measured number and the bound it was
compared against, so a failure tells you how far off it was rather than just
that it happened.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4.api import Satrec

from leosim.config import SimConfig
from leosim.geometry import max_geometric_range_km, pairwise_geometry, segment_min_radius
from leosim.linkbudget import capacity_bps, free_space_path_loss_db, latency_s, snr_db
from leosim.propagate import propagate_single, resolve_start_time
from leosim.simulation import run
from leosim.tle import load_tles

ROOT = Path(__file__).parent
SPEED_OF_LIGHT_M_S = 299_792_458.0
EARTH_MU_KM3_S2 = 398600.4418
EARTH_EQUATORIAL_KM = 6378.137

# --------------------------------------------------------------------------
# Pinned absolute link-budget values at a reference distance.
#
# Every other link-budget check is a *relative* one -- monotonicity, or one
# array agreeing with another. Relative checks cannot see an error that moves
# every value by the same factor, so these pin three absolute numbers.
#
# The literals below were computed by hand from the closed forms, at the
# documented default parameters, and are hard-coded rather than recomputed at
# runtime. A reference the suite recomputes from the same inputs each run would
# track the production code if the production code drifted; a literal does not.
#
# They depend on the ASSUMED tx power and noise temperature, so the two that
# do are asserted only while the config still holds its documented defaults
# (see pins_apply below) -- changing those values in config.py is supported and
# must not read as a validation failure.
# --------------------------------------------------------------------------
PIN_DISTANCE_KM = 1000.0
PIN_FSPL_DB = 174.17497983537834  # 20log10(1e6 m) + 20log10(12.2e9) + 20log10(4pi/c)
PIN_SNR_DB = 33.92422882253281  # Pt+G-FSPL-L - 10log10(k*T*B)
PIN_CAPACITY_BPS = 1126996921.9720628  # B*log2(1+10^(SNR/10))
PIN_LATENCY_S = 0.0033356409519815205  # 1e6 m / c
PIN_REL_TOL = 1e-9

# Config values the capacity and SNR pins were computed at.
PIN_CONFIG = {
    "carrier_hz": 12.2e9,
    "bandwidth_hz": 100e6,
    "gain_tx_dbi": 40.0,
    "gain_rx_dbi": 40.0,
    "polarization_loss_db": 4.5,
    "misalignment_loss_db": 0.5,
    "tx_power_dbw": 10.0,
    "noise_temperature_k": 354.81,
}


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds for the validation suite.

    Deliberately separate from SimConfig: these are test tolerances, not
    physical parameters, and keeping them out of SimConfig avoids invalidating
    the simulation cache whenever a threshold is tuned.
    """

    # Agreement required between the empirically propagated period and the
    # period Kepler's third law predicts from the propagated orbital radius.
    period_kepler_tolerance: float = 0.01

    # The original spec asked for a 90-95 min period, but that is inconsistent
    # with the 500-600 km altitude band it was paired with: Kepler ties the two
    # together, and 500-600 km implies 94.6-96.7 min. A 550 km Starlink orbits
    # in 95.65 min. The band actually asserted is therefore DERIVED from the
    # configured altitude band, so the two constraints cannot contradict each
    # other; the spec band is reported alongside for reference.
    spec_period_band_min: tuple[float, float] = (90.0, 95.0)
    period_band_margin_min: float = 0.5

    in_plane_cv_max: float = 0.05
    capacity_step_tolerance_bps: float = 1e4


def kepler_period_min(radius_km: float) -> float:
    """Circular-orbit period from geocentric radius, via Kepler's third law."""
    return 2.0 * np.pi * np.sqrt(radius_km**3 / EARTH_MU_KM3_S2) / 60.0


def kepler_altitude_km(period_min: float) -> float:
    """Inverse of :func:`kepler_period_min`: altitude implied by a period."""
    a = (EARTH_MU_KM3_S2 * (period_min * 60.0 / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    return a - EARTH_EQUATORIAL_KM


def period_band_from_altitudes(
    band_km: tuple[float, float], margin_min: float
) -> tuple[float, float]:
    """Period band implied by an altitude band, plus a small margin for J2."""
    lo, hi = band_km
    return (
        kepler_period_min(EARTH_EQUATORIAL_KM + lo) - margin_min,
        kepler_period_min(EARTH_EQUATORIAL_KM + hi) + margin_min,
    )


class Checks:
    """Minimal pass/fail harness."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str) -> bool:
        self.results.append((name, bool(ok), detail))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        for line in detail.splitlines():
            print(f"         {line}")
        return bool(ok)

    def between(self, name: str, value, lo, hi, unit="", extra="") -> bool:
        ok = lo <= value <= hi
        detail = f"measured {value:.6g}{unit}, required {lo:g}..{hi:g}{unit}"
        if extra:
            detail += "\n" + extra
        return self.record(name, ok, detail)

    def close_to(self, name: str, value, expected, rel_tol, unit="", extra="") -> bool:
        """Assert ``value`` matches a pinned ``expected`` to a relative tolerance."""
        rel = abs(value - expected) / abs(expected) if expected else abs(value)
        detail = (
            f"measured {value:.10g}{unit}, pinned {expected:.10g}{unit}, "
            f"relative error {rel:.2e} (tolerance {rel_tol:.0e})"
        )
        if extra:
            detail += "\n" + extra
        return self.record(name, rel <= rel_tol, detail)

    def summarise(self) -> int:
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        print()
        print("=" * 72)
        print(f"  {passed}/{total} checks passed")
        if passed != total:
            print("  failed:")
            for name, ok, _ in self.results:
                if not ok:
                    print(f"    - {name}")
        print("=" * 72)
        return 0 if passed == total else 1


def section(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 68 - len(title)))


def tle_period_min(record) -> float:
    """Orbital period in minutes from the TLE Kozai mean motion."""
    return 2.0 * np.pi / Satrec.twoline2rv(record.line1, record.line2).no_kozai


# --------------------------------------------------------------------------
# 1. Orbital period
# --------------------------------------------------------------------------
def check_orbital_period(chk: Checks, cfg: SimConfig, dataset, vcfg: ValidationConfig) -> None:
    section("1. Orbital period")

    band = period_band_from_altitudes(
        cfg.selection.altitude_band_km, vcfg.period_band_margin_min
    )
    spec_lo, spec_hi = vcfg.spec_period_band_min
    alt_lo, alt_hi = cfg.selection.altitude_band_km
    print(
        f"         note: the asserted band {band[0]:.2f}-{band[1]:.2f} min is derived\n"
        f"         from the {alt_lo:g}-{alt_hi:g} km altitude band via Kepler's third law.\n"
        f"         The original {spec_lo:g}-{spec_hi:g} min spec band implies "
        f"{kepler_altitude_km(spec_lo):.0f}-{kepler_altitude_km(spec_hi):.0f} km\n"
        f"         and is therefore incompatible with it above "
        f"{kepler_altitude_km(spec_hi):.0f} km.\n"
    )

    catalog = {r.norad_id: r for r in load_tles(cfg.tle_path)}
    chosen = [catalog[i] for i in dataset.norad_ids]

    periods = np.array([tle_period_min(r) for r in chosen])
    chk.between(
        "TLE mean-motion period, fastest of 100 sats",
        float(periods.min()),
        band[0],
        band[1],
        " min",
        extra=(
            f"across all 100: min {periods.min():.3f}, mean {periods.mean():.3f}, "
            f"max {periods.max():.3f} min"
        ),
    )
    chk.between(
        "TLE mean-motion period, slowest of 100 sats",
        float(periods.max()),
        band[0],
        band[1],
        " min",
    )

    # Empirical, from propagated geometry. The 30-minute simulation window is
    # shorter than one period, so this propagates its own longer window and
    # measures the rate of change of the argument of latitude.
    start = resolve_start_time(chosen, cfg)
    rec = chosen[0]
    pos, offsets_s, _ = propagate_single(rec, start, duration_s=7200.0, step_s=10.0)

    normal = np.cross(pos[0], pos[1])
    normal /= np.linalg.norm(normal)
    e1 = pos[0] / np.linalg.norm(pos[0])
    e2 = np.cross(normal, e1)
    theta = np.unwrap(np.arctan2(pos @ e2, pos @ e1))
    rate_rad_s = np.polyfit(offsets_s, theta, 1)[0]
    period_emp = (2.0 * np.pi / abs(rate_rad_s)) / 60.0

    reference = tle_period_min(rec)
    chk.between(
        f"Empirical propagated period ({rec.name})",
        float(period_emp),
        band[0],
        band[1],
        " min",
        extra=(
            f"measured by unwrapping argument of latitude over 120 min of SGP4\n"
            f"TLE mean-motion value {reference:.4f} min, difference "
            f"{abs(period_emp - reference) * 60:.2f} s"
        ),
    )

    # The strongest check in this section: the propagated period and the
    # propagated orbital radius must satisfy Kepler's third law independently.
    # This catches unit errors, wrong time scales and frame-origin mistakes that
    # any absolute band wide enough to be safe would let through.
    mean_radius = float(np.linalg.norm(pos, axis=1).mean())
    predicted = kepler_period_min(mean_radius)
    rel_err = abs(period_emp - predicted) / predicted
    chk.record(
        "Propagated period satisfies Kepler's third law at the propagated radius",
        rel_err <= vcfg.period_kepler_tolerance,
        f"empirical {period_emp:.4f} min vs Kepler {predicted:.4f} min "
        f"at mean radius {mean_radius:.1f} km\n"
        f"relative error {rel_err * 100:.4f}% "
        f"(tolerance {vcfg.period_kepler_tolerance * 100:g}%, "
        f"residual is the J2 nodal-vs-Keplerian difference)",
    )


# --------------------------------------------------------------------------
# 2. Altitude band
# --------------------------------------------------------------------------
def check_altitudes(chk: Checks, cfg: SimConfig, dataset) -> None:
    section("2. Altitude band")

    lo, hi = cfg.selection.altitude_band_km
    alt = np.asarray(dataset.altitudes)

    per_sat_min = alt.min(axis=0)
    per_sat_max = alt.max(axis=0)
    bad = np.where((per_sat_min < lo) | (per_sat_max > hi))[0]

    extra = (
        f"WGS84 geodetic altitude over {dataset.n_steps} steps x "
        f"{dataset.n_sats} sats, mean {alt.mean():.1f} km"
    )
    if bad.size:
        names = ", ".join(
            f"{dataset.sat_names[i]} ({per_sat_min[i]:.1f}-{per_sat_max[i]:.1f})"
            for i in bad[:5]
        )
        extra += f"\nout-of-band satellites ({bad.size}): {names}"

    chk.record(
        f"All geodetic altitudes within {lo:g}-{hi:g} km",
        bad.size == 0,
        f"observed {alt.min():.1f} .. {alt.max():.1f} km, "
        f"required {lo:g} .. {hi:g} km\n{extra}",
    )

    # Guards against a units or frame-origin error that a band check alone
    # would not catch if the band happened to be wide.
    radius = np.linalg.norm(np.asarray(dataset.positions, dtype=np.float64), axis=2)
    chk.between(
        "Geocentric radius consistent with a LEO shell",
        float(radius.mean()),
        6378.0 + lo - 30.0,
        6378.0 + hi + 30.0,
        " km",
        extra="catches metre/kilometre unit errors and wrong frame origins",
    )


# --------------------------------------------------------------------------
# 3. Symmetry
# --------------------------------------------------------------------------
def check_symmetry(chk: Checks, dataset) -> None:
    section("3. Matrix symmetry")

    vis = np.asarray(dataset.visible)
    dist = np.asarray(dataset.distance)

    asym = int((vis != np.transpose(vis, (0, 2, 1))).sum())
    chk.record(
        "Visibility matrix symmetric at every timestep",
        asym == 0,
        f"asymmetric entries: {asym} of {vis.size}",
    )

    dist_asym = float(np.abs(dist - np.transpose(dist, (0, 2, 1))).max())
    chk.record(
        "Distance matrix symmetric at every timestep",
        dist_asym == 0.0,
        f"max |D - D^T| = {dist_asym:.3e} km",
    )

    diag = np.arange(dataset.n_sats)
    self_vis = int(vis[:, diag, diag].sum())
    self_dist = float(np.abs(dist[:, diag, diag]).max())
    chk.record(
        "Diagonal is zero-distance and non-visible (no self-links)",
        self_vis == 0 and self_dist == 0.0,
        f"self-visible entries {self_vis}, max diagonal distance {self_dist:.3e} km",
    )


# --------------------------------------------------------------------------
# 4. Earth occlusion
# --------------------------------------------------------------------------
def check_antipodal(chk: Checks, cfg: SimConfig, dataset) -> None:
    section("4. Earth occlusion")

    geo = cfg.geometry
    r = geo.earth_radius_km + 550.0

    # Exact antipodal pair, driven through the production geometry code.
    synthetic = np.array([[[r, 0.0, 0.0], [-r, 0.0, 0.0]]], dtype=np.float64)
    d_syn, v_syn = pairwise_geometry(synthetic, geo)
    clearance = float(segment_min_radius(synthetic[0, 0], synthetic[0, 1]))
    verdict = "blocked" if clearance < geo.blocking_radius_km else "CLEAR (wrong)"
    chk.record(
        "Exact antipodal pair is NOT visible",
        not bool(v_syn[0, 0, 1]),
        f"separation {d_syn[0, 0, 1]:.1f} km, segment clearance from Earth centre "
        f"{clearance:.1f} km\nblocking radius {geo.blocking_radius_km:.1f} km "
        f"-> {verdict}",
    )

    # The other side of the boundary: a grazing pair that just clears the
    # atmosphere must be visible. Without this, a function that returned False
    # unconditionally would still pass the antipodal test.
    b = geo.blocking_radius_km + 1.0
    half = np.sqrt(r * r - b * b)
    graze = np.array([[[b, -half, 0.0], [b, half, 0.0]]], dtype=np.float64)
    d_gr, v_graze = pairwise_geometry(graze, geo)
    chk.record(
        "Grazing pair 1 km above the atmosphere IS visible",
        bool(v_graze[0, 0, 1]),
        f"separation {d_gr[0, 0, 1]:.1f} km, segment clearance "
        f"{segment_min_radius(graze[0, 0], graze[0, 1]):.1f} km vs blocking radius "
        f"{geo.blocking_radius_km:.1f} km",
    )

    # The other side of that boundary. The grazing check above only proves the
    # test says "visible" somewhere above the blocking radius, and the antipodal
    # check only proves it says "blocked" at a clearance of zero. Between them
    # sits a 6428 km gap in which the threshold could sit anywhere: loosening
    # blocking_radius_km by 200 km admits links straight through the solid body
    # of the Earth and both of those checks still pass. These two pin the
    # threshold itself.
    def pair_with_clearance(clearance_km: float) -> np.ndarray:
        """Two points at radius ``r`` whose segment's closest approach is exact.

        Equal radii put t* at exactly 1/2, so the closest point on the segment
        is the midpoint (clearance_km, 0, 0), at radius clearance_km.
        """
        half_chord = np.sqrt(r * r - clearance_km * clearance_km)
        return np.array(
            [[[clearance_km, -half_chord, 0.0], [clearance_km, half_chord, 0.0]]],
            dtype=np.float64,
        )

    below = pair_with_clearance(geo.blocking_radius_km - 1.0)
    _, v_below = pairwise_geometry(below, geo)
    chk.record(
        "Pair 1 km below the blocking radius is NOT visible",
        not bool(v_below[0, 0, 1]),
        f"segment clearance {segment_min_radius(below[0, 0], below[0, 1]):.1f} km vs "
        f"blocking radius {geo.blocking_radius_km:.1f} km\n"
        f"brackets the threshold from below; the grazing check above brackets it "
        f"from above, together pinning it to +-1 km",
    )

    # Stronger form: sweep straight through the threshold and require the
    # visibility verdict to flip exactly where the blocking radius says it must.
    sweep = np.linspace(
        geo.blocking_radius_km - 200.0, geo.blocking_radius_km + 200.0, 401
    )
    got = np.array(
        [bool(pairwise_geometry(pair_with_clearance(c), geo)[1][0, 0, 1]) for c in sweep]
    )
    want = sweep >= geo.blocking_radius_km
    wrong = int((got != want).sum())
    first_visible = float(sweep[got.argmax()]) if got.any() else float("nan")
    chk.record(
        "Visibility flips exactly at the blocking radius across a 400 km sweep",
        wrong == 0,
        f"401 pairs with clearances {sweep[0]:.0f}..{sweep[-1]:.0f} km in 1 km steps\n"
        f"first clearance reported visible {first_visible:.1f} km, "
        f"blocking radius {geo.blocking_radius_km:.1f} km\n"
        f"{wrong} of 401 verdicts disagree with the configured threshold",
    )

    # The most nearly antipodal real pair in the constellation.
    pos = np.asarray(dataset.positions[0], dtype=np.float64)
    unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    cosang = unit @ unit.T
    np.fill_diagonal(cosang, 1.0)
    i, j = np.unravel_index(np.argmin(cosang), cosang.shape)
    ang = np.degrees(np.arccos(np.clip(cosang[i, j], -1.0, 1.0)))
    chk.record(
        "Most nearly antipodal real pair is NOT visible",
        not bool(dataset.visible[0, i, j]),
        f"{dataset.sat_names[i]} <-> {dataset.sat_names[j]}\n"
        f"angular separation {ang:.1f} deg, distance "
        f"{dataset.distance[0, i, j]:.1f} km",
    )

    # The segment clamp: prove it is load-bearing on a case where the foot of
    # the perpendicular falls outside the segment.
    #
    # It cannot be demonstrated on this dataset. For two satellites at *equal*
    # radius r separated by angle theta,
    #     t* = -(p1.d)/(d.d) = r^2(1-cos t) / 2r^2(1-cos t) = 1/2
    # exactly, for every theta. Within one shell radii are near-equal, so t*
    # clusters tightly around 0.5; it strays outside [0,1] only for a handful of
    # near-antipodal pairs with differing radii, and even then the verdict is
    # unchanged, because the nearest point is then an endpoint at satellite
    # radius, well clear of the blocking sphere. The clamp only alters an
    # *outcome* once radii differ substantially -- a second shell, a ground
    # station, or a GEO relay in a later stage.
    near = np.array([geo.earth_radius_km + 550.0, 0.0, 0.0])
    far = np.array([10000.0, 3000.0, 0.0])

    d_vec = far - near
    t_star = float(-(near @ d_vec) / (d_vec @ d_vec))
    perp_point = near + t_star * d_vec
    unclamped_clearance = float(np.linalg.norm(perp_point))
    clamped_clearance = float(segment_min_radius(near, far))

    pair = np.array([[near, far]])
    _, v_pair = pairwise_geometry(pair, geo)

    unclamped_says_blocked = unclamped_clearance < geo.blocking_radius_km
    clamped_says_clear = clamped_clearance >= geo.blocking_radius_km
    chk.record(
        "Segment clamp is load-bearing (unclamped line test gives a wrong answer)",
        unclamped_says_blocked and clamped_says_clear and bool(v_pair[0, 0, 1]),
        f"differing-radius pair r={np.linalg.norm(near):.0f} km and "
        f"r={np.linalg.norm(far):.0f} km, t* = {t_star:.3f} (outside [0,1])\n"
        f"unclamped infinite-line clearance {unclamped_clearance:.1f} km -> "
        f"{'blocked (WRONG)' if unclamped_says_blocked else 'clear'}\n"
        f"clamped segment clearance {clamped_clearance:.1f} km -> "
        f"{'clear (correct)' if clamped_says_clear else 'blocked'}\n"
        f"production code reports visible = {bool(v_pair[0, 0, 1])}",
    )

    # Informational: how often the clamp binds on the real single-shell data.
    p = np.asarray(dataset.positions[0], dtype=np.float64)
    p_i, p_j = p[:, None, :], p[None, :, :]
    d = p_j - p_i
    dd = np.einsum("...i,...i->...", d, d)
    pd = np.einsum("...i,...i->...", p_i, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_all = np.where(dd > 0.0, -pd / dd, 0.5)
    off_diag = ~np.eye(len(p), dtype=bool)
    t_off = t_all[off_diag]
    binds = ((t_all < 0.0) | (t_all > 1.0)) & off_diag

    perp_all = p_i + t_all[..., None] * d
    unclamped_all = np.sqrt(np.einsum("...i,...i->...", perp_all, perp_all))
    clamped_all = segment_min_radius(p_i, p_j)
    verdict_differs = (
        (unclamped_all < geo.blocking_radius_km)
        != (clamped_all < geo.blocking_radius_km)
    ) & off_diag

    radii = np.linalg.norm(p, axis=1)
    print(
        f"         info: radii span {radii.min():.1f}..{radii.max():.1f} km, so t* is\n"
        f"         near but not exactly 0.5: range {t_off.min():.6f}..{t_off.max():.6f}, "
        f"{np.mean(np.abs(t_off - 0.5) < 0.01) * 100:.2f}% within 0.5+-0.01.\n"
        f"         Clamp binds on {int(binds.sum() // 2)} of "
        f"{int(off_diag.sum() // 2)} pairs and changes the verdict on "
        f"{int(verdict_differs.sum() // 2)}."
    )


# --------------------------------------------------------------------------
# 5. In-plane neighbours
# --------------------------------------------------------------------------
def check_in_plane(chk: Checks, dataset, vcfg: ValidationConfig) -> None:
    section("5. In-plane neighbour stability")

    planes = np.array(dataset.plane_index)
    dist = np.asarray(dataset.distance, dtype=np.float64)
    vis = np.asarray(dataset.visible)

    same_plane = (planes[:, None] == planes[None, :]) & (planes[:, None] >= 0)
    np.fill_diagonal(same_plane, False)
    if not same_plane.any():
        chk.record(
            "In-plane pair exists",
            False,
            "no two selected satellites share an orbital plane",
        )
        return

    mean_dist = dist.mean(axis=0)
    rows, cols = np.where(same_plane)
    best = int(np.argmin(mean_dist[rows, cols]))
    i, j = int(rows[best]), int(cols[best])

    series = dist[:, i, j]
    mean_km = float(series.mean())
    std_km = float(series.std())
    cv = std_km / mean_km
    vis_frac = float(vis[:, i, j].mean())

    chk.record(
        "Adjacent in-plane pair is visible at every timestep",
        vis_frac == 1.0,
        f"{dataset.sat_names[i]} <-> {dataset.sat_names[j]} (plane {planes[i]})\n"
        f"visible for {vis_frac * 100:.1f}% of {dataset.n_steps} timesteps",
    )
    chk.record(
        "Adjacent in-plane distance is stable over time",
        cv < vcfg.in_plane_cv_max,
        f"mean {mean_km:.1f} km, std {std_km:.3f} km, "
        f"coefficient of variation {cv * 100:.4f}% (required < 5%)\n"
        f"range {series.min():.1f} .. {series.max():.1f} km",
    )

    # Confirms the stability metric discriminates rather than passing on
    # everything: a cross-plane pair should be measurably less stable.
    cross = ~same_plane
    np.fill_diagonal(cross, False)
    cross_always_visible = cross & vis.all(axis=0)
    if cross_always_visible.any():
        cr, cc = np.where(cross_always_visible)
        cross_cv = np.array(
            [dist[:, a, b].std() / dist[:, a, b].mean() for a, b in zip(cr, cc)]
        )
        worst = int(np.argmax(cross_cv))
        ci, cj = int(cr[worst]), int(cc[worst])
        chk.record(
            "Cross-plane distance is measurably less stable than in-plane",
            float(cross_cv.max()) > cv,
            f"{dataset.sat_names[ci]} <-> {dataset.sat_names[cj]}: "
            f"CV {cross_cv.max() * 100:.3f}% vs in-plane {cv * 100:.4f}%\n"
            f"({len(cross_cv)} always-visible cross-plane pairs, "
            f"median CV {np.median(cross_cv) * 100:.3f}%)",
        )


# --------------------------------------------------------------------------
# 6. Link budget
# --------------------------------------------------------------------------
def check_capacity(chk: Checks, cfg: SimConfig, dataset, vcfg: ValidationConfig) -> None:
    section("6. Link budget")

    lb = cfg.link_budget

    d = np.linspace(100.0, cfg.geometry.max_range_km, 2000)
    c = capacity_bps(d, lb)
    diffs = np.diff(c)
    chk.record(
        "Capacity strictly decreases with distance (analytic sweep)",
        bool(np.all(diffs < 0.0)),
        f"{len(d)} points from {d[0]:.0f} to {d[-1]:.0f} km\n"
        f"C({d[0]:.0f} km) = {c[0] / 1e6:.1f} Mbps, "
        f"C({d[-1]:.0f} km) = {c[-1] / 1e6:.1f} Mbps\n"
        f"largest step {diffs.max():.3e} bps (must be negative)",
    )

    vis = np.asarray(dataset.visible[0])
    dd = np.asarray(dataset.distance[0], dtype=np.float64)[vis]
    cc = np.asarray(dataset.capacity[0], dtype=np.float64)[vis]
    order = np.argsort(dd)
    steps = np.diff(cc[order])
    tol = vcfg.capacity_step_tolerance_bps  # float32 storage rounding
    chk.record(
        "Capacity non-increasing across real links sorted by distance",
        bool(steps.max() <= tol),
        f"{len(dd)} visible links at t=0, distances "
        f"{dd.min():.0f}..{dd.max():.0f} km\n"
        f"max upward step {steps.max():.3e} bps "
        f"(tolerance {tol:.0e} for float32 storage)",
    )

    # --- absolute values, driven through the production functions -----------
    #
    # These call leosim.linkbudget rather than recomputing the closed form
    # locally. An earlier version of this check computed FSPL inline and
    # asserted on its own arithmetic, which made it unfalsifiable: replacing
    # 20log10 with 10log10 in the production function -- a 60 dB error at
    # 1000 km -- left the whole suite still reporting every check passed.
    d_ref = PIN_DISTANCE_KM
    d_arr = np.array([d_ref])

    fspl = float(free_space_path_loss_db(d_arr, lb)[0])
    reference = (
        20 * np.log10(d_ref * 1e3)
        + 20 * np.log10(lb.carrier_hz)
        + 20 * np.log10(4 * np.pi / SPEED_OF_LIGHT_M_S)
    )
    chk.between(
        f"FSPL at {d_ref:.0f} km, {lb.carrier_hz / 1e9:g} GHz matches closed form",
        fspl,
        reference - 1e-6,
        reference + 1e-6,
        " dB",
        extra=(
            f"leosim.linkbudget.free_space_path_loss_db vs independently "
            f"recomputed 20log10(d_m) + 20log10(f_Hz) + 20log10(4pi/c) = "
            f"{reference:.6f} dB"
        ),
    )

    lat_s = float(latency_s(d_arr)[0])
    chk.between(
        f"Propagation latency at {d_ref:.0f} km equals d/c",
        lat_s * 1e3,
        3.335,
        3.337,
        " ms",
        extra=f"{d_ref:.0f} km / 299792.458 km/s = {PIN_LATENCY_S * 1e3:.5f} ms",
    )

    # Absolute pins. Everything above is relative -- a monotone sweep, or one
    # array agreeing with another -- and a uniform scale error passes all of it.
    pins_apply = all(
        getattr(lb, k) == v for k, v in PIN_CONFIG.items()
    )
    if pins_apply:
        note = f"pinned literal, config at documented defaults"
        chk.close_to(
            f"FSPL at {d_ref:.0f} km matches its pinned absolute value",
            fspl, PIN_FSPL_DB, PIN_REL_TOL, " dB", extra=note,
        )
        chk.close_to(
            f"SNR at {d_ref:.0f} km matches its pinned absolute value",
            float(snr_db(d_arr, lb)[0]), PIN_SNR_DB, PIN_REL_TOL, " dB", extra=note,
        )
        chk.close_to(
            f"Capacity at {d_ref:.0f} km matches its pinned absolute value",
            float(capacity_bps(d_arr, lb)[0]), PIN_CAPACITY_BPS, PIN_REL_TOL,
            " bps", extra=f"{PIN_CAPACITY_BPS / 1e6:.1f} Mbps; {note}",
        )
    else:
        differing = ", ".join(
            f"{k}={getattr(lb, k)!r} (pinned at {v!r})"
            for k, v in PIN_CONFIG.items()
            if getattr(lb, k) != v
        )
        print(
            f"         note: absolute FSPL/SNR/capacity pins skipped -- link budget\n"
            f"         config differs from the values they were computed at:\n"
            f"         {differing}"
        )

    # Latency depends on nothing but the speed of light, so its pin is
    # unconditional.
    chk.close_to(
        f"Latency at {d_ref:.0f} km matches its pinned absolute value",
        lat_s, PIN_LATENCY_S, PIN_REL_TOL, " s",
        extra="pinned literal; depends only on c, so no config guard applies",
    )

    mask = np.asarray(dataset.visible)
    lat = np.asarray(dataset.latency)[mask].astype(np.float64)
    dst = np.asarray(dataset.distance)[mask].astype(np.float64)
    err = float(np.abs(lat - dst * 1e3 / SPEED_OF_LIGHT_M_S).max())
    chk.record(
        "Latency array equals distance/c for every visible link",
        err < 1e-8,
        f"max absolute error {err:.3e} s over {lat.size} visible link-timesteps",
    )


# --------------------------------------------------------------------------
# 7. Range limit
# --------------------------------------------------------------------------
def check_range_cap(chk: Checks, cfg: SimConfig, dataset) -> None:
    section("7. Range limit consistency")

    geo = cfg.geometry
    alt = float(np.asarray(dataset.altitudes).mean())
    horizon = max_geometric_range_km(alt, geo)

    vis = np.asarray(dataset.visible)
    dist = np.asarray(dataset.distance, dtype=np.float64)
    observed_max = float(dist[vis].max()) if vis.any() else 0.0

    chk.record(
        "No visible link exceeds the configured maximum range",
        observed_max <= geo.max_range_km,
        f"longest visible link {observed_max:.1f} km, cap {geo.max_range_km:.0f} km",
    )
    chk.record(
        "Occlusion, not the range cap, is the binding constraint",
        observed_max < geo.max_range_km,
        f"grazing-tangent horizon at mean altitude {alt:.0f} km is {horizon:.0f} km, "
        f"inside the {geo.max_range_km:.0f} km cap\n"
        f"the range limit therefore filters no links in this shell",
    )


def main() -> int:
    cfg = SimConfig().resolve(ROOT)
    vcfg = ValidationConfig()
    print("=" * 72)
    print("  LEO CONSTELLATION SIMULATOR -- STAGE 1 VALIDATION")
    print("=" * 72)

    dataset = run(cfg, verbose=True)
    print()
    print(dataset.summary())

    chk = Checks()
    check_orbital_period(chk, cfg, dataset, vcfg)
    check_altitudes(chk, cfg, dataset)
    check_symmetry(chk, dataset)
    check_antipodal(chk, cfg, dataset)
    check_in_plane(chk, dataset, vcfg)
    check_capacity(chk, cfg, dataset, vcfg)
    check_range_cap(chk, cfg, dataset)
    return chk.summarise()


if __name__ == "__main__":
    sys.exit(main())
