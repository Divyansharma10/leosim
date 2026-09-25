"""Pairwise distance and Earth-occlusion line-of-sight geometry."""

from __future__ import annotations

import numpy as np

from .config import GeometryConfig


def segment_min_radius(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Smallest distance from Earth's centre to the *segment* p1->p2.

    This is deliberately not the perpendicular distance to the infinite line
    through p1 and p2. The foot of that perpendicular can lie outside the
    segment, in which case the line dives through Earth but the piece of it
    between the two satellites never does -- an unclamped test reports those
    pairs as occluded when they are in fact mutually visible.

    Restricting the search to the segment means that where the foot of the
    perpendicular falls outside it, the nearest point is an endpoint at
    satellite radius, which is always well clear of Earth.

    Symmetry by construction
    ------------------------
    The obvious parameterisation, ``|p1 + clamp(t*) * (p2 - p1)|``, is not
    symmetric in floating point: computed as (i, j) and as (j, i) it rounds
    differently, and pairs whose clearance sits within an ulp of the blocking
    radius then come out visible one way and occluded the other. On a
    187-satellite run that was 6 asymmetric entries in 63 million -- harmless
    in aggregate, but a visibility matrix that disagrees with its own transpose
    breaks the bidirectional-link constraint every selector is checked against.

    So the quantity is built from expressions that are exactly antisymmetric or
    exactly symmetric under the swap, and therefore agree bit for bit:

    - the foot of the perpendicular lies inside the segment iff
      ``p1.d <= 0 <= p2.d``, which maps to itself when p1 <-> p2 and d -> -d;
    - inside, the distance is ``|p1 x p2| / |d|``; the cross product negates
      exactly and d negates exactly, so both norms are unchanged;
    - outside, the nearest point is an endpoint, giving ``min(|p1|, |p2|)``.

    Broadcasting: p1 and p2 may carry any leading batch dimensions with a
    trailing axis of length 3.
    """
    d = p2 - p1
    dd = np.einsum("...i,...i->...", d, d)
    p1d = np.einsum("...i,...i->...", p1, d)
    p2d = np.einsum("...i,...i->...", p2, d)

    r1 = np.sqrt(np.einsum("...i,...i->...", p1, p1))
    r2 = np.sqrt(np.einsum("...i,...i->...", p2, p2))
    endpoint = np.minimum(r1, r2)

    cross = np.cross(p1, p2)
    cross_norm = np.sqrt(np.einsum("...i,...i->...", cross, cross))
    with np.errstate(divide="ignore", invalid="ignore"):
        perpendicular = np.where(dd > 0.0, cross_norm / np.sqrt(dd), endpoint)

    inside = (p1d <= 0.0) & (p2d >= 0.0) & (dd > 0.0)
    return np.where(inside, perpendicular, endpoint)


def pairwise_geometry(
    positions_chunk: np.ndarray, cfg: GeometryConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Distance matrix and visibility mask for one block of timesteps.

    Parameters
    ----------
    positions_chunk : (C, N, 3) km, Earth-centred inertial.

    Returns
    -------
    distance : (C, N, N) km, zero on the diagonal.
    visible  : (C, N, N) bool, symmetric, False on the diagonal.
    """
    dtype = positions_chunk.dtype
    p_i = positions_chunk[:, :, None, :]  # (C, N, 1, 3)
    p_j = positions_chunk[:, None, :, :]  # (C, 1, N, 3)

    delta = p_j - p_i
    distance = np.sqrt(np.einsum("...i,...i->...", delta, delta)).astype(dtype)

    clearance = segment_min_radius(p_i, p_j)

    visible = (
        (clearance >= cfg.blocking_radius_km)
        & (distance <= cfg.max_range_km)
        & (distance > 0.0)
    )

    n = positions_chunk.shape[1]
    diag = np.arange(n)
    visible[:, diag, diag] = False
    distance[:, diag, diag] = 0.0

    return distance, visible


def occlusion_clearance(
    positions_chunk: np.ndarray,
) -> np.ndarray:
    """Clearance radius matrix only -- used by validation to inspect geometry."""
    p_i = positions_chunk[:, :, None, :]
    p_j = positions_chunk[:, None, :, :]
    return segment_min_radius(p_i, p_j)


def max_geometric_range_km(altitude_km: float, cfg: GeometryConfig) -> float:
    """Grazing-tangent range between two satellites at equal altitude.

    For a shell at 550 km over a 6428 km blocking sphere this is ~5168 km, well
    inside the 7000 km hardware limit -- so the range cap is normally inactive
    and occlusion is the binding constraint.
    """
    r = cfg.earth_radius_km + altitude_km
    b = cfg.blocking_radius_km
    if r <= b:
        return 0.0
    return 2.0 * float(np.sqrt(r * r - b * b))
