# data/

## `starlink.tle`

Starlink two-line element sets in three-line format (name line + two element lines),
10,744 satellites. Committed because it cannot be re-fetched: the public catalogue is
replaced daily, and the stage-1 cache key hashes these bytes, so any other copy is a
different constellation.

| | |
|---|---|
| Source | Not recorded at fetch time. The format matches CelesTrak's GP `FORMAT=tle` output (the source the base paper used), but treat this as unconfirmed. |
| Fetched | 2026-08-21, around 04:58 UTC (file mtime 10:28 IST) |
| Element epochs | 2026-07-30 to 2026-08-21 02:23 UTC |
| SHA-256 | `e86a21a3c4868b541f39ff5db96477aa24a8de44a11b0d6da83f3c5ce1d53d54` |

The newest element epoch is 2026-08-21 02:23 UTC, so the file cannot have been fetched
before then. `results/REPRODUCTION.md` §2 says "fetched 2026-08-20"; that date is the
simulation epoch (median element epoch of the 187 selected satellites, 2026-08-20 15:15
UTC), not the fetch date.

The paper's epoch was 2024-06-10 08:23 UTC. Matching it needs a historical TLE file,
which this is not.
