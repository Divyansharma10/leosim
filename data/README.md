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

## `cities.csv`

Phase 2 demand sources: the 200 most populous places in GeoNames `cities15000`
(CC BY 4.0, https://download.geonames.org/export/dump/cities15000.zip, fetched
2026-09-25), keeping feature codes PPL, PPLA-PPLA4, PPLC and PPLG. PPLX (city
sections) is dropped so districts are not counted twice. Columns: `geonameid, name,
country, lat, lon, population, timezone`. SHA-256
`5b91796b55fa8b57441b6355e6d23f23d2da8054c7a1876f64e38469aab9c0ab`.

Known biases, not corrected:

- GeoNames gives several Chinese cities their whole municipality's population
  (Chongqing 22 M, Chengdu 21 M), so China holds 72 of the 200 rows.
- Populations are administrative, not metropolitan: New York City is 8.8 M, not
  the ~19 M of its metro area, and San Francisco does not make the list.
- The list is population, not demand. It says nothing about where the service is
  licensed or who subscribes.

## `gateways_krishnan.csv`

The eight ground gateways named in Fig. 1 of Krishnan et al., *Reinforcement Learning
for Opportunistic Routing in Software-Defined LEO-Terrestrial Systems*, arXiv
2601.13662v1 (read 2026-09-26): Los Angeles, London, Dubai, Tokyo, Singapore, Sao
Paulo, Johannesburg, Sydney. The paper gives names on a map only; coordinates are
the GeoNames `cities15000` entry for each city (same source as `cities.csv`).

That paper gives NO numeric values for gateway-link bandwidth, transmit power, noise
temperature, path-loss exponent or elevation mask -- they appear only as symbols
(Eqs. 8-11) and there is no parameter table. Nothing else was taken from it.
