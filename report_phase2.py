"""Phase 2 comparison: Greedy and DoTD-forward, extension terms off and on.

    python report_phase2.py                 # reads results/phase2/*.csv

Reads every run_traffic.py output in the directory, labels each row by its
selector and extension weights (from the stamped topology_variant), and prints:

1. per load and configuration: delivered, loss, delay (mean, p95), routed path
   length, stretch, gateway imbalance;
2. terms-on minus terms-off for each selector, per load;
3. timestep-paired comparison, terms on vs off, same selector: on how many
   steps each delivers more, and has lower gateway imbalance. One seed and one
   window, so this is within-run evidence, NOT a variance estimate;
4. per-gateway load share against capacity share at the highest load.

Nothing is recomputed here: every number is read from what TrafficSim wrote.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {
    "delivered_frac_of_routed": "delivered",
    "lost_frac_of_routed": "loss",
    "delay_mean_ms": "delay ms",
    "delay_p95_ms_median": "p95 ms",
    "path_km_mean": "path km",
    "stretch_mean": "stretch",
    "gateway_imbalance_mean": "imbal mean",
    "gateway_imbalance_median": "imbal med",
}


def weights(variant: str) -> tuple[float, float]:
    m = re.search(r"w_egress=([\d.]+), w_congestion=([\d.]+)", variant)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def label(row) -> str:
    e, c = row["w_egress"], row["w_congestion"]
    terms = "off" if e == 0 and c == 0 else f"egress {e:g} / cong {c:g}"
    return f"{row['algorithm']} | {terms}"


def load_all(d: Path, kind: str) -> pd.DataFrame:
    frames = []
    for f in sorted(d.glob(f"traffic_*{kind}")):
        if kind == ".csv" and (f.stem.endswith("_gateways") or "_steps" in f.name):
            continue
        df = pd.read_csv(f)
        if kind != ".csv":
            # steps/gateways files carry no variant column: take it from the name.
            m = re.search(r"_egress([\d.]+)_cong([\d.]+)", f.name)
            df["w_egress"], df["w_congestion"] = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/phase2")
    args = p.parse_args()
    d = Path(args.dir)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    s = load_all(d, ".csv")
    s[["w_egress", "w_congestion"]] = s["topology_variant"].apply(lambda v: pd.Series(weights(v)))
    s["config"] = s.apply(label, axis=1)
    print(f"variants: {s['routing_variant'].iloc[0]} | {s['traffic_variant'].iloc[0].split(', ', 1)[1]}")
    print(f"n_sats {s['n_sats'].iloc[0]}, step {s['step_s'].iloc[0]:g} s, seeds {sorted(s['seed'].unique())}\n")

    print("## 1. Per load and configuration\n")
    t1 = s.sort_values(["peak_total_gbps", "config"])[["peak_total_gbps", "config", *METRICS, "feeder_bound_frac"]]
    print(t1.rename(columns=METRICS).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n## 2. Terms on minus terms off, same selector\n")
    rows = []
    for (alg, load), g in s.groupby(["algorithm", "peak_total_gbps"]):
        off = g[(g["w_egress"] == 0) & (g["w_congestion"] == 0)]
        for _, on in g[(g["w_egress"] > 0) | (g["w_congestion"] > 0)].iterrows():
            if off.empty:
                continue
            r = {"selector": alg, "load": load, "terms": on["config"].split(" | ")[1]}
            for col, name in METRICS.items():
                r[name] = on[col] - off.iloc[0][col]
            rows.append(r)
    t2 = pd.DataFrame(rows).sort_values(["selector", "terms", "load"])
    print(t2.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n## 3. Timestep-paired, terms on vs off (one seed, one window: not a variance estimate)\n")
    st = load_all(d, "_steps.csv.gz")
    st["config"] = st.apply(label, axis=1)
    rows = []
    for (alg, load), g in st.groupby(["algorithm", "peak_total_gbps"]):
        off = g[(g["w_egress"] == 0) & (g["w_congestion"] == 0)].set_index("t")
        for cfg, on in g[(g["w_egress"] > 0) | (g["w_congestion"] > 0)].groupby("config"):
            on = on.set_index("t")
            dd = on["delivered_bits"] - off["delivered_bits"]
            di = (on["gateway_imbalance"] - off["gateway_imbalance"]).dropna()
            rows.append({"config": cfg, "load": load,
                         "steps": len(dd),
                         "deliv more": (dd > 1e-6).mean(), "deliv less": (dd < -1e-6).mean(),
                         "imbal lower": (di < -1e-9).mean(), "imbal higher": (di > 1e-9).mean(),
                         "imbal median diff": di.median()})
    print(pd.DataFrame(rows).sort_values(["config", "load"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n## 4. Per-gateway feeder load share vs capacity share, highest load\n")
    gw = load_all(d, "_gateways.csv")
    gw["config"] = gw.apply(label, axis=1)
    top = gw[gw["peak_total_gbps"] == gw["peak_total_gbps"].max()]
    wide = top.pivot_table(index="gateway", columns="config", values="load_share")
    wide.insert(0, "capacity share", top.groupby("gateway")["capacity_share"].first())
    over = top.pivot_table(index="gateway", columns="config", values="steps_over_capacity")
    print(wide.sort_values("capacity share", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nsteps on which the gateway's offered feeder load exceeded its capacity:\n")
    print(over.loc[wide.sort_values("capacity share", ascending=False).index].to_string())

    t1.to_csv(d / "phase2_by_config.csv", index=False)
    t2.to_csv(d / "phase2_on_minus_off.csv", index=False)
    print(f"\n[out] {d / 'phase2_by_config.csv'}, {d / 'phase2_on_minus_off.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
