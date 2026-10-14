"""Demand-perturbation summary: gain of both terms over terms-off, vs predictor error.

Run from the repo root:  python results/phase2/robust/summarise.py
"""
import glob
import re

import pandas as pd

P = "results/phase2/traffic_187sats_max_d_over_c_{}hop_populous20_feeder-stage1_step5.csv"
KEY = ["algorithm", "peak_total_gbps"]
COL = "delivered_frac_of_routed"

off = pd.read_csv(P.format("")).set_index(KEY)[COL]
rows = [pd.read_csv(P.format("egress0.2_cong0.2_")).assign(sd=0.0, nseed=0)]
for f in glob.glob("results/phase2/robust/*_prednoise*-s*.csv"):
    if f.endswith(("_gateways.csv", "_steps.csv.gz")):
        continue
    sd, s = re.search(r"_prednoise([\d.]+)-s(\d+)\.csv$", f).groups()
    rows.append(pd.read_csv(f).assign(sd=float(sd), nseed=int(s)))
on = pd.concat(rows)
on = on[on.peak_total_gbps >= 3]
on["gain_pts"] = 100 * (on[COL].values - off.reindex(pd.MultiIndex.from_frame(on[KEY])).values)

t = on.pivot_table(index=["algorithm", "sd"], columns="peak_total_gbps", values="gain_pts",
                   aggfunc=["mean", "min", "max"])
pd.set_option("display.width", 250)
print("delivered, both terms (0.2/0.2) minus terms off, percentage points; 5 s step, traffic seed 7")
print("sd = per-city lognormal predictor error; sd 0 = perfect forecast; min/max over noise seeds 1-3\n")
print(t.round(2).to_string())
t.to_csv("results/phase2/robust/gain_vs_noise.csv")
