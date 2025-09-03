import pandas as pd
from lib.simulate import (
    DEFAULT_DEVICES, generate_workload, evaluate_policy,
    pol_always_local, pol_always_cloud, pol_conf_threshold, pol_latency_aware, pol_hybrid
)

devices = DEFAULT_DEVICES.copy()
devices["MCU"]["net_rtt"] = (0, 0)
devices["EDGE"]["net_rtt"] = (5, 15)
devices["CLOUD"]["net_rtt"] = (30, 120)

workload = generate_workload(n=150, easy_ratio=0.6, deadline_range=(20,120), lm_ratio=0.25)

policies = {
    "AlwaysLocal": pol_always_local,
    "AlwaysCloud": pol_always_cloud,
    "ConfThresh":  lambda r,d: pol_conf_threshold(r,d,thr=0.78),
    "LatencyAware":lambda r,d: pol_latency_aware(r,d,margin=0.90),
    "Hybrid":      lambda r,d: pol_hybrid(r,d,conf_thr=0.8, margin=0.85),
}

rows = []
for name, fn in policies.items():
    df = evaluate_policy(workload, fn, devices)
    summary = df.agg(
        avg_latency_ms=("latency_ms","mean"),
        deadline_met_pct=("deadline_met","mean"),
        accuracy=("correct","mean"),
        avg_energy_mJ=("energy_mJ","mean"),
    ).to_frame().T
    summary["policy"] = name
    rows.append(summary)

out = pd.concat(rows).reset_index(drop=True)
out["deadline_met_pct"] = (out["deadline_met_pct"]*100).round(1)
out["accuracy"] = (out["accuracy"]*100).round(1)
print(out[["policy","accuracy","deadline_met_pct","avg_latency_ms","avg_energy_mJ"]].to_string(index=False))
