import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lib.simulate import (
    DEFAULT_DEVICES, ACC_MATRIX, generate_workload, evaluate_policy,
    pol_always_local, pol_always_cloud, pol_conf_threshold, pol_latency_aware, pol_hybrid
)

st.set_page_config(page_title="Distributed AI Inference in the Computing Continuum", layout="wide")
st.title("Distributed AI Inference in the Computing Continuum")

with st.sidebar:
    st.header("Configuration")
    n_requests = st.slider("Workload size", 1, 20, 10, 1)
    easy_ratio = st.slider("CV easy-ratio", 0.0, 1.0, 0.6, 0.05)
    deadline_min, deadline_max = st.slider("CV deadlines (ms)", 10, 200, (20, 120), 5)
    lm_ratio = st.slider("LM request ratio", 0.0, 0.8, 0.25, 0.05)
    mcu_low = st.number_input("MCU RTT low", 0, 200, 0)
    mcu_high = st.number_input("MCU RTT high", 0, 200, 0)
    edge_low = st.number_input("EDGE RTT low", 0, 200, 5)
    edge_high = st.number_input("EDGE RTT high", 0, 200, 15)
    cloud_low = st.number_input("CLOUD RTT low", 0, 300, 30)
    cloud_high = st.number_input("CLOUD RTT high", 0, 300, 120)
    conf_thr = st.slider("Confidence Threshold", 0.5, 0.95, 0.78, 0.01)
    margin = st.slider("Deadline margin", 0.5, 1.0, 0.9, 0.01)

devices = DEFAULT_DEVICES.copy()
devices["MCU"]["net_rtt"] = (int(mcu_low), int(mcu_high))
devices["EDGE"]["net_rtt"] = (int(edge_low), int(edge_high))
devices["CLOUD"]["net_rtt"] = (int(cloud_low), int(cloud_high))

workload = generate_workload(n=n_requests, easy_ratio=easy_ratio, deadline_range=(deadline_min, deadline_max), lm_ratio=lm_ratio)

st.subheader("Network Topology (Live)")
G = nx.DiGraph()
G.add_node("MCU 1", tier="MCU")
G.add_node("MCU 2", tier="MCU")
G.add_node("EDGE", tier="EDGE")
G.add_node("CLOUD", tier="CLOUD")
G.add_edge("MCU 1", "EDGE", link="mcu↔edge")
G.add_edge("MCU 2", "EDGE", link="mcu↔edge")
G.add_edge("EDGE", "CLOUD", link="edge↔cloud")

pos = {"MCU 1":(-1,0), "MCU 2":(1,0), "EDGE":(0,1), "CLOUD":(0,4)}
tier_color = {"MCU":"#cdeaf4", "EDGE":"#e8f5d4", "CLOUD":"#f4e3ff"}
node_colors = [tier_color[G.nodes[n]["tier"]] for n in G.nodes()]

fig, ax = plt.subplots(figsize=(7.5,4))
nx.draw(G, pos, with_labels=True, node_size=2200, node_color=node_colors,
        font_size=9, font_weight="bold", arrows=True, ax=ax)

edge_labels = {
    ("MCU 1","EDGE"): f"RTT {devices['EDGE']['net_rtt'][0]}–{devices['EDGE']['net_rtt'][1]} ms",
    ("MCU 2","EDGE"): f"RTT {devices['EDGE']['net_rtt'][0]}–{devices['EDGE']['net_rtt'][1]} ms",
    ("EDGE","CLOUD"): f"RTT {devices['CLOUD']['net_rtt'][0]}–{devices['CLOUD']['net_rtt'][1]} ms",
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
ax.set_axis_off()
st.pyplot(fig)

st.subheader("Run Policies")
policies = {
    f"AlwaysLocal(tinycnn)": pol_always_local,
    f"AlwaysCloud(fp32/LLM)": pol_always_cloud,
    f"ConfThresh({conf_thr:.2f})": lambda req, devs: pol_conf_threshold(req, devs, thr=conf_thr),
    f"LatencyAware({margin:.2f})": lambda req, devs: pol_latency_aware(req, devs, margin=margin),
    f"Hybrid(conf={conf_thr:.2f},m={margin:.2f})": lambda req, devs: pol_hybrid(req, devs, conf_thr=conf_thr, margin=margin),
}
tabs = st.tabs(list(policies.keys()))

summaries = []
for tab, (name, fn) in zip(tabs, policies.items()):
    with tab:
        df = evaluate_policy(workload, fn, devices)
        st.dataframe(df.head(20))
        summary = pd.DataFrame([{
            "avg_latency_ms": df["latency_ms"].mean(),
            "deadline_met_pct": df["deadline_met"].mean() * 100,
            "accuracy": df["correct"].mean() * 100,
            "avg_energy_mJ": df["energy_mJ"].mean(),
            "policy": name,
        }]).round({"deadline_met_pct": 1, "accuracy": 1})
        summaries.append(summary)

if summaries:
    summary_df = pd.concat(summaries, ignore_index=True)
    st.subheader("Summary")
    st.dataframe(summary_df)

    col2, col3 = st.columns(2)
    with col2:
        st.subheader("Accuracy vs Deadline Satisfaction")
        fig2, ax2 = plt.subplots()
        for _, r in summary_df.iterrows():
            ax2.scatter(r["accuracy"], r["deadline_met_pct"])
            ax2.text(r["accuracy"]+0.2, r["deadline_met_pct"]+0.2, r["policy"], fontsize=8)
        ax2.set_xlabel("Accuracy (%)")
        ax2.set_ylabel("Deadline Met (%)")
        st.pyplot(fig2)
    with col3:
        st.subheader("Energy vs Accuracy")
        fig3, ax3 = plt.subplots()
        for _, r in summary_df.iterrows():
            ax3.scatter(r["avg_energy_mJ"], r["accuracy"])
            ax3.text(r["avg_energy_mJ"]+0.02, r["accuracy"]+0.2, r["policy"], fontsize=8)
        ax3.set_xlabel("Avg Energy (mJ)")
        ax3.set_ylabel("Accuracy (%)")
        st.pyplot(fig3)

st.caption("Summer School on Edge Artificial Intelligence – KTH (Stockholm), 3rd of September 2025. Lecturer: Roberto Morabito (EURECOM)")
