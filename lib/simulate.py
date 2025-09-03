import numpy as np
import random
import pandas as pd

# Energy model constant
ENERGY_K = 0.02

#Emulated device profiles (latency ms, energy factor)
DEFAULT_DEVICES = {
    "MCU":   {"latency": {"fp32": 80, "int8": 25, "tinycnn": 7, "slm": 12000}, "energy_factor": 1.0, "net_rtt": (0, 0)},
    "EDGE":  {"latency": {"fp32": 22, "int8": 10, "tinycnn": 4, "slm": 4000},  "energy_factor": 1.4, "net_rtt": (5, 15)},
    "CLOUD": {"latency": {"fp32": 12, "int8": 12, "tinycnn": 12, "slm": 1500}, "energy_factor": 1.9, "net_rtt": (30, 120)},
}

# Baseline accuracy matrix for "vision-like" tasks
# Accuracy model (per difficulty). Rough, but shows trade-offs.
# You can adjust for your audience: larger gaps → clearer trade-offs.
ACC_MATRIX = {
    "easy": {"fp32": 0.92, "int8": 0.88, "tinycnn": 0.80, "slm": 0.0},
    "hard": {"fp32": 0.86, "int8": 0.80, "tinycnn": 0.70, "slm": 0.0},
}

# For language model (SLM/LLM) requests we model "quality" as accuracy-like metric
LM_QUALITY = {
    "short": {"slm": 0.70, "llm_cloud": 0.90},
    "long":  {"slm": 0.60, "llm_cloud": 0.92},
}

def set_int8_accuracy(measured_acc: float):
    ACC_MATRIX["easy"]["int8"] = float(max(0.0, min(1.0, measured_acc)))
    ACC_MATRIX["hard"]["int8"] = float(max(0.0, min(1.0, measured_acc - 0.05)))

def rnd_net_rtt(low_high):
    low, high = low_high
    if low==0 and high==0:
        return 0
    return random.randint(low, high)

def infer(device_name: str, model_name: str, difficulty: str, devices: dict) -> tuple:
    dev = devices[device_name]
    base_lat = dev["latency"].get(model_name, 999)
    rtt = rnd_net_rtt(dev["net_rtt"])
    latency = base_lat + rtt
    # Energy model (arbitrary but consistent):
    # energy (mJ) = energy_factor * latency_ms * k
    energy = dev["energy_factor"] * latency * ENERGY_K
    # Simulate confidence loosely from accuracy
    acc = ACC_MATRIX.get(difficulty, {}).get(model_name, 0.75)
    conf = float(np.clip(np.random.normal(loc=0.65 + 0.3*acc, scale=0.1), 0.01, 0.999))
    correct = (random.random() < acc)
    return correct, float(latency), float(energy), conf

def lm_infer(route: str, length: str, devices: dict) -> tuple:
    """
    route: 'MCU/slm', 'EDGE/slm', or 'CLOUD/llm_cloud'
    length: 'short' or 'long'
    Returns (correct, latency_ms, energy_mJ, confidence, quality)
    """
    if route.startswith("CLOUD"):
        base = devices["CLOUD"]["latency"].get("slm", 1500)
        rtt = rnd_net_rtt(devices["CLOUD"]["net_rtt"])
        latency = base + rtt + (10 if length=="long" else 5)
        energy = devices["CLOUD"]["energy_factor"] * latency * ENERGY_K
        quality = LM_QUALITY[length]["llm_cloud"]
    else:
        tier = "EDGE" if route.startswith("EDGE") else "MCU"
        base = devices[tier]["latency"].get("slm", 12000)
        rtt = rnd_net_rtt(devices[tier]["net_rtt"])
        latency = base + rtt + (20 if length=="long" else 8)
        energy = devices[tier]["energy_factor"] * latency * ENERGY_K
        quality = LM_QUALITY[length]["slm"]
    correct = (random.random() < quality)
    conf = float(np.clip(np.random.normal(loc=0.6 + 0.3*quality, scale=0.1), 0.01, 0.999))
    return correct, float(latency), float(energy), conf, quality

#Generate workload (10 requests by default)
def generate_workload(n=10, easy_ratio=0.6, deadline_range=(20,120), lm_ratio=0.2):
    rows = []
    for i in range(n):
        if random.random() < lm_ratio:
            length = "long" if random.random() < 0.4 else "short"
            deadline = random.randint(deadline_range[0]+2000, deadline_range[1]+13000)
            rows.append({"id": i, "type":"lm", "length": length, "difficulty": None, "deadline_ms": deadline})
        else:
            difficulty = "easy" if random.random() < easy_ratio else "hard"
            deadline = random.randint(*deadline_range)
            rows.append({"id": i, "type":"cv", "length": None, "difficulty": difficulty, "deadline_ms": deadline})
    print(pd.DataFrame(rows))
    return pd.DataFrame(rows)

def evaluate_policy(workload_df: pd.DataFrame, policy_fn, devices: dict) -> pd.DataFrame:
    out = []
    for _, req in workload_df.iterrows():
        route, correct, latency, energy, conf = policy_fn(req, devices)
        out.append({
            "id": int(req["id"]), "type": req["type"], "route": route, "correct": bool(correct),
            "latency_ms": float(latency), "energy_mJ": float(energy), "confidence": float(conf),
            "deadline_ms": int(req["deadline_ms"]), "deadline_met": float(latency) <= float(req["deadline_ms"])
        })
    print(pd.DataFrame(out))
    return pd.DataFrame(out)

# Reference policies
def pol_always_local(req, devices):
    if req["type"]=="lm":
        route = "MCU/slm"
        correct, lat, eng, conf, _q = lm_infer(route, req["length"], devices)
        return route, correct, lat, eng, conf
    else:
        correct, lat, eng, conf = infer("MCU", "tinycnn", req["difficulty"], devices)
        return "MCU/tinycnn", correct, lat, eng, conf

def pol_always_cloud(req, devices):
    if req["type"]=="lm":
        route = "CLOUD/llm_cloud"
        correct, lat, eng, conf, _q = lm_infer(route, req["length"], devices)
        return route, correct, lat, eng, conf
    else:
        correct, lat, eng, conf = infer("CLOUD", "fp32", req["difficulty"], devices)
        return "CLOUD/fp32", correct, lat, eng, conf

def pol_conf_threshold(req, devices, thr=0.78):
    if req["type"]=="lm":
        c1, l1, e1, conf1, _q1 = lm_infer("MCU/slm", req["length"], devices)
        if conf1 >= thr:
            return "MCU/slm", c1, l1, e1, conf1
        c2, l2, e2, conf2, _q2 = lm_infer("EDGE/slm", req["length"], devices)
        if conf2 >= thr:
            return "EDGE/slm", c2, l2, e2, conf2
        c3, l3, e3, conf3, _q3 = lm_infer("CLOUD/llm_cloud", req["length"], devices)
        return "CLOUD/llm_cloud", c3, l3, e3, conf3
    else:
        c1, l1, e1, conf1 = infer("MCU", "int8", req["difficulty"], devices)
        if conf1 >= thr:
            return "MCU/int8", c1, l1, e1, conf1
        c2, l2, e2, conf2 = infer("EDGE", "fp32", req["difficulty"], devices)
        if conf2 >= thr:
            return "EDGE/fp32", c2, l2, e2, conf2
        c3, l3, e3, conf3 = infer("CLOUD", "fp32", req["difficulty"], devices)
        return "CLOUD/fp32", c3, l3, e3, conf3

def pol_latency_aware(req, devices, margin=0.9):
    if req["type"]=="lm":
        c1, l1, e1, conf1, _ = lm_infer("MCU/slm", req["length"], devices)
        if l1 <= margin * req["deadline_ms"]:
            return "MCU/slm", c1, l1, e1, conf1
        c2, l2, e2, conf2, _ = lm_infer("EDGE/slm", req["length"], devices)
        if l2 <= margin * req["deadline_ms"]:
            return "EDGE/slm", c2, l2, e2, conf2
        c3, l3, e3, conf3, _ = lm_infer("CLOUD/llm_cloud", req["length"], devices)
        return "CLOUD/llm_cloud", c3, l3, e3, conf3
    else:
        c1, l1, e1, conf1 = infer("MCU", "tinycnn", req["difficulty"], devices)
        if l1 <= margin * req["deadline_ms"]:
            return "MCU/tinycnn", c1, l1, e1, conf1
        c2, l2, e2, conf2 = infer("EDGE", "tinycnn", req["difficulty"], devices)
        if l2 <= margin * req["deadline_ms"]:
            return "EDGE/tinycnn", c2, l2, e2, conf2
        c3, l3, e3, conf3 = infer("CLOUD", "fp32", req["difficulty"], devices)
        return "CLOUD/fp32", c3, l3, e3, conf3

def pol_hybrid(req, devices, conf_thr=0.8, margin=0.85):
    if req["type"]=="lm":
        c1, l1, e1, conf1, _ = lm_infer("MCU/slm", req["length"], devices)
        if conf1 >= conf_thr and l1 <= margin * req["deadline_ms"]:
            return "MCU/slm", c1, l1, e1, conf1
        c2, l2, e2, conf2, _ = lm_infer("EDGE/slm", req["length"], devices)
        if l2 <= margin * req["deadline_ms"]:
            return "EDGE/slm", c2, l2, e2, conf2
        c3, l3, e3, conf3, _ = lm_infer("CLOUD/llm_cloud", req["length"], devices)
        return "CLOUD/llm_cloud", c3, l3, e3, conf3
    else:
        c1, l1, e1, conf1 = infer("MCU", "int8", req["difficulty"], devices)
        if conf1 >= conf_thr and l1 <= margin * req["deadline_ms"]:
            return "MCU/int8", c1, l1, e1, conf1
        c2, l2, e2, conf2 = infer("EDGE", "fp32", req["difficulty"], devices)
        if l2 <= margin * req["deadline_ms"]:
            return "EDGE/fp32", c2, l2, e2, conf2
        c3, l3, e3, conf3 = infer("CLOUD", "fp32", req["difficulty"], devices)
        return "CLOUD/fp32", c3, l3, e3, conf3