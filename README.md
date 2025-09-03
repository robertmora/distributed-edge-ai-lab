
# Summer School on Edge AI – Hands-On Lab

This repository accompanies the keynote *"From Edge to Tiny: Reimagining AI in the Era of Generative and Embedded Intelligence".*  
It provides hands-on exercises on **TinyML training** and a **distributed AI Inference simulator**.

## What’s Inside
- `notebooks/01_tinyml_training.ipynb` → Train & quantize a TinyML CNN (MNIST)  
- `notebooks/02_simulator_colab.ipynb` → Colab-friendly step-by-step simulator  
- `app/streamlit_app.py` → Interactive simulator UI (Streamlit)  
- `lib/simulate.py` → Simulation logic (devices, workloads, policies)  
- `scripts/run_simulator.py` → CLI runner for quick experiments  
- `INSTALL.md` → Full installation instructions  

## Quick Start (Simulator Only)
```bash
git clone https://github.com/robertmora/distributed-edge-ai-lab.git
cd distributed-edge-ai-lab
python3 -m venv .venv
source .venv/bin/activate   # (Windows: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 to explore.

## Run in Colab
- `TinyML Training` [![Open TinyML Training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15OEifrX-zNcVeYvsXhLkoi9MfLw40x4W?usp=sharing)

- `Simulator` [![Open Simulator in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jwduBVq0Xcla1Q-lLdC-gbtwBRiTJcB5?usp=sharing)

## Requirements
- Python 3.7–3.12  
- Streamlit, NumPy, Pandas, Matplotlib, NetworkX  
- (Optional) TensorFlow for TinyML training notebook  

See [INSTALL.md](INSTALL.md) for detailed instructions.

---
Have fun experimenting with TinyML and Distributed AI inference!
