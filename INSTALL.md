
# INSTALL.md – Hands-On Lab

Welcome to this hands-on lab session!  
This exercise accompanies the keynote *"From Edge to Tiny: Reimagining AI in the Era of Generative and Embedded Intelligence."*  

The goal is to let you **experience the trade-offs of AI in the computing continuum and at the edge**:  
- **TinyML training & quantization** → see how models shrink, and what accuracy is lost.  
- **Distributed AI Inference simulator** → explore latency, energy, and accuracy trade-offs under different policies for AI inference execution allocation.  
- **Group challenges** → extend the simulator with new ideas (e.g. topology, devices, energy models, etc).  

You do **not need any hardware**: everything runs on your laptop (or in Google Colab).

Please note that the code was tested on a Mac machine.

---

## 1) Prerequisites
- Python 3.7–3.12
- Git  
- (Optional) Jupyter/Colab for notebooks  
- No GPU required; runs on CPU.  

If you only want to run the **simulator** (recommended baseline), you do *not* need TensorFlow.

---

## 2) Get the code
Clone the repository and move inside:
```
https://github.com/robertmora/distributed-edge-ai-lab
cd distributed-edge-ai-lab
```

---

## 3) Create a virtual environment

**IMPORTANT!** If you already have a virtual environment from another hands-on lab at the summer school, you can reuse it.
In that case, just activate the existing environment and install the requirements (see Step 4).
You can safely skip this step.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 4) Install dependencies

### A) Simulator only (recommended for all participants)
```bash
pip install -r requirements.txt
```
This installs: `streamlit`, `numpy`, `pandas`, `matplotlib`, `networkx`.

### B) Full (if you want to run the TinyML training locally -- Not Recommended!)

You *can* install TensorFlow locally, **but I highly recommend running the training notebook in Google Colab instead**.
This avoids compatibility issues and ensures a smooth experience for everyone.

```bash
pip install tensorflow
```
> On Apple Silicon use:  
> ```
> pip install tensorflow-macos
> ```

⚠️ If TensorFlow installation fails, **skip it** and run the training notebook in **Colab** instead. The simulator does *not* require TensorFlow.

---

## 5) Run the simulator (Streamlit UI)
```
streamlit run app/streamlit_app.py
```
Then open the link shown in the terminal (usually http://localhost:8501).

**In the app you can:**
- Configure devices and RTTs (MCU, EDGE, CLOUD).  
- Generate workloads (CV + LM tasks).  
- Test different policies for AI inference execution (Always Local, Always Cloud, Confidence Threshold, Latency-Aware, Hybrid).  
- Compare trade-offs with plots (Accuracy vs Deadline, Energy vs Accuracy).  
- See a live topology graph with RTT labels.

---

## 6) Notebooks
- `notebooks/01_tinyml_training.ipynb`  
  Train a small CNN on MNIST, quantize to INT8, and measure accuracy.  
- `notebooks/02_simulator_colab.ipynb`  
  Step-by-step simulator walkthrough, Colab-friendly.  
  Lets you modify policies and visualize results line by line.

---

## 7) CLI runner (optional)
For a quick headless run:
```
python scripts/run_simulator.py
```
This prints a comparison table of policies (accuracy, deadlines met, latency, energy).

---

## 8) Troubleshooting
- **Port already in use (8501):**
  ```streamlit run app/streamlit_app.py --server.port 8502
  ```

- **TensorFlow install issues:**  
  Skip local training, use Colab for `01_tinyml_training.ipynb`.  

- **Windows venv activation blocked:**  
  Open PowerShell as Administrator and run:  
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
  
---

## 9) Next Steps (check the shared slides!)
- Run the simulator → observe how policies behave.  
- Try changing workload parameters (LM ratio, deadlines).  
- In group work, extend the simulator with:  
  - richer LM modeling  
  - new accelerators (GPU/TPU)  
  - improved energy models  
  - traffic patterns or new policies
  - ...  

This activtiy tries to mirror **real distributed AI inference research**: start from a baseline, extend it, measure trade-offs.

Have fun experimenting!
