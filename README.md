# AI Drug Repurposing System

An AI system that predicts new therapeutic uses for existing drugs using **TxGNN** (Graph Neural Network on a biomedical knowledge graph) and **GPT-OSS 20B via Groq API** (natural language explanations).

**Input:** A disease name
**Output:** Ranked list of existing drugs that could treat it, with biological explanations

## Architecture

```
User Query (disease name)
        |
        v
+------------------+        +------------------------+
|  TxGNN Module    |        |  NLP Module (Groq API) |
|  (Local CPU/GPU) |        |  (GPT-OSS 20B Cloud)   |
|                  |        |                        |
|  Pre-built KG    |        |  Disease context       |
|  123K nodes      |        |  Drug explanations     |
|  8M edges        |        |  Literature evidence   |
+--------+---------+        +-----------+------------+
         |                              |
         v                              v
+------------------------------------------------+
|          Result Fusion & Report Generation      |
|  Ranked drugs + scores + biological rationales  |
+------------------------------------------------+
```

## What's Already Set Up

The following have been installed and verified in the `venv/` virtual environment:

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.9.0 | Installed |
| PyTorch | 2.3.0 (CPU or CUDA) | See [GPU setup](#gtx-1650--windows-gpu-setup) |
| DGL | 2.2.1 | Installed |
| TxGNN | 0.0.3 | Installed |
| Groq SDK | Latest | Installed |
| Streamlit | 1.50.0 | Installed |
| scikit-learn, pandas, matplotlib, etc. | Latest | Installed |

## What's Already Done

- Virtual environment (`venv/`) created with all dependencies installed
- Groq API key configured in `.env`
- TxGNN knowledge graph downloaded (~1.5 GB) to `data/`
- Node name mappings configured (17,080 diseases, 7,957 drugs)
- TxGNN pandas compatibility patched (DataFrame.append -> pd.concat)
- All modules verified working: `python verify_setup.py` passes all checks
- NLP module tested and confirmed working with Groq API

## What YOU Need To Do

### Step 1: Train the GNN Model

Activate the virtual environment first, then train:

```bash
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Quick test (10 epochs, ~5-15 minutes on CPU)
python main.py --train --epochs 10 --split random --disease "Alzheimer disease"

# Full training (500 epochs, ~3-7 hours on CPU)
python main.py --train --epochs 500 --split random --disease "Alzheimer disease"
```

**Training time estimates (CPU):**
| Epochs | Time |
|--------|------|
| 10 | ~5-15 min |
| 50 | ~20-40 min |
| 100 | ~45-90 min |
| 500 | ~3-7 hours |

**Tips if training is slow or runs out of memory:**
- Reduce batch size: Edit `graph_module.py`, change `batch_size=1024` to `512` or `256`
- Reduce model size: Change `n_hid=100` to `n_hid=64`
- Close other GPU-using programs (browsers with hardware acceleration, games)

### Step 2: Run Drug Repurposing

After training, run predictions (loads saved model automatically):

```bash
# Use saved model (no --train flag)
python main.py --disease "Alzheimer disease" --top-k 20 --explain-top 5

# Try other diseases
python main.py --disease "type 2 diabetes" --top-k 15
python main.py --disease "breast cancer" --top-k 20
```

Results are saved to `outputs/repurposing_<disease>_<timestamp>.json`.

### Step 3: Launch Web Interface

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Click **Initialize System** in the sidebar, then enter a disease name.

### Step 4: Run Evaluation (Optional)

```bash
# Test the evaluation module with synthetic data
python eval_module.py
```

For full model evaluation, use the Python API:
```python
from main import DrugRepurposingSystem

system = DrugRepurposingSystem()
system.setup_gnn(split="random", train=False)
results = system.evaluate()
```

### Step 5: Train GraphMask Explainability (Optional)

```python
from main import DrugRepurposingSystem

system = DrugRepurposingSystem()
system.setup_gnn(split="random", train=False)
system.explain_with_graphmask(relation="indication")
```

This takes 10-30 minutes and enables multi-hop path explanations showing which biological pathways connect a drug to a disease.

## Project Structure

```
Drug-Repurposing-project/
|
├── .env.example            # Template for API key config
├── .env                    # Your Groq API key (create this, NEVER commit)
├── .gitignore              # Excludes .env, data/, models/, etc.
├── requirements.txt        # Package list (see install notes inside)
├── requirements-lock.txt   # Exact pinned versions
├── verify_setup.py         # Setup verification script
|
├── graph_module.py         # Module 1: TxGNN GNN (knowledge graph + predictions)
├── nlp_module.py           # Module 2: GPT-OSS 20B via Groq (explanations)
├── explain_module.py       # Module 3: GraphMask + NLP explainability
├── eval_module.py          # Module 4: AUPRC, Hits@K, MRR evaluation
├── main.py                 # Orchestrator (connects all modules, CLI interface)
├── app.py                  # Streamlit web UI
|
├── scripts/                # install-gpu-windows.ps1 — CUDA PyTorch + DGL
├── venv/                   # Python virtual environment (already created)
├── data/                   # TxGNN knowledge graph (auto-downloaded, ~1.5 GB)
├── models/                 # Saved model checkpoints
├── outputs/                # Results, reports, plots
├── cache/                  # Cached NLP API responses
├── logs/                   # Training logs
└── plan/                   # PRD document
```

## GTX 1650 / Windows GPU setup

[`graph_module.py`](graph_module.py) uses **`cuda:0` when CUDA is available on Linux or WSL2**. On **native Windows**, TxGNN + DGL still run on **CPU**: the published DGL Windows CUDA wheels raise errors such as `COOToCSR does not support cuda device` on heterogeneous graphs during `model_initialize`. PyTorch CUDA remains useful for other projects and keeps you ready for **WSL2/Ubuntu** GPU training with the same `venv` pattern there.

If you see **`[GNN] Device: cpu`** and **no** GPU message, install CUDA PyTorch (below). If you see **`CUDA GPU: ...`** but **`Device: cpu`** on Windows, that is intentional until you move the repo to WSL2/Linux.

Official compatibility matrices:

- **PyTorch + CUDA wheels:** [PyTorch Get Started](https://pytorch.org/get-started/locally/) (pick Windows, Pip, CUDA 12.1 or 11.8).
- **DGL vs PyTorch/CUDA:** [DGL Get Started](https://www.dgl.ai/pages/start.html) (PyTorch 2.3 supports CUDA 11.8 and 12.1 on Windows).

### One-shot install (recommended)

From the **repository root**, in **PowerShell** (same `venv` you already use):

```powershell
# Allow scripts once if needed (user scope):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.\venv\Scripts\Activate.ps1
.\scripts\install-gpu-windows.ps1
```

Default is **CUDA 12.1** wheels (`-Cuda 121`). If `nvidia-smi` shows an older driver / CUDA **11.x**, use:

```powershell
.\scripts\install-gpu-windows.ps1 -Cuda 118
```

Then confirm:

```powershell
python verify_setup.py
```

On **WSL2/Linux**, you want **`CUDA available: True`** and **`[GNN] Device: cuda:0`** when you train. On **native Windows**, the GNN stays on **CPU** even if CUDA PyTorch is installed (see above).

### WSL2 (Ubuntu) — use your GPU for TxGNN

Do GNN training inside Linux so DGL’s **heterogeneous-graph CUDA** path works. Your **Windows NVIDIA driver** must support WSL (recent **Game Ready** or **Studio** drivers usually do). You do **not** need a separate “CUDA installer” on Windows for WSL; inside Ubuntu, PyTorch ships its own CUDA runtime via pip wheels.

1. **Windows:** Update WSL and the GPU driver, then install Ubuntu if needed:
   ```powershell
   wsl --update
   wsl --install -d Ubuntu
   ```
2. **Ubuntu (WSL):** Confirm the GPU is visible:
   ```bash
   nvidia-smi
   ```
   If this fails, fix the Windows driver / WSL update first ([NVIDIA WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)).

3. **Project files:** Either work on the Windows drive (slower I/O) or copy the repo into your Linux home (faster training):
   ```bash
   # Option A — use Windows checkout (path may differ)
   cd /mnt/d/02-professional/00-Work/Drug-Repurposing-project

   # Option B — copy once (recommended for heavy training)
   # cp -r /mnt/d/02-professional/00-Work/Drug-Repurposing-project ~/Drug-Repurposing-project
   # cd ~/Drug-Repurposing-project
   ```

4. **Python in WSL:** On **Ubuntu 24.04**, `apt` often has only **`python3` (3.12)** — packages like `python3.10` / `python3.11` may be missing. That is fine: use **`python3 -m venv`** and **DGL 2.3.0** in step 6.
   ```bash
   sudo apt update && sudo apt install -y python3 python3-venv python3-pip
   ```
   If you need an older Python, add [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa):  
   `sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update && sudo apt install -y python3.10 python3.10-venv` (or `python3.11`).

5. **New venv inside WSL** (separate from Windows `venv`):
   ```bash
   # Use a binary that exists: python3, python3.10, or python3.11 (not `python -3.11`).
   python3 -m venv venv       # OK on Ubuntu 24.04 (3.12) + DGL 2.3.0
   # or: python3.10 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```
   If you see **No such file or directory** for `venv/bin/python`, run `deactivate`, then `rm -rf venv`, then recreate the venv.

6. **CUDA PyTorch + DGL (Linux wheels)** — install **PyTorch first**, then DGL. Example **CUDA 12.1**:
   ```bash
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
   # DGL 2.3.0: better Graphbolt + PyTorch 2.3 pairing (use on Python 3.12).
   pip install dgl==2.3.0 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
   # If you use Python 3.10 and prefer the older pin: dgl==2.2.1 with the same -f URL.
   pip install "numpy<2" torchdata==0.7.1 pyyaml
   pip install git+https://github.com/mims-harvard/TxGNN.git
   pip install groq python-dotenv streamlit scikit-learn matplotlib pandas tqdm requests
   ```
   If you see **`Cannot load Graphbolt C++ library`**, upgrade DGL as above or recreate the venv with **Python 3.10** and `dgl==2.2.1`.

7. **Config & data:** Copy `.env` from Windows (Groq key) into this folder, or create it again from [`.env.example`](.env.example). Reuse the KG under `data/` (copy or symlink from Windows) so you do not re-download ~1.5 GB:
   ```bash
   # Example: symlink to existing Windows data folder
   # rm -rf data && ln -s /mnt/d/02-professional/00-Work/Drug-Repurposing-project/data ./data
   ```

8. **Verify and train:**
   ```bash
   python verify_setup.py
   python main.py --train --epochs 10 --split random --disease "Alzheimer disease"
   ```
   You should see **`[GNN] Device: cuda:0`**.

   **WSL note:** If `import dgl` fails with **`libnvrtc.so.12: cannot open shared object file`**, the project prepends PyTorch’s `lib` dir via [`torch_cuda_ld_path.py`](torch_cuda_ld_path.py) when you use `main.py` / `verify_setup.py`. For a plain Python one-liner, run `source scripts/wsl-cuda-lib-path.sh` first (after `source venv/bin/activate`).

### Same steps as manual `pip` (no script)

Activate `venv`, then (example **CUDA 12.1**):

```powershell
pip uninstall -y torch torchvision torchaudio dgl
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install "numpy<2" torchdata==0.7.1 pyyaml
```

For **CUDA 11.8**, replace `cu121` with `cu118` in both URLs.

**GTX 1650 (4 GB):** keep `batch_size=1024` in `graph_module.train()`; if you hit OOM, use `512` or `256`.

## Reinstalling From Scratch

If you need to recreate the environment:

```bash
# 1. Create venv with Python 3.9+
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch CPU
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install DGL
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html

# 5. Fix DGL dependencies
pip install "numpy<2" torchdata==0.7.1 pyyaml

# 6. Install TxGNN from GitHub
pip install git+https://github.com/mims-harvard/TxGNN.git

# 7. Install remaining packages
pip install groq python-dotenv streamlit scikit-learn matplotlib pandas tqdm requests

# 8. Verify
python verify_setup.py
```

After a CPU reinstall, switch to GPU with `.\scripts\install-gpu-windows.ps1` (see [GTX 1650 / Windows GPU setup](#gtx-1650--windows-gpu-setup)).

## API Cost

| Task | Cost per Call | Calls per Disease |
|------|--------------|-------------------|
| Disease context | ~$0.0003 | 1 |
| Drug explanation | ~$0.0004 | 5-10 |
| Full report | ~$0.0015 | 1 |
| **Total per disease** | **~$0.006** | |

You can analyze ~1,000 diseases for about $6.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `batch_size` to 512 or 256 in `graph_module.py` |
| `COOToCSR does not support cuda device` (Windows) | Expected with DGL+TxGNN on Windows GPU; this repo forces CPU for GNN on Windows. Use WSL2/Linux for GPU, or stay on CPU. |
| `Cannot load Graphbolt C++ library` / `libnvrtc.so.12: cannot open shared object file` (WSL/Linux) | PyTorch ships CUDA libs under `site-packages/torch/lib/`, but the loader needs that on **`LD_LIBRARY_PATH`**. This repo runs [`torch_cuda_ld_path.apply()`](torch_cuda_ld_path.py) before importing DGL in `main.py`, `graph_module.py`, `verify_setup.py`, etc. For ad-hoc `python -c "import dgl"`, run `source scripts/wsl-cuda-lib-path.sh` after activating `venv`. Also: `sudo apt install -y libgomp1`. See [dmlc/dgl#7450](https://github.com/dmlc/dgl/issues/7450). |
| `DGL import error` | Match wheels to PyTorch: `torch==2.3.0` + `dgl==2.3.0` or `dgl==2.2.1` from `torch-2.3/cu121` (or `cu118`) find-links. |
| `TxGNN pip install fails` | Install from GitHub: `pip install git+https://github.com/mims-harvard/TxGNN.git` |
| `Disease not found` | Run `python main.py --discover` to see available disease names |
| `Groq API rate limit` | Built-in retry with backoff. Reduce `max_drugs` in batch_explain() |
| `numpy` version error | Run `pip install "numpy<2"` |
| KG download fails | Download manually from https://doi.org/10.7910/DVN/IXA7BM |

## References

- **TxGNN:** Huang et al., "A foundation model for clinically actionable drug repurposing" (Nature Medicine, 2024)
- **Knowledge Graph:** Harvard Dataverse (17K diseases, 8K drugs, 8M edges)
- **GPT-OSS 20B:** OpenAI open-weight model, served via Groq LPU inference
