"""Run this to verify your entire setup is correct."""

import torch_cuda_ld_path

torch_cuda_ld_path.apply()

import sys

print("=" * 60)
print("SETUP VERIFICATION")
print("=" * 60)

errors = []

# 1. Python version
print(f"\n[1] Python: {sys.version}")
if sys.version_info[:2] >= (3, 8):
    print("    OK - Python 3.8+ confirmed")
else:
    errors.append("Need Python 3.8+")
    print("    ERROR: Need Python 3.8+")

# 2. PyTorch
try:
    import torch
    print(f"\n[2] PyTorch: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    VRAM: {vram:.1f} GB")
        print("    OK - GPU ready")
    else:
        print("    INFO - No GPU detected, TxGNN will run on CPU (slower but works)")
except ImportError as e:
    errors.append(f"PyTorch not installed: {e}")
    print(f"\n[2] PyTorch: NOT INSTALLED - {e}")

# 3. DGL
try:
    import dgl
    print(f"\n[3] DGL: {dgl.__version__}")
    print("    OK - DGL installed")
except ImportError as e:
    errors.append(f"DGL not installed: {e}")
    print(f"\n[3] DGL: NOT INSTALLED - {e}")
    if "Graphbolt" in str(e) or "graphbolt" in str(e).lower():
        print("    Graphbolt .so failed to load (often libnvrtc / LD_LIBRARY_PATH on WSL).")
        print("    This repo calls torch_cuda_ld_path.apply() before importing DGL in main.py etc.")
        print("    For a raw shell: source scripts/wsl-cuda-lib-path.sh")
        print("    Also try: sudo apt install -y libgomp1")
        print("    Reinstall DGL: pip install --no-cache-dir dgl==2.3.0 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html")

# 4. TxGNN
try:
    from txgnn import TxData, TxGNN, TxEval
    print("\n[4] TxGNN: imported successfully")
    print("    OK - TxGNN ready")
except ImportError as e:
    errors.append(f"TxGNN not installed: {e}")
    print(f"\n[4] TxGNN: IMPORT ERROR - {e}")

# 5. Groq SDK
try:
    from groq import Groq
    print("\n[5] Groq SDK: imported successfully")
    print("    OK - Groq SDK ready")
except ImportError as e:
    errors.append(f"Groq SDK not installed: {e}")
    print(f"\n[5] Groq SDK: NOT INSTALLED - {e}")

# 6. Groq API Key
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if key and key != "your_actual_api_key_here" and len(key) > 10:
        print(f"\n[6] Groq API Key: {'*' * 20}...{key[-4:]}")
        print("    OK - API key loaded")
    else:
        errors.append("Groq API key not configured")
        print("\n[6] Groq API Key: NOT CONFIGURED")
        print("    Copy .env.example to .env and add your key from console.groq.com")
except ImportError:
    errors.append("python-dotenv not installed")
    print("\n[6] python-dotenv: NOT INSTALLED")

# 7. Other libraries
try:
    import sklearn
    import matplotlib
    import pandas
    import numpy
    import streamlit
    print(f"\n[7] Other libs: sklearn={sklearn.__version__}, "
          f"pandas={pandas.__version__}, streamlit={streamlit.__version__}")
    print("    OK - All libraries installed")
except ImportError as e:
    errors.append(f"Missing library: {e}")
    print(f"\n[7] Missing library: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"SETUP INCOMPLETE - {len(errors)} issue(s) found:")
    for err in errors:
        print(f"  - {err}")
    print("\nSee README.md for installation instructions.")
else:
    print("SETUP COMPLETE - You're ready to build!")
print("=" * 60)
