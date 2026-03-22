# Reinstall PyTorch + DGL with NVIDIA CUDA support (Windows).
# Tested stack: torch 2.3.x + DGL 2.2.1 - matches this repo's CPU install path.
#
# Prerequisites: recent NVIDIA driver (run: nvidia-smi). GTX 1650 is supported.
# Pick -Cuda 121 if nvidia-smi shows CUDA Version 12.x, else use 118 for older drivers.
#
# Usage (from repo root, PowerShell):
#   .\scripts\install-gpu-windows.ps1
#   .\scripts\install-gpu-windows.ps1 -Cuda 118

param(
    [ValidateSet("118", "121")]
    [string]$Cuda = "121"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$VenvActivate = Join-Path $RepoRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "venv not found at $RepoRoot\venv. Create it first: python -m venv venv"
}

. $VenvActivate

Write-Host "`n=== [1/5] Removing CPU PyTorch / existing DGL ===" -ForegroundColor Cyan
$ErrorActionPreference = "Continue"
pip uninstall -y torch torchvision torchaudio dgl
$ErrorActionPreference = "Stop"

$TorchIndex = if ($Cuda -eq "121") {
    "https://download.pytorch.org/whl/cu121"
} else {
    "https://download.pytorch.org/whl/cu118"
}

$DglFindLinks = if ($Cuda -eq "121") {
    "https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html"
} else {
    "https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html"
}

Write-Host "`n=== [2/5] Installing PyTorch 2.3.0 + CUDA $Cuda ===" -ForegroundColor Cyan
pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url $TorchIndex

Write-Host "`n=== [3/5] Installing DGL 2.2.1 (CUDA $Cuda, PyTorch 2.3 repo) ===" -ForegroundColor Cyan
pip install dgl==2.2.1 -f $DglFindLinks

Write-Host "`n=== [4/5] Re-applying pinned deps from README ===" -ForegroundColor Cyan
pip install "numpy<2" torchdata==0.7.1 pyyaml

Write-Host "`n=== [5/5] Quick CUDA check ===" -ForegroundColor Cyan
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU - check driver / CUDA wheel choice')"

Write-Host "`nRun: python verify_setup.py" -ForegroundColor Green
