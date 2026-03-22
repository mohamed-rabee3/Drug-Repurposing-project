"""
Prepend PyTorch's bundled CUDA libraries to LD_LIBRARY_PATH on Linux,
and pre-load critical shared objects so DGL Graphbolt can find them.

DGL Graphbolt's CUDA .so depends on libnvrtc.so.*; pip-installed PyTorch ships
those either in site-packages/torch/lib/ (older builds) or in separate
nvidia-* packages under site-packages/nvidia/*/lib/ (PyTorch 2.x + CUDA 12).

Setting LD_LIBRARY_PATH mid-process does NOT affect ctypes.CDLL / dlopen on
Linux, so we also explicitly pre-load the needed .so files via ctypes.
"""

import ctypes
import glob
import os
import sys
from typing import List, Optional


def _find_nvidia_lib_dirs(site_packages: str) -> List[str]:
    """Find all nvidia/*/lib/ directories under site-packages."""
    nvidia_dir = os.path.join(site_packages, "nvidia")
    if not os.path.isdir(nvidia_dir):
        return []
    dirs = []
    for sub in os.listdir(nvidia_dir):
        lib_dir = os.path.join(nvidia_dir, sub, "lib")
        if os.path.isdir(lib_dir):
            dirs.append(lib_dir)
    return dirs


def apply() -> None:
    if not sys.platform.startswith("linux"):
        return

    # Locate site-packages and torch/lib
    site_packages: Optional[str] = None
    torch_lib: Optional[str] = None

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        pattern = os.path.join(venv, "lib", "python*", "site-packages")
        for sp in sorted(glob.glob(pattern)):
            if os.path.isdir(sp):
                site_packages = sp
                tl = os.path.join(sp, "torch", "lib")
                if os.path.isdir(tl):
                    torch_lib = tl
                break

    if torch_lib is None:
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if not os.path.isdir(torch_lib):
                torch_lib = None
            # Derive site-packages from torch location
            if site_packages is None:
                site_packages = os.path.dirname(os.path.dirname(torch.__file__))
        except ImportError:
            return

    if not torch_lib:
        return

    # Collect all lib dirs: torch/lib + nvidia/*/lib/
    lib_dirs = [torch_lib]
    if site_packages:
        lib_dirs.extend(_find_nvidia_lib_dirs(site_packages))

    # 1. Set LD_LIBRARY_PATH for any child processes
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    prev_parts = prev.split(os.pathsep) if prev else []
    new_parts = [d for d in lib_dirs if d not in prev_parts]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            new_parts + ([prev] if prev else [])
        )

    # 2. Pre-load critical CUDA .so files so dlopen can find them
    #    when DGL Graphbolt loads its own .so
    cuda_libs = [
        "libnvrtc.so*",
        "libnvrtc-builtins.so*",
        "libcudart.so*",
        "libcublas.so*",
        "libcublasLt.so*",
        "libcusparse.so*",
        "libcusolver.so*",
        "libcufft.so*",
        "libcurand.so*",
        "libnvJitLink.so*",
    ]
    for lib_dir in lib_dirs:
        for lib_pattern in cuda_libs:
            matches = sorted(glob.glob(os.path.join(lib_dir, lib_pattern)))
            for lib_path in matches:
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass  # Not all libs exist in every PyTorch build
