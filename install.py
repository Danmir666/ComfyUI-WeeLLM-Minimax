"""
ComfyUI-WeeLLM install.py

Two one-time setup tasks run transparently on Windows:

1. Triton tcc headers — copies Python C headers into tcc's include directory
   so torch.compile can compile CUDA kernels.

2. Disable cudaMallocAsync via a site-packages .pth file.

Both are no-ops on Linux / macOS and on every run after the first.
"""

import os
import sys
import shutil
import logging
import zipfile
import urllib.request
import tempfile
from pathlib import Path

log = logging.getLogger("ComfyUI-WeeLLM")

# ---------------------------------------------------------------------------
# 1.  Triton tcc header fix
# ---------------------------------------------------------------------------

def _find_tcc_include():
    for path in sys.path:
        candidate = Path(path) / "triton" / "runtime" / "tcc" / "include"
        if candidate.is_dir():
            return candidate
    return None


def _python_headers_present(tcc_include):
    return (tcc_include / "Python.h").exists()


def _install_headers_from_existing(tcc_include):
    local_include = Path(sys.base_prefix) / "Include"
    if (local_include / "Python.h").exists():
        shutil.copytree(str(local_include), str(tcc_include), dirs_exist_ok=True)
        log.info("[WeeLLM] Python headers copied from local Python Include dir.")
        return True
    return False


def _fetch_and_install_headers(tcc_include):
    major = sys.version_info.major
    minor = sys.version_info.minor
    url = f"https://www.nuget.org/api/v2/package/python/{major}.{minor}"
    log.info("[WeeLLM] Downloading Python %s.%s headers from NuGet ...", major, minor)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "python.zip")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [m for m in zf.namelist() if m.startswith("tools/include/")]
                if not members:
                    log.warning("[WeeLLM] NuGet package had no include/ directory.")
                    return False
                zf.extractall(tmp, members=members)
            src = Path(tmp) / "tools" / "include"
            shutil.copytree(str(src), str(tcc_include), dirs_exist_ok=True)
            log.info("[WeeLLM] Python headers installed into tcc include dir.")
            return True
    except Exception as exc:
        log.warning("[WeeLLM] Could not fetch Python headers: %s", exc)
        return False


def setup_triton_headers():
    if os.name != "nt":
        return
    tcc_include = _find_tcc_include()
    if tcc_include is None:
        return
    if _python_headers_present(tcc_include):
        return
    log.info("[WeeLLM] Triton tcc include dir is missing Python.h -- fixing automatically ...")
    ok = _install_headers_from_existing(tcc_include) or _fetch_and_install_headers(tcc_include)
    if ok:
        log.info("[WeeLLM] Triton header setup complete. torch.compile will work correctly.")
    else:
        log.warning("[WeeLLM] Could not install Python headers for Triton. torch.compile may fall back to eager mode.")


# ---------------------------------------------------------------------------
# 3.  Disable cudaMallocAsync via Python .pth file
# ---------------------------------------------------------------------------

def setup_disable_cuda_malloc():
    """
    On Windows, ComfyUI enables cudaMallocAsync (if torch >= 2.0).
    This conflicts with torch.cuda.empty_cache() and Triton JIT compilation.
    To disable it without requiring the user to edit run_nvidia_gpu.bat, we place
    a .pth file in the Python environment's site-packages that automatically appends
    '--disable-cuda-malloc' to sys.argv when main.py starts.
    """
    import site
    site_packages = site.getsitepackages()
    if not site_packages:
        return
    
    # Usually the first or last site-packages is fine, we pick the first one that exists
    target_dir = None
    for sp in site_packages:
        if os.path.exists(sp):
            target_dir = sp
            break
            
    if not target_dir:
        return

    pth_path = os.path.join(target_dir, "zz_weellm_init.pth")
    # Only append if main.py is the script being run (to avoid breaking pip or other tools)
    pth_content = "import sys, os; sys.argv.append('--disable-cuda-malloc') if '--disable-cuda-malloc' not in sys.argv and getattr(sys, 'argv', [''])[0].endswith('main.py') else None\n"
    
    try:
        if os.path.exists(pth_path):
            with open(pth_path, "r") as f:
                if f.read() == pth_content:
                    return
        with open(pth_path, "w") as f:
            f.write(pth_content)
        log.info("[WeeLLM] Added --disable-cuda-malloc automation via site-packages.")
    except Exception as exc:
        log.warning("[WeeLLM] Could not automate --disable-cuda-malloc: %s", exc)

if __name__ == "__main__":
    setup_triton_headers()
    setup_disable_cuda_malloc()
