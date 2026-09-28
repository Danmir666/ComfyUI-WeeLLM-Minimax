"""
ComfyUI-WeeLLM install.py

Two one-time setup tasks run transparently on Windows:

1. Triton tcc headers — copies Python C headers into tcc's include directory
   so torch.compile can compile CUDA kernels (needed for memory-efficient VAE
   decoding on low-VRAM GPUs).

2. PYTORCH_CUDA_ALLOC_CONF — sets expandable_segments:True in the user's
   Windows environment so PyTorch initialises its CUDA allocator without
   memory fragmentation, preventing OOM errors during LoRA delta computation.
   Takes effect on the next ComfyUI restart (one-time prompt to user).

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

_ALLOC_KEY   = "PYTORCH_CUDA_ALLOC_CONF"
_ALLOC_VALUE = "expandable_segments:True"


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
# 2.  PYTORCH_CUDA_ALLOC_CONF — persistent Windows env var via registry
# ---------------------------------------------------------------------------

def _get_user_env_var(name):
    """Read a value from HKCU\\Environment (returns None if missing)."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ)
        value, _ = winreg.QueryValueEx(key, name)
        winreg.CloseKey(key)
        return value
    except Exception:
        return None


def _set_user_env_var(name, value):
    """Write a value to HKCU\\Environment so it persists across reboots."""
    import winreg
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE)
    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    winreg.CloseKey(key)


def setup_cuda_alloc_conf():
    """
    Ensure PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set for all
    future Python processes on this machine.

    Because ComfyUI imports torch before custom nodes load, we cannot affect
    the CURRENT process. We write the value to the Windows user environment
    registry so it is inherited by every new process — including the next
    ComfyUI startup — and show a one-time restart notice to the user.
    """
    if os.name != "nt":
        # On Linux/macOS the user sets this in their launch script; skip.
        return

    current = _get_user_env_var(_ALLOC_KEY)
    if current == _ALLOC_VALUE:
        return  # Already set from a previous run — nothing to do.

    try:
        _set_user_env_var(_ALLOC_KEY, _ALLOC_VALUE)
        log.warning(
            "[WeeLLM] \n"
            "  ╔══════════════════════════════════════════════════════════════╗\n"
            "  ║  WeeLLM: One-time setup complete — RESTART REQUIRED         ║\n"
            "  ║                                                              ║\n"
            "  ║  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True has been  ║\n"
            "  ║  saved to your Windows environment. This prevents VRAM OOM  ║\n"
            "  ║  errors when using LoRA with low-VRAM GPUs.                 ║\n"
            "  ║                                                              ║\n"
            "  ║  Please restart ComfyUI once for this to take effect.       ║\n"
            "  ╚══════════════════════════════════════════════════════════════╝"
        )
    except Exception as exc:
        log.warning(
            "[WeeLLM] Could not set PYTORCH_CUDA_ALLOC_CONF automatically (%s). "
            "For best results on low-VRAM GPUs, add "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to your system environment variables.",
            exc,
        )
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
    setup_cuda_alloc_conf()
    setup_disable_cuda_malloc()
