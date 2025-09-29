from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List

# ---- Model registry ----
# Short unique IDs -> full checkpoint base name on GCS
# (all files are under: https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/)
MODEL_REGISTRY: Dict[str, str] = {
    # id     # full checkpoint basename
    "s14h":  "tips_oss_s14_highres_distilled",
    "b14h":  "tips_oss_b14_highres_distilled",
    "l14h":  "tips_oss_l14_highres_distilled",
    "so4h":  "tips_oss_so400m14_highres_largetext_distilled",
    "g14l":  "tips_oss_g14_lowres",
    "g14h":  "tips_oss_g14_highres",
}

TOKENIZER_FILENAME = "tokenizer.model"
TOKENIZER_URL = "https://storage.googleapis.com/tips_data/v1_0/checkpoints/tokenizer.model"

BASE_URL = "https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch"

def _require_wget() -> None:
    from shutil import which
    if which("wget") is None:
        raise EnvironmentError(
            "wget is required but was not found on PATH. "
            "Please install wget or add it to your PATH."
        )

def _wget(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -q for quiet except errors; --show-progress if attached to TTY would be nice,
    # but we keep it simple and quiet here.
    result = subprocess.run(
        ["wget", "-q", "-O", str(dest), url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        # Clean up partial file if any
        if dest.exists() and dest.stat().st_size == 0:
            try:
                dest.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download {url} -> {dest}\n"
            f"wget stderr:\n{result.stderr.strip()}"
        )

def _model_files_for_basename(base: str) -> List[str]:
    return [f"{base}_vision.npz", f"{base}_text.npz"]

def list_models() -> Dict[str, str]:
    return dict(MODEL_REGISTRY)

def ensure_model_files(model_id: str, save_dir: str) -> Dict[str, str]:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_id '{model_id}'. "
            f"Valid options: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
        )

    _require_wget()

    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    # 1) Tokenizer
    tokenizer_path = save_path / TOKENIZER_FILENAME
    if not tokenizer_path.exists():
        _wget(TOKENIZER_URL, tokenizer_path)

    # 2) Model files
    base = MODEL_REGISTRY[model_id]
    required_files = _model_files_for_basename(base)

    out_paths = {"tokenizer": str(tokenizer_path)}

    for fname in required_files:
        local_path = save_path / fname
        if not local_path.exists():
            # Compose the correct URL for this file
            url = f"{BASE_URL}/{fname}"
            _wget(url, local_path)

        # record
        if fname.endswith("_vision.npz"):
            out_paths["vision"] = str(local_path)
        elif fname.endswith("_text.npz"):
            out_paths["text"] = str(local_path)

    return out_paths