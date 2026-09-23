"""Model registry and constants for Stable-one-shot.

The six released adapters correspond one-to-one with Table 1 of the paper. Each is named by its
backbone and rank, as in the paper (S = SA3-Small-Music, M = SA3-Medium).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ASSETS = Path(__file__).resolve().parent / "assets"

# --- audio / canvas constants (Section 3.2) ----------------------------------
SR = 44100                    # SA3 operates at 44.1 kHz stereo
LOOP_SECONDS = 4.0            # input drum loop is cropped / padded to 4 s
CANVAS_FRAMES = 256           # the DiT's minimum sequence length
GAP_FRAMES = 3                # ~0.28 s silence gap between loop and one-shot
ONESHOT_FRAMES = 11           # ~1 s one-shot at the 10.77 Hz latent frame rate
SAMPLING_STEPS = 50           # rectified-flow Euler solver steps at inference

SILENCE_LATENT = ASSETS / "same_silence_latent.npy"

# (output id, prompt label). The prompt label is part of the trained conditioning -- see PROMPT.
INSTRUMENTS = [("kick", "kick"), ("snare", "snare"), ("hihat", "hi-hat")]

# Text-pathway template used for instrument selection (Section 3.2). The released adapters expect
# exactly this string, so it must not be altered: changing it moves the model off the conditioning
# it was adapted under.
PROMPT = "isolated {label} drum one-shot"

# --- backbones ---------------------------------------------------------------
# Each backbone is a separate download; see the README in each folder.
BACKBONES = {
    "small":  {"dir": MODELS / "sa3-base",        "hf": "stabilityai/stable-audio-3-small-music"},
    "medium": {"dir": MODELS / "sa3-base-medium", "hf": "stabilityai/stable-audio-3-medium"},
}

# --- released adapters (Table 1) ---------------------------------------------
ADAPTERS = {
    "S-r4":  {"backbone": "small",  "rank": 4,  "params": "2.6M"},
    "S-r8":  {"backbone": "small",  "rank": 8,  "params": "5.1M"},
    "S-r16": {"backbone": "small",  "rank": 16, "params": "10.3M"},
    "S-r32": {"backbone": "small",  "rank": 32, "params": "20.6M"},
    "M-r4":  {"backbone": "medium", "rank": 4,  "params": "5.2M"},
    "M-r16": {"backbone": "medium", "rank": 16, "params": "20.7M"},
}
DEFAULT_ADAPTER = "S-r4"      # best MSS per trainable parameter (Table 1)

# --- autoencoder -------------------------------------------------------------
# The paper uses SAME-L (the medium autoencoder) for all encoding and decoding, including for the
# S-* adapters, which operate in the SAME-L latent space. SAME-S is distilled from SAME-L and
# shares that latent space, so "same-s" also works and avoids the medium download -- but it is not
# the configuration the paper reports.
AUTOENCODERS = ("same-l", "same-s")
DEFAULT_AUTOENCODER = "same-l"
SAME_L_BACKBONE = "medium"    # SAME-L ships inside the SA3-Medium checkpoint


def adapter_path(name):
    return MODELS / "lora" / f"{name}.safetensors"


def resolve(name=DEFAULT_ADAPTER, autoencoder=DEFAULT_AUTOENCODER):
    """Resolve an adapter name to everything needed to run it.

    Returns the adapter path and rank, the backbone's config/checkpoint paths, and -- when the
    small backbone must borrow SAME-L -- where to read that autoencoder from. Environment
    variables SA3_REPO_CFG / SA3_BASE_CKPT override the backbone location, and
    SA3_SAME_L_CFG / SA3_SAME_L_CKPT the SAME-L source, which is useful when a base model already
    lives elsewhere on disk.
    """
    if name not in ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; expected one of {list(ADAPTERS)}")
    if autoencoder not in AUTOENCODERS:
        raise ValueError(f"unknown autoencoder {autoencoder!r}; expected one of {list(AUTOENCODERS)}")
    spec = ADAPTERS[name]
    base = BACKBONES[spec["backbone"]]["dir"]
    same_l_dir = BACKBONES[SAME_L_BACKBONE]["dir"]
    # The medium backbone already carries SAME-L, so only the small one ever needs the swap.
    swap = autoencoder == "same-l" and spec["backbone"] != SAME_L_BACKBONE
    return {
        "name": name,
        "rank": spec["rank"],
        "backbone": spec["backbone"],
        "adapter": adapter_path(name),
        "repo_cfg": Path(os.environ.get("SA3_REPO_CFG", base / "model_config.json")),
        "base_ckpt": Path(os.environ.get("SA3_BASE_CKPT", base / "model.safetensors")),
        "autoencoder": "SAME-L" if (autoencoder == "same-l") else "SAME-S",
        "swap_same_l": swap,
        "same_l_cfg": Path(os.environ.get("SA3_SAME_L_CFG", same_l_dir / "model_config.json")),
        "same_l_ckpt": Path(os.environ.get("SA3_SAME_L_CKPT", same_l_dir / "model.safetensors")),
    }


def _download_hint(backbone, missing):
    folder = BACKBONES[backbone]["dir"]
    hf = BACKBONES[backbone]["hf"]
    return ("\n  " + "\n  ".join(str(p) for p in missing)
            + f"\n\nDownload model.safetensors and model_config.json from "
              f"https://huggingface.co/{hf}\ninto {folder}  (see that folder's README).")


def check(spec):
    """Fail early with an actionable message rather than deep inside the model loader."""
    if not spec["adapter"].exists():
        raise FileNotFoundError(f"adapter missing: {spec['adapter']}")
    missing = [p for p in (spec["repo_cfg"], spec["base_ckpt"]) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{spec['name']} needs the SA3-{spec['backbone']} base model, but these are missing:"
            + _download_hint(spec["backbone"], missing)
            + "\nAlternatively set SA3_REPO_CFG / SA3_BASE_CKPT.")
    if spec["swap_same_l"]:
        missing = [p for p in (spec["same_l_cfg"], spec["same_l_ckpt"]) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{spec['name']} is configured to use SAME-L, which ships inside the SA3-Medium "
                f"checkpoint, but these are missing:"
                + _download_hint(SAME_L_BACKBONE, missing)
                + "\nAlternatively set SA3_SAME_L_CFG / SA3_SAME_L_CKPT, or pass "
                  "--autoencoder same-s to use the small backbone's own autoencoder instead "
                  "(no extra download, but not the paper's configuration).")


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
