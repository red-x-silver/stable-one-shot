"""Loading the Stable Audio 3 backbone and applying the trained LoRA adapter.

The bodies below are the inference subset of the research implementation, kept unchanged so the
released adapters load and behave exactly as they did in the paper's experiments. In particular the
LoRA injection (which layer types are wrapped, and the ``seconds_total`` exclusion) must match how
the adapters were produced, or the checkpoint keys will not line up.
"""
import json
import struct

import numpy as np
import torch

from . import config as C


def read_safetensors_all(path, prefix=None, strip_prefix=False):
    """Read tensors via plain file reads (avoids mmap/commit issues on the 9 GB medium file).

    `prefix` reads only the keys under it -- used to pull the SAME-L autoencoder (3.4 GB) out of
    the medium checkpoint without materialising the whole 9.2 GB. `strip_prefix` removes it from
    the returned keys so the result loads straight into the submodule.
    """
    npd = {'F32': np.float32, 'F16': np.float16, 'F64': np.float64,
           'I64': np.int64, 'I32': np.int32, 'U8': np.uint8, 'BOOL': np.bool_}
    tdt = {'F32': torch.float32, 'F16': torch.float16, 'F64': torch.float64, 'BF16': torch.bfloat16,
           'I64': torch.int64, 'I32': torch.int32, 'U8': torch.uint8, 'BOOL': torch.bool}
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        start = 8 + n
        for k, m in h.items():
            if k == '__metadata__':
                continue
            if prefix is not None and not k.startswith(prefix):
                continue
            key = k[len(prefix):] if (prefix is not None and strip_prefix) else k
            a, b = m['data_offsets']
            if b - a == 0:
                out[key] = torch.empty(m['shape'], dtype=tdt[m['dtype']])
                continue
            if m['dtype'] == 'BF16':
                arr = np.fromfile(path, dtype=np.uint16, count=(b - a) // 2, offset=start + a)
                out[key] = torch.from_numpy(arr).view(torch.bfloat16).reshape(m['shape'])
            else:
                dt = npd[m['dtype']]
                arr = np.fromfile(path, dtype=dt, count=(b - a) // np.dtype(dt).itemsize,
                                  offset=start + a)
                out[key] = torch.from_numpy(arr).reshape(m['shape'])
    return out


def load_base(repo_cfg, base_ckpt, device=None, verbose=True):
    """Instantiate the frozen SA3 backbone (DiT + SAME autoencoder) from its config and weights."""
    from stable_audio_tools.models.factory import create_model_from_config
    device = device or C.device()
    cfg = json.load(open(repo_cfg))
    model = create_model_from_config(cfg)
    sd = read_safetensors_all(str(base_ckpt))
    r = model.load_state_dict(sd, strict=False)
    # The DiT ('model.*') and the inpaint/seconds projections must all load. The T5Gemma text
    # encoder is fetched from the HuggingFace cache by stable_audio_tools, so it is absent here.
    dit_missing = [k for k in r.missing_keys if k.startswith("model.")]
    if dit_missing:
        raise RuntimeError(f"DiT weights incomplete: {len(dit_missing)} missing, "
                           f"e.g. {dit_missing[:3]}")
    if verbose:
        print(f"[base] {len(sd)} tensors loaded | missing {len(r.missing_keys)} "
              f"(t5gemma from HF cache), unexpected {len(r.unexpected_keys)}", flush=True)
    del sd
    return model.to(device).eval(), cfg


def load_same_l(repo_cfg, base_ckpt, device=None, verbose=True):
    """Build the SAME-L autoencoder alone, from the SA3-Medium config and checkpoint.

    Only the ``pretransform.*`` weights are read (3.4 GB of the 9.2 GB file) and only the
    autoencoder is instantiated, so this does not pay for the 2.3 B-parameter DiT.
    """
    from stable_audio_tools.models.factory import create_pretransform_from_config
    device = device or C.device()
    cfg = json.load(open(repo_cfg))
    pre = create_pretransform_from_config(cfg["model"]["pretransform"], cfg["sample_rate"])
    sd = read_safetensors_all(str(base_ckpt), prefix="pretransform.", strip_prefix=True)
    if not sd:
        raise RuntimeError(f"no 'pretransform.*' weights found in {base_ckpt}")
    # AutoencoderPretransform overrides load_state_dict to delegate to its inner model and returns
    # None, so call nn.Module's implementation directly to keep the missing/unexpected report.
    missing, unexpected = torch.nn.Module.load_state_dict(pre, sd, strict=False)
    if missing:
        raise RuntimeError(f"SAME-L weights incomplete: {len(missing)} missing, e.g. {missing[:3]}")
    if verbose:
        print(f"[same-l] autoencoder loaded ({len(sd)} tensors, "
              f"{sum(p.numel() for p in pre.parameters()) / 1e6:.0f} M params)", flush=True)
    del sd
    return pre.to(device).eval()


def use_same_l(model, repo_cfg, base_ckpt, device=None, verbose=True):
    """Replace a backbone's own autoencoder with SAME-L.

    SAME-S is distilled from SAME-L and shares its latent space, so the DiT is unaffected; this
    only changes which autoencoder encodes the input loop and decodes the generated one-shot.
    """
    same_l = load_same_l(repo_cfg, base_ckpt, device=device, verbose=verbose)
    if same_l.downsampling_ratio != model.pretransform.downsampling_ratio:
        raise RuntimeError(
            f"autoencoder mismatch: SAME-L downsamples by {same_l.downsampling_ratio} but the "
            f"backbone expects {model.pretransform.downsampling_ratio}")
    same_l.requires_grad_(False)
    model.pretransform = same_l
    return model


def add_lora(model, rank, alpha=None, verbose=True):
    """Inject LoRA parametrisations into the frozen backbone's linear/conv projections.

    Must mirror the training-time injection exactly: same layer types, same ``seconds_total``
    exclusion, alpha defaulting to the rank.
    """
    from functools import partial
    from stable_audio_tools.models.lora import add_lora as _add_lora, get_lora_params, \
        LoRAParametrization
    alpha = rank if alpha is None else alpha
    adapter, exclude = "lora", ["seconds_total"]
    lora_cfg = {
        torch.nn.Linear: {"weight": partial(LoRAParametrization.from_linear, rank=rank,
                                            lora_alpha=alpha, adapter_type=adapter)},
        torch.nn.Conv1d: {"weight": partial(LoRAParametrization.from_conv1d, rank=rank,
                                            lora_alpha=alpha, adapter_type=adapter)},
    }
    model.model.requires_grad_(False)
    model.conditioner.requires_grad_(False)
    if model.pretransform is not None:
        model.pretransform.requires_grad_(False)
    _add_lora(model.model, lora_cfg, include=None, exclude=exclude)
    _add_lora(model.conditioner, lora_cfg, include=None, exclude=exclude)
    if verbose:
        n = sum(p.numel() for p in [*get_lora_params(model.model),
                                    *get_lora_params(model.conditioner)])
        print(f"[lora] rank={rank} alpha={alpha} | adapter parameters: {n:,}", flush=True)
    return model


def load_adapter(model, path, verbose=True):
    """Load a released adapter's weights into the LoRA parametrisations added by `add_lora`."""
    from safetensors.torch import load_file
    sd = load_file(str(path))
    r1 = model.model.load_state_dict(sd, strict=False)
    r2 = model.conditioner.load_state_dict(sd, strict=False)
    unmatched = [k for k in sd if k in r1.unexpected_keys and k in r2.unexpected_keys]
    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} adapter tensors did not match the model, e.g. {unmatched[:3]}. "
            f"This usually means the rank or the backbone is wrong for this adapter.")
    if verbose:
        print(f"[lora] loaded {len(sd)} adapter tensors from {path.name}", flush=True)
    return model


def load(adapter="S-r4", device=None, autoencoder=C.DEFAULT_AUTOENCODER, verbose=True):
    """Load a released model end to end: backbone + autoencoder + adapter, ready for extraction.

    `autoencoder` is "same-l" (the paper's configuration: SAME-L encodes and decodes for every
    model) or "same-s" (use whichever autoencoder ships with the loaded backbone). It only has an
    effect on the S-* adapters, since the medium backbone already carries SAME-L.
    """
    spec = C.resolve(adapter, autoencoder=autoencoder)
    C.check(spec)
    device = device or C.device()
    if verbose:
        print(f"[model] {spec['name']} (SA3-{spec['backbone']}, rank {spec['rank']}) on {device} | "
              f"autoencoder: {spec['autoencoder']}", flush=True)
    model, _cfg = load_base(spec["repo_cfg"], spec["base_ckpt"], device=device, verbose=verbose)
    if spec["swap_same_l"]:
        use_same_l(model, spec["same_l_cfg"], spec["same_l_ckpt"], device=device, verbose=verbose)
    add_lora(model, spec["rank"], verbose=verbose)
    load_adapter(model, spec["adapter"], verbose=verbose)
    return model.eval(), spec
