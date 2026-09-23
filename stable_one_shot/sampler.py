"""Rectified-flow sampling of the masked one-shot region, and decoding it back to audio."""
import numpy as np
import torch

from . import config as C


def build_cond(model, batch, device):
    """Assemble the three conditioning pathways: text, duration, and inpainting."""
    metadata = [{"prompt": p, "seconds_total": s}
                for p, s in zip(batch["prompt"], batch["seconds_total"])]
    cond = model.conditioner(metadata, device)
    cond["inpaint_mask"] = [batch["inpaint_mask"].to(device)]
    cond["inpaint_masked_input"] = [batch["inpaint_masked_input"].to(device)]
    return cond


@torch.no_grad()
def sample(model, batch, steps=C.SAMPLING_STEPS, device=None):
    """Integrate the predicted velocity field from noise with a Euler solver.

    The DiT predicts v = x1 - noise; starting from Gaussian noise at t=1 and stepping to t=0
    recovers a clean latent canvas, of which only the masked one-shot region is used.
    """
    device = device or next(model.parameters()).device
    x1 = batch["x1"].to(device)
    padding = batch["padding_mask"].to(device)
    cond = build_cond(model, batch, device)

    x = torch.randn_like(x1)
    ts = torch.linspace(1, 0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(x1.shape[0])
        v = model(x, t, cond=cond, cfg_dropout_prob=0.0, padding_mask=padding)
        x = x + v * (ts[i + 1] - ts[i])
    return x                                                          # (B, 256, 256)


@torch.no_grad()
def decode_region(model, canvas, gen_slice, trim=3e-3):
    """Decode one canvas's one-shot region through the frozen SAME decoder, to mono audio.

    `trim` drops the trailing silence the decoder emits after the one-shot's tail: everything past
    the last sample exceeding `trim` (3e-3 ~= -50 dBFS) is cut. This is a cosmetic post-process on
    the returned audio only; pass trim=0 to keep the full decoded region.
    """
    start, end = gen_slice
    model.pretransform.eval()
    audio = model.pretransform.decode(canvas[:, :, start:end])
    audio = audio.squeeze(0).mean(0).clamp(-1, 1).cpu().numpy()       # stereo -> mono
    if trim and trim > 0:
        loud = np.where(np.abs(audio) > trim)[0]
        if len(loud):
            return audio[:loud[-1] + 1]
    return audio
