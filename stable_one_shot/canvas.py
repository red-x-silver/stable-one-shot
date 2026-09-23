"""The continuation canvas and inpainting mask (Section 3.2, Figure 1).

The inpainting pathway requires the conditioning context to have the same shape as the sequence
being generated, but a 4 s loop encodes to ~44 latent frames while a 1 s one-shot occupies ~11. The
canvas reconciles them inside the DiT's 256-frame minimum sequence length:

    [ encoded drum loop | silence gap | one-shot region | padding ]
      given (mask 1)      3 frames      11 frames         to 256
                          given         GENERATED (0)     attention-masked

The silence gap *and* the trailing padding are filled with the SAME encoding of silence rather than
zeros, because zero is not silence in the normalised latent space. The padded frames are
additionally excluded from attention by `padding_mask`.
"""
import numpy as np
import torch

from . import config as C


def load_silence_latent(device):
    """The SAME encoding of silence, shape (256, 1). Used to fill the gap in-distribution."""
    z = np.load(C.SILENCE_LATENT).astype(np.float32)
    return torch.from_numpy(z).unsqueeze(1).to(device)               # (256, 1)


def build_canvas(loop_latent, silence, prompts, latent_fps):
    """Assemble the batched inpainting inputs for one loop across several instruments.

    loop_latent : (256, nL) encoded 4 s drum loop
    silence     : (256, 1) SAME silence latent
    prompts     : list of text prompts, one per instrument -- the batch dimension
    latent_fps  : latent frame rate, used for the duration conditioning

    Returns the batch dict consumed by `sampler.sample`, plus the (start, end) one-shot slice.
    """
    device = loop_latent.device
    n_latent = loop_latent.shape[1]
    N, gap, n_shot = C.CANVAS_FRAMES, C.GAP_FRAMES, C.ONESHOT_FRAMES
    start = n_latent + gap
    end = start + n_shot
    if end > N:
        raise ValueError(f"canvas of {N} frames too small for {n_latent}+{gap}+{n_shot} frames")

    x1 = torch.zeros(256, N, device=device)
    x1[:, :n_latent] = loop_latent                                    # given: the drum loop
    x1[:, n_latent:start] = silence.expand(-1, gap)                   # given: the silence gap
    x1[:, end:] = silence.expand(-1, N - end)                         # trailing padding: silence

    mask = torch.ones(1, N, device=device)
    mask[:, start:end] = 0.0                                          # 0 = generate this region
    masked_input = x1 * mask

    padding = torch.zeros(N, dtype=torch.bool, device=device)
    padding[:end] = True                                              # attend to loop+gap+one-shot

    batch_size = len(prompts)
    rep = lambda t: t.unsqueeze(0).repeat(batch_size, *([1] * t.dim()))
    return {
        "x1": rep(x1),
        "inpaint_mask": rep(mask),
        "inpaint_masked_input": rep(masked_input),
        "padding_mask": padding.unsqueeze(0).repeat(batch_size, 1),
        "seconds_total": [end / latent_fps] * batch_size,
        "prompt": list(prompts),
        "gen_slice": [(start, end)] * batch_size,
    }, (start, end)
