"""End-to-end one-shot extraction: a drum loop in, one isolated one-shot per instrument out."""
import numpy as np
import soundfile as sf
import torch

from . import config as C
from . import sa3
from .canvas import build_canvas, load_silence_latent
from .sampler import decode_region, sample


def load_loop(path, sr=C.SR, seconds=C.LOOP_SECONDS, start_sec=0.0):
    """Read an audio file as stereo at `sr` and crop/pad it to a `seconds`-long window."""
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)       # (T, ch)
    x = torch.from_numpy(x.T)                                         # (ch, T)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)                                            # SA3 expects stereo
    elif x.shape[0] > 2:
        x = x[:2]
    if file_sr != sr:
        import torchaudio
        x = torchaudio.functional.resample(x, file_sr, sr)
    n = int(round(seconds * sr))
    s0 = int(round(start_sec * sr))
    seg = x[:, s0:s0 + n]
    if seg.shape[-1] < n:
        seg = torch.nn.functional.pad(seg, (0, n - seg.shape[-1]))
    return seg                                                        # (2, n)


class OneShotExtractor:
    """Holds the loaded backbone + adapter so many loops can be processed in one session."""

    def __init__(self, adapter=C.DEFAULT_ADAPTER, device=None,
                 autoencoder=C.DEFAULT_AUTOENCODER, verbose=True):
        self.device = device or C.device()
        self.model, self.spec = sa3.load(adapter, device=self.device,
                                         autoencoder=autoencoder, verbose=verbose)
        self.silence = load_silence_latent(self.device)
        self.sr = self.model.sample_rate
        self.latent_fps = self.sr / self.model.pretransform.downsampling_ratio

    @torch.no_grad()
    def extract(self, loop_path, seed=0, steps=C.SAMPLING_STEPS, start_sec=0.0, trim=3e-3):
        """Extract a kick, snare and hi-hat one-shot from one drum loop.

        All three instruments are produced in a single batched forward pass -- they differ only in
        the text prompt, which is what lets one adapted model cover the whole instrument set.

        Returns {instrument: 1-D float32 waveform at self.sr}.
        """
        seg = load_loop(loop_path, sr=self.sr, start_sec=start_sec)
        self.model.pretransform.eval()
        loop_latent = self.model.pretransform.encode(
            seg.unsqueeze(0).to(self.device)).squeeze(0)              # (256, ~44)

        prompts = [C.PROMPT.format(label=label) for _, label in C.INSTRUMENTS]
        batch, gen_slice = build_canvas(loop_latent, self.silence, prompts, self.latent_fps)

        torch.manual_seed(seed)
        canvas = sample(self.model, batch, steps=steps, device=self.device)
        return {name: decode_region(self.model, canvas[i:i + 1], gen_slice, trim=trim)
                for i, (name, _label) in enumerate(C.INSTRUMENTS)}

    def extract_to(self, loop_path, out_dir, stem=None, **kw):
        """Extract and write `<stem>_<instrument>.wav` into `out_dir`. Returns the written paths."""
        import os
        os.makedirs(out_dir, exist_ok=True)
        stem = stem or os.path.splitext(os.path.basename(loop_path))[0]
        written = {}
        for name, audio in self.extract(loop_path, **kw).items():
            path = os.path.join(out_dir, f"{stem}_{name}.wav")
            sf.write(path, audio, self.sr, subtype="PCM_24")
            written[name] = path
        return written


def extract_one_shots(loop_path, out_dir=None, adapter=C.DEFAULT_ADAPTER, device=None,
                      autoencoder=C.DEFAULT_AUTOENCODER, seed=0, steps=C.SAMPLING_STEPS,
                      verbose=True):
    """One-call convenience wrapper. Loads the model, extracts, and optionally writes wavs.

    Prefer `OneShotExtractor` when processing more than one loop -- it loads the model only once.
    """
    ex = OneShotExtractor(adapter=adapter, device=device, autoencoder=autoencoder, verbose=verbose)
    if out_dir:
        return ex.extract_to(loop_path, out_dir, seed=seed, steps=steps)
    return ex.extract(loop_path, seed=seed, steps=steps)
