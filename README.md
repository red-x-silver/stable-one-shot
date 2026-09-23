# Stable-one-shot

Drum one-shot extraction by repurposing a generative foundation model.

Reference implementation and released model adapters for **"Repurposing a Generative Foundation
Model for Drum One-Shot Extraction"** (Xiaowan Yi and Mathieu Barthet, Centre for Digital Music,
Queen Mary University of London).

Given a **drum loop**, the system recovers the isolated single-hit **kick, snare and hi-hat
one-shots** present in it. A frozen [Stable Audio 3](https://huggingface.co/stabilityai) backbone is
steered by small low-rank adapters, reusing the model's own conditioning pathways rather than adding
new modules:

- the **inpainting pathway** carries the input loop as fixed context on a *continuation canvas*,
  with the one-shot region left to be generated;
- the **text pathway** selects which instrument to extract, so a **single adapted model covers all
  three instruments** — no per-instrument models.

```
loop.wav ─▶ SAME encode ─▶ [ loop | gap | ▓ one-shot ▓ | padding ] ─▶ DiT (rectified-flow, 50 steps)
                             given  given   GENERATED    masked          ▲
                                                                   "isolated {kick|snare|hi-hat}
                                                                        drum one-shot"
         ─▶ SAME decode of the generated region ─▶ kick.wav, snare.wav, hihat.wav
```

## Released adapters

All six adapters from the paper are bundled in [`models/lora/`](models/lora). They differ only in
backbone and LoRA rank; MSS is the paired multi-scale spectral similarity against ground truth on
the 101-200 evaluation set (**lower is better**), FAD is codec-isolated.

| `--adapter` | Backbone | Rank | Trainable params | MSS ↓ | FAD<sub>VGGish</sub> ↓ | FAD<sub>CLAP</sub> ↓ | Size |
|---|---|:--:|--:|:--:|:--:|:--:|--:|
| **`S-r4`** *(default)* | SA3-Small-Music | 4 | 2.6 M | 1.47 | 2.04 | 0.62 | 5 MB |
| `S-r8` | SA3-Small-Music | 8 | 5.1 M | 1.49 | 2.27 | 0.65 | 10 MB |
| `S-r16` | SA3-Small-Music | 16 | 10.3 M | 1.54 | 2.17 | 0.61 | 20 MB |
| `S-r32` | SA3-Small-Music | 32 | 20.6 M | 1.57 | 2.36 | 0.63 | 39 MB |
| `M-r4` | SA3-Medium | 4 | 5.2 M | 1.49 | 0.69 | 0.17 | 10 MB |
| **`M-r16`** | SA3-Medium | 16 | 20.7 M | **1.44** | **0.61** | **0.16** | 40 MB |

*For reference on the same benchmark: DOSE scores 2.03 MSS with 497.4 M trainable parameters, and
the non-adapted SA3-Small backbone scores 4.15 MSS — every adapter here beats both. See Table 1 of
the paper for the full comparison.*

**Which should you use?**

- **`S-r4`** — the default. Best spectral fidelity per parameter, smallest download (2.2 GB
  backbone), fastest inference. Start here.
- **`M-r16`** — best overall. Marginally better MSS, and clearly better FAD, i.e. the outputs are
  distributionally more realistic. Costs an 8.6 GB backbone and roughly 2× the inference time.

On SA3-Small, MSS degrades monotonically with rank: the task repurposes existing pathways rather
than learning new behaviour, so a rank-4 update suffices. On SA3-Medium, rank 16 beats rank 4.

## Setup

One Python environment (3.10+), built around `stable-audio-tools`.

```bash
# 1. the foundation-model library (pins its own torch build -- install it first)
pip install git+https://github.com/Stability-AI/stable-audio-tools

# 2. this package's remaining dependencies
pip install -r requirements.txt
```

Install a `torch` / `torchaudio` build matching your CUDA setup; a GPU is used automatically when
available. Verified with Python 3.10 and torch 2.7.1+cu126.

**Then download the base models.** The adapters are bundled, but the frozen backbones they adapt
are not redistributed here:

| Download | Into | Needed for |
|---|---|---|
| [`stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music) (~2.2 GB) | [`models/sa3-base/`](models/sa3-base/README.md) | the `S-*` adapters |
| [`stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium) (~8.6 GB) | [`models/sa3-base-medium/`](models/sa3-base-medium/README.md) | the `M-*` adapters, **and** the SAME-L autoencoder used by every model |

As in the paper, **SAME-L** — the autoencoder that ships inside the medium checkpoint — encodes the
input loop and decodes the extracted one-shot for *all* models, including the `S-*` ones, whose
adapters operate in the SAME-L latent space. Running an `S-*` adapter therefore reads the DiT from
the small checkpoint and the autoencoder from the medium one.

If you would rather not download the medium checkpoint, pass **`--autoencoder same-s`** to use the
small backbone's own autoencoder instead. SAME-S is distilled from SAME-L and shares its latent
space, so this works and is much lighter — but it is not the configuration the paper reports, and
it changes the output (in our testing SAME-L yields tighter decays and brighter hi-hats). The flag
has no effect on the `M-*` adapters, which already carry SAME-L.

Each folder's README covers pointing at a copy that already exists elsewhere on disk, via
`SA3_REPO_CFG` / `SA3_BASE_CKPT` for the backbone and `SA3_SAME_L_CFG` / `SA3_SAME_L_CKPT` for the
autoencoder. The first run additionally pulls the frozen T5Gemma text encoder into your
HuggingFace cache.

## Usage

### Command line

```bash
python extract.py loop.wav --out out/                     # default adapter (S-r4)
python extract.py loop.wav --out out/ --adapter M-r16     # highest fidelity
python extract.py loops/  --out out/                      # a whole directory, model loaded once
python extract.py --list                                  # show the released adapters
```

Writes `<stem>_kick.wav`, `<stem>_snare.wav` and `<stem>_hihat.wav` into `--out` as 24-bit
44.1 kHz mono.

| Flag | Meaning |
|---|---|
| `--adapter` | which released adapter to use (default `S-r4`) |
| `--autoencoder` | `same-l` (default, the paper's configuration) or `same-s` to skip the medium download; no effect on `M-*` |
| `--seed` | sampling seed; the model is generative, so different seeds give different one-shots (default 0) |
| `--steps` | rectified-flow Euler steps (default 50) |
| `--start-sec` | offset of the 4 s analysis window into a longer input (default 0) |
| `--no-trim` | keep the decoder's trailing silence instead of trimming it |
| `--device` | force `cuda` or `cpu` |

### As a library

```python
from stable_one_shot import OneShotExtractor

ex = OneShotExtractor("S-r4")                 # loads backbone + adapter once
one_shots = ex.extract("loop.wav")            # {"kick": ndarray, "snare": ..., "hihat": ...}
ex.extract_to("loop.wav", "out/")             # ... or write wavs directly
```

`OneShotExtractor` keeps the model resident, so reuse one instance across many loops. All three
instruments are produced in a **single batched forward pass** — they differ only in the text prompt.

## How it works

The input loop is cropped or zero-padded to **4 s**, encoded by the frozen SAME autoencoder to a
continuous latent (~44 frames at the 10.77 Hz latent rate), and laid out on a fixed **256-frame
continuation canvas**:

| Region | Frames | Role |
|---|---|---|
| drum loop | 0 … ~44 | given context (inpainting mask 1) |
| silence gap | 3 | given; filled with the SAME encoding of silence, since zero is *not* silence in the normalised latent space |
| **one-shot** | 11 (~1 s) | **generated** (inpainting mask 0) |
| padding | to 256 | excluded from attention by the padding mask |

The DiT then integrates its predicted velocity field from Gaussian noise to a clean latent with a
50-step Euler solver, and only the one-shot region is decoded back to audio.

Instrument selection is pure text conditioning — the prompt template is
`"isolated {kick|snare|hi-hat} drum one-shot"`. **The adapters expect exactly this string and it
must not be changed**: altering it moves the model off the conditioning it was adapted under.

The adaptation lives entirely in the LoRA weights (rank 4–32 on the DiT's linear projections); the
SAME autoencoder, the text encoder and the original DiT weights all stay frozen.

## Reproducing the paper's numbers

This code is **bit-for-bit equivalent** to the research implementation used for the paper: on the
same input with the same seed, `extract.py` reproduces the original outputs exactly (verified for
`S-r4` and `M-r16`).

Two details matter if you are comparing numbers closely:

- **Use the default autoencoder.** `--autoencoder same-l` (the default) matches the paper, which
  encodes and decodes with SAME-L throughout. `--autoencoder same-s` is a convenience for skipping
  the 8.6 GB download and will not reproduce the reported figures.
- **Sampling is stochastic.** MSS/FAD in the paper are averages over a benchmark set at a fixed
  seed. A single loop at a single seed will not reproduce a table value.

This repository is the inference and model release; see the paper for the method and its
evaluation.

## Layout

```
extract.py                  # CLI + entry point
stable_one_shot/
  config.py                 # adapter registry, canvas constants, prompt template
  sa3.py                    # backbone loading + LoRA injection / adapter loading
  canvas.py                 # continuation canvas and inpainting mask (Fig. 1)
  sampler.py                # rectified-flow Euler sampling + SAME decoding
  extract.py                # OneShotExtractor: loop in, one-shots out
  assets/
    same_silence_latent.npy # SAME encoding of silence, fills the canvas gap
models/
  lora/                     # the six released adapters (bundled)
  sa3-base/                 # <- download SA3-Small-Music here
  sa3-base-medium/          # <- download SA3-Medium here
```

## Citation

```bibtex
@inproceedings{yi2026stableoneshot,
  title     = {Repurposing a Generative Foundation Model for Drum One-Shot Extraction},
  author    = {Yi, Xiaowan and Barthet, Mathieu},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```

## Licence and attribution

The adapters in `models/lora/` are released for research use. They are **LoRA updates only** — they
contain no Stable Audio 3 weights, and require you to obtain the base model from Stability AI under
its own licence. Stable Audio 3 is trained exclusively on licensed and Creative Commons data.
