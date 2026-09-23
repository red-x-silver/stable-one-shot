# Stable Audio 3 **medium** base model goes here

This folder holds the backbone for the two **`M-*`** adapters (`M-r4`, `M-r16`).

It is **also needed by the `S-*` adapters**, because the **SAME-L** autoencoder lives inside this
checkpoint and — as in the paper — SAME-L encodes and decodes for every model. When running an
`S-*` adapter only the autoencoder is read from here (3.4 GB of the file), while the DiT comes from
[`../sa3-base/`](../sa3-base/README.md). Pass `--autoencoder same-s` to skip this download and use
the small backbone's own autoencoder instead; that is lighter, but not the paper's configuration.

The base weights (~8.6 GB) are **not** redistributed here — download them and place both files in
this folder:

```
models/sa3-base-medium/
  model.safetensors     # ~8.6 GB base weights
  model_config.json     # base model config
```

Download from the Stability AI model page (log in and accept the model licence first):

- https://huggingface.co/stabilityai/stable-audio-3-medium

If you keep the base model elsewhere, point at it with environment variables instead of copying:

```bash
export SA3_REPO_CFG=/path/to/model_config.json
export SA3_BASE_CKPT=/path/to/model.safetensors
```

The first run also downloads the frozen **T5Gemma** text encoder into your HuggingFace cache; this
is handled by `stable-audio-tools` and needs no manual step beyond network access.

The LoRA adapters themselves are bundled in [`../lora/`](../lora) — only the base model needs
downloading.

> **Adapters and backbones are not interchangeable.** An `M-*` adapter will not load onto the small
> backbone, and an `S-*` adapter will not load onto the medium one. `--adapter` selects both
> together, so prefer it over overriding paths by hand.
