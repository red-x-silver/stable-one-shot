# Stable Audio 3 **small** base model goes here

This folder holds the backbone for the four **`S-*`** adapters (`S-r4`, `S-r8`, `S-r16`, `S-r32`).
For the `M-*` adapters see [`../sa3-base-medium/`](../sa3-base-medium/README.md).

The base weights (~2.2 GB) are **not** redistributed here — download them and place both files in
this folder:

```
models/sa3-base/
  model.safetensors     # ~2.2 GB base weights
  model_config.json     # base model config
```

Download from the Stability AI model page (log in and accept the model licence first):

- https://huggingface.co/stabilityai/stable-audio-3-small-music

> `SA3-Small` throughout the paper refers to this **music** variant, `stable-audio-3-small-music`,
> not the SFX variant.

If you keep the base model elsewhere, point at it with environment variables instead of copying:

```bash
export SA3_REPO_CFG=/path/to/model_config.json
export SA3_BASE_CKPT=/path/to/model.safetensors
```

The first run also downloads the frozen **T5Gemma** text encoder into your HuggingFace cache; this
is handled by `stable-audio-tools` and needs no manual step beyond network access.

> **The `S-*` adapters also need the medium checkpoint**, because — as in the paper — the
> **SAME-L** autoencoder encodes and decodes for every model, and SAME-L ships inside
> [`../sa3-base-medium/`](../sa3-base-medium/README.md). Only the autoencoder is read from it
> (3.4 GB of the file); the DiT still comes from the small checkpoint here.
>
> To avoid that download, run with `--autoencoder same-s` and the small backbone's own
> autoencoder is used instead — lighter, but not the paper's configuration.

The LoRA adapters themselves are bundled in [`../lora/`](../lora) — only the base model needs
downloading.
