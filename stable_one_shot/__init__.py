"""Stable-one-shot: drum one-shot extraction by repurposing a generative foundation model.

Reference implementation for "Repurposing a Generative Foundation Model for Drum One-Shot
Extraction" (Yi and Barthet). A frozen Stable Audio 3 backbone is steered by low-rank adapters:
the inpainting pathway carries the input drum loop on a continuation canvas, and the text pathway
selects which instrument to extract, so one adapted model covers kick, snare and hi-hat.

    from stable_one_shot import OneShotExtractor
    ex = OneShotExtractor("S-r4")
    ex.extract_to("loop.wav", "out/")
"""
from .config import ADAPTERS, DEFAULT_ADAPTER, INSTRUMENTS
from .extract import OneShotExtractor, extract_one_shots

__all__ = ["OneShotExtractor", "extract_one_shots", "ADAPTERS", "DEFAULT_ADAPTER", "INSTRUMENTS"]
__version__ = "1.0.0"
