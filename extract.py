"""Stable-one-shot — extract drum one-shots from a drum loop.

    python extract.py loop.wav --out out/
    python extract.py loops/ --out out/ --adapter M-r16
    python extract.py loop.wav --out out/ --adapter S-r16 --seed 3

Point it at a single audio file or at a directory (globbed recursively for *.wav). The model is
loaded once and reused across every input. Each run writes <stem>_kick.wav, <stem>_snare.wav and
<stem>_hihat.wav into --out.
"""
import argparse
import glob
import os
import sys
import time

from stable_one_shot import config as C
from stable_one_shot.extract import OneShotExtractor


def _inputs(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.wav"), recursive=True))
        if not files:
            raise SystemExit(f"no .wav files found under {path}")
        return files
    if not os.path.exists(path):
        raise SystemExit(f"input not found: {path}")
    return [path]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("loop", nargs="?", help="drum loop audio file, or a directory of .wav files")
    ap.add_argument("--out", default="out", help="output directory (default: out/)")
    ap.add_argument("--adapter", default=C.DEFAULT_ADAPTER, choices=list(C.ADAPTERS),
                    help=f"released adapter to use (default: {C.DEFAULT_ADAPTER})")
    ap.add_argument("--autoencoder", default=C.DEFAULT_AUTOENCODER, choices=list(C.AUTOENCODERS),
                    help="which SAME autoencoder encodes/decodes (default: same-l, the paper's "
                         "configuration). 'same-s' uses the small backbone's own autoencoder and "
                         "needs no medium download; it has no effect on the M-* adapters.")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    ap.add_argument("--steps", type=int, default=C.SAMPLING_STEPS,
                    help=f"Euler solver steps (default: {C.SAMPLING_STEPS})")
    ap.add_argument("--start-sec", type=float, default=0.0,
                    help="offset of the 4 s analysis window into the input (default: 0)")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep the decoder's trailing silence instead of trimming it")
    ap.add_argument("--device", default=None, help="force 'cuda' or 'cpu' (default: auto)")
    ap.add_argument("--list", action="store_true", help="list the released adapters and exit")
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'adapter':<8} {'backbone':<16} {'rank':>4} {'params':>8}")
        for name, spec in C.ADAPTERS.items():
            base = "SA3-Small-Music" if spec["backbone"] == "small" else "SA3-Medium"
            print(f"{name:<8} {base:<16} {spec['rank']:>4} {spec['params']:>8}")
        return 0

    if not args.loop:
        ap.error("the following arguments are required: loop")
    files = _inputs(args.loop)
    t0 = time.time()
    ex = OneShotExtractor(adapter=args.adapter, device=args.device, autoencoder=args.autoencoder)
    trim = 0.0 if args.no_trim else 3e-3

    for i, f in enumerate(files, 1):
        written = ex.extract_to(f, args.out, seed=args.seed, steps=args.steps,
                                start_sec=args.start_sec, trim=trim)
        names = ", ".join(os.path.basename(p) for p in written.values())
        print(f"[{i}/{len(files)}] {os.path.basename(f)} -> {names}", flush=True)

    print(f"[done] {len(files)} loop(s) in {time.time() - t0:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
