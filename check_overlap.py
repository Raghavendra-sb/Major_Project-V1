"""
check_overlap.py
----------------
Find (near-)duplicate images between the train / valid / test splits of a dataset,
and inside a split. Use it to answer: "Can I trust my test results, or did the model
already see these pictures in training?"

It uses a perceptual hash (dHash), so it also catches copies that were resized,
re-saved as JPEG, or flipped left-right. It does NOT catch rotated, zoomed or
heavily cropped copies, so a clean result means "no obvious duplicates", not a proof.

Usage (PowerShell, venv active, from the project folder):
    python check_overlap.py --root "C:\\Users\\raghu\\datasets\\skin\\SkinDisease"

Expected layout:  <root>/train/<Class>/*.jpg   <root>/valid/<Class>/*.jpg   <root>/test/<Class>/*.jpg
Only your 8 classes are compared by default (--classes all compares every folder).

Optional flags:
    --threshold 5      max hash distance (out of 64 bits) that counts as a duplicate
    --out overlap_results
    --classes our8|all

Outputs (in --out):
    summary.txt          the numbers printed below
    overlap_pairs.csv    every duplicate pair found, so you can open and look at them
"""

import argparse
import csv
import os
import re
import sys
import time

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("❌ Pillow is missing. Run:  python -m pip install pillow")

OUR_CLASSES = ["Acne", "Actinic_Keratosis", "Benign_tumors", "Eczema",
               "Lupus", "SkinCancer", "Vasculitis", "Warts"]
SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_lines: list[str] = []


def log(text=""):
    print(text)
    _lines.append(text)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
_FLIP = getattr(getattr(Image, "Transpose", Image), "FLIP_LEFT_RIGHT")


def dhash(gray: "Image.Image") -> int:
    """64-bit difference hash of a grayscale PIL image."""
    small = np.asarray(gray.resize((9, 8), _RESAMPLE), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hash_image(path: str):
    """Return (hash, hash_of_mirror_image) or None if the file cannot be read."""
    try:
        with Image.open(path) as img:
            try:
                img.draft("L", (128, 128))      # faster decoding of big JPEGs
            except Exception:
                pass
            gray = img.convert("L")
            return dhash(gray), dhash(gray.transpose(_FLIP))
    except Exception:
        return None


def collect(root: str, classes: str):
    wanted = {_norm(c) for c in OUR_CLASSES}
    splits = {}
    for entry in sorted(os.listdir(root)):
        split = SPLIT_ALIASES.get(entry.lower())
        full = os.path.join(root, entry)
        if split is None or not os.path.isdir(full):
            continue
        rows = []
        for cls in sorted(os.listdir(full)):
            cdir = os.path.join(full, cls)
            if not os.path.isdir(cdir):
                continue
            if classes == "our8" and _norm(cls) not in wanted:
                continue
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith(IMAGE_EXTS):
                    rows.append((os.path.join(cdir, f), cls))
        if rows:
            splits[split] = rows
    return splits


def hash_split(name: str, rows):
    paths, classes, h, hf = [], [], [], []
    bad = 0
    start = time.time()
    for i, (path, cls) in enumerate(rows, 1):
        result = hash_image(path)
        if result is None:
            bad += 1
            continue
        paths.append(path)
        classes.append(cls)
        h.append(result[0])
        hf.append(result[1])
        if i % 250 == 0 or i == len(rows):
            print(f"\r  hashing {name}: {i}/{len(rows)}", end="", flush=True)
    print(f"\r  hashed {name}: {len(paths)} images in {time.time() - start:.0f}s"
          + (f"  ({bad} unreadable skipped)" if bad else "") + " " * 10)
    return {"paths": paths, "classes": classes,
            "h": np.array(h, dtype=np.uint64), "hf": np.array(hf, dtype=np.uint64)}


def distances(one: int, many: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(many, np.uint64(one))
    return POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def compare(a: dict, b: dict, threshold: int, same_split: bool):
    """For every image in `a` find its closest image in `b` (also checking b mirrored)."""
    pool = np.concatenate([b["h"], b["hf"]])
    n_b = len(b["h"])
    matches = []
    for i, ha in enumerate(a["h"]):
        d = distances(int(ha), pool)
        if same_split:                       # do not match an image with itself or its own mirror
            d[i] = 64
            d[n_b + i] = 64
        j = int(d.argmin())
        if d[j] <= threshold:
            j %= n_b
            matches.append((i, j, int(d.min())))
    return matches


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find near-duplicate images across dataset splits.")
    ap.add_argument("--root", required=True, help="folder that contains train / valid / test")
    ap.add_argument("--threshold", type=int, default=5, help="max hash distance (0-64) counted as duplicate")
    ap.add_argument("--classes", choices=["our8", "all"], default="our8")
    ap.add_argument("--out", default="overlap_results")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        sys.exit(f"❌ Folder not found: {args.root}")
    splits = collect(args.root, args.classes)
    if len(splits) < 2:
        sys.exit("❌ Need at least two of the folders train / valid / test inside --root.")

    hashed = {name: hash_split(name, rows) for name, rows in splits.items()}
    os.makedirs(args.out, exist_ok=True)

    log("\n" + "=" * 64)
    log(f"NEAR-DUPLICATE CHECK  |  threshold {args.threshold}/64  |  classes: {args.classes}")
    log("=" * 64)
    log("Split sizes: " + ", ".join(f"{k}={len(v['paths'])}" for k, v in hashed.items()))

    pair_rows = []
    pairs_to_check = [("valid", "train"), ("test", "train"), ("test", "valid")]
    log("\nHow many images have a near-duplicate in ANOTHER split (this inflates test scores):")
    for a_name, b_name in pairs_to_check:
        if a_name not in hashed or b_name not in hashed:
            continue
        a, b = hashed[a_name], hashed[b_name]
        matches = compare(a, b, args.threshold, same_split=False)
        share = len(matches) / max(len(a["paths"]), 1)
        same_cls = sum(1 for i, j, _ in matches if a["classes"][i] == b["classes"][j])
        log(f"  {a_name:>5} -> {b_name:<5}: {len(matches):>5} of {len(a['paths'])} images "
            f"({share:.1%}) have a match; {same_cls} of those in the same class")
        for i, j, dist in matches:
            pair_rows.append([a_name, a["paths"][i], b_name, b["paths"][j], dist,
                              int(a["classes"][i] == b["classes"][j])])

    log("\nHow many images have a near-duplicate INSIDE their own split (copies / augmented variants):")
    for name, data in hashed.items():
        matches = compare(data, data, args.threshold, same_split=True)
        share = len(matches) / max(len(data["paths"]), 1)
        log(f"  {name:>5}: {len(matches):>5} of {len(data['paths'])} images ({share:.1%})")
        seen = set()
        for i, j, dist in matches:
            key = (min(i, j), max(i, j))       # list each unordered pair once
            if key in seen:
                continue
            seen.add(key)
            pair_rows.append([name, data["paths"][i], name, data["paths"][j], dist,
                              int(data["classes"][i] == data["classes"][j])])

    log("\nHow to read this:")
    log("  - Across splits ~0-1%   : results are trustworthy on this point.")
    log("  - Across splits >5%     : the model has probably seen these pictures; scores are inflated.")
    log("  - Inside a split high % : the split holds many copies of the same photo, so it has fewer")
    log("                            independent images than files; report the smaller number.")
    log("  - Rotated / zoomed / cropped copies are NOT detected, so a clean result is not a proof.")

    with open(os.path.join(args.out, "overlap_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split_a", "path_a", "split_b", "path_b", "distance", "same_class"])
        w.writerows(sorted(pair_rows, key=lambda r: r[4]))
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n✅ Saved in: {os.path.abspath(args.out)}  (open overlap_pairs.csv and look at a few pairs)")


if __name__ == "__main__":
    main()
