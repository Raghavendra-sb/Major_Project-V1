"""
prepare_subset.py
-----------------
Step 1 of retraining. Copies ONLY your 8 classes out of the big Kaggle dataset
(21 classes, 1.6 GB) into a smaller folder, and zips it so you can upload it to
Google Drive and train on Colab.

Optionally it also removes TRAINING images that are near-copies of validation/test
images (using the overlap_pairs.csv from check_overlap.py), so the new model can
never have "seen" the pictures it will be tested on.

Usage (PowerShell, from your project folder, venv active):

    python prepare_subset.py `
        --root "C:\\Users\\raghu\\datasets\\skin\\SkinDisease" `
        --out  "C:\\Users\\raghu\\datasets\\skin8" `
        --drop-overlap overlap_results\\overlap_pairs.csv `
        --zip

What you get:
    <out>\\train\\<Class>\\*.jpg     <out>\\valid\\<Class>\\*.jpg     <out>\\test\\<Class>\\*.jpg
    <out>.zip                        (only if you pass --zip; this is the file you upload)

Optional flags:
    --drop-dist 2      remove a training image if it matches a valid/test image with hash
                       distance <= 2 (same meaning as in check_overlap.py). Default 2.
    --max-side 0       shrink pictures so their longest side is at most this many pixels
                       (e.g. 512) to make the upload smaller. 0 = keep the originals (default).
                       The model only sees 224x224 anyway; keep 0 unless your internet is slow.

Put --out OUTSIDE your project folder (as above) so VS Code and git do not try to track thousands of images.
"""

import argparse
import csv
import os
import re
import shutil
import sys

CLASS_NAMES = ["Acne", "Actinic_Keratosis", "Benign_tumors", "Eczema",
               "Lupus", "SkinCancer", "Vasculitis", "Warts"]
SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def load_drop_set(csv_path: str, max_dist: int) -> set:
    """Training images that are near-copies of a valid/test image (path_b of valid->train / test->train rows)."""
    if not os.path.exists(csv_path):
        sys.exit(f"❌ Overlap file not found: {csv_path}")
    drop = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split_b"] == "train" and row["split_a"] in ("valid", "test") and int(row["distance"]) <= max_dist:
                drop.add(_key(row["path_b"]))
    return drop


def copy_image(src: str, dst: str, max_side: int) -> None:
    if not max_side:
        shutil.copy2(src, dst)
        return
    from PIL import Image
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        img.save(os.path.splitext(dst)[0] + ".jpg", quality=95)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Copy the 8 project classes into a small train/valid/test folder.")
    ap.add_argument("--root", required=True, help="the dataset folder that contains train / valid / test")
    ap.add_argument("--out", required=True, help="new folder to create (must not exist yet)")
    ap.add_argument("--drop-overlap", default=None, metavar="CSV", help="overlap_pairs.csv from check_overlap.py")
    ap.add_argument("--drop-dist", type=int, default=2)
    ap.add_argument("--max-side", type=int, default=0)
    ap.add_argument("--zip", action="store_true", help="also create <out>.zip for uploading")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        sys.exit(f"❌ Dataset folder not found: {args.root}")
    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit(f"❌ {args.out} already exists and is not empty. Delete it or choose another --out.")

    wanted = {_norm(c): c for c in CLASS_NAMES}
    drop = load_drop_set(args.drop_overlap, args.drop_dist) if args.drop_overlap else set()
    if args.drop_overlap:
        print(f"Will drop up to {len(drop)} training images that duplicate a valid/test image (distance <= {args.drop_dist}).")

    counts = {}      # (split, class) -> number copied
    dropped = 0
    for entry in sorted(os.listdir(args.root)):
        split = SPLIT_ALIASES.get(entry.lower())
        split_dir = os.path.join(args.root, entry)
        if split is None or not os.path.isdir(split_dir):
            continue
        for folder in sorted(os.listdir(split_dir)):
            canon = wanted.get(_norm(folder))
            src_dir = os.path.join(split_dir, folder)
            if canon is None or not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(args.out, split, canon)
            os.makedirs(dst_dir, exist_ok=True)
            n = 0
            for fname in sorted(os.listdir(src_dir)):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                src = os.path.join(src_dir, fname)
                if split == "train" and _key(src) in drop:
                    dropped += 1
                    continue
                copy_image(src, os.path.join(dst_dir, fname), args.max_side)
                n += 1
            counts[(split, canon)] = n
            print(f"\r  copied {split}/{canon}: {n} images" + " " * 20, end="", flush=True)
    print()

    # ---- report ----
    print("\nImages per class")
    print(f"{'Class':<20}{'train':>8}{'valid':>8}{'test':>8}")
    missing = []
    for c in CLASS_NAMES:
        row = [counts.get((s, c), 0) for s in ("train", "valid", "test")]
        print(f"{c:<20}{row[0]:>8}{row[1]:>8}{row[2]:>8}")
        missing += [f"{s}/{c}" for s, v in zip(("train", "valid", "test"), row) if v == 0]
    totals = [sum(counts.get((s, c), 0) for c in CLASS_NAMES) for s in ("train", "valid", "test")]
    print(f"{'TOTAL':<20}{totals[0]:>8}{totals[1]:>8}{totals[2]:>8}")
    if args.drop_overlap:
        print(f"\nDropped {dropped} training images that duplicate valid/test images.")
    if missing:
        print("\n⚠️  These class folders were NOT found or are empty: " + ", ".join(missing))
        print("    Check the folder names inside --root and tell me what you see.")

    if args.zip:
        print("\nCreating the zip (this can take a minute)...")
        base = os.path.abspath(args.out).rstrip("\\/")
        zip_path = shutil.make_archive(base, "zip", root_dir=os.path.dirname(base), base_dir=os.path.basename(base))
        print(f"✅ Zip ready: {zip_path}  ({os.path.getsize(zip_path) / 1e6:.0f} MB)")
    print(f"✅ Folder ready: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
