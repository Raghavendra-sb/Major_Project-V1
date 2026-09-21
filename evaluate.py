"""
evaluate.py
-----------
Evaluate the skin-disease CNN on a held-out test set and tune the triage thresholds.

Usage (PowerShell, from the project folder, venv active):
    python evaluate.py --data "C:\\path\\to\\test_folder"

Expected folder layout (one sub-folder per class, names = your class labels):
    test_folder/
        Acne/               img1.jpg  img2.jpg ...
        Actinic_Keratosis/  ...
        Benign_tumors/
        Eczema/
        Lupus/
        SkinCancer/
        Vasculitis/
        Warts/

Optional flags:
    --model skin_major_model.keras   path to the trained model
    --out eval_results               where reports are saved
    --limit 50                       use at most 50 images per class (quick trial run)
    --batch 32                       prediction batch size
    --low-conf 0.55                  same low-confidence threshold as rag_graph.py
    --exclude-overlap overlap_pairs.csv
                                     skip test/valid images that check_overlap.py found a near-duplicate
                                     of in the TRAIN split (gives a "cleaned" test set)
    --exclude-dist 5                 only exclude pairs with hash distance <= this (default 5)
    --map "Eczema Photos=Eczema"     use a folder whose name differs from your class (repeatable),
                                     e.g. for the DermNet dataset. Folder names that already match
                                     your classes (spaces / capitals ignored) need no --map.

IMPORTANT: the test images must NOT have been used for training or validation.
For the most convincing numbers, use images from a different source than the training set.

Outputs (in --out):
    summary.txt                      everything printed below
    metrics_per_class.csv            precision / recall / F1 / support per class
    confusion_matrix.csv / .png      counts (and a row-normalised .png)
    predictions.csv                  one row per image, so you can inspect the mistakes
"""

import argparse
import csv
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must match LABELS in app.py (same order as the CNN's output neurons).
LABELS = {
    0: "Acne",
    1: "Actinic_Keratosis",
    2: "Benign_tumors",
    3: "Eczema",
    4: "Lupus",
    5: "SkinCancer",
    6: "Vasculitis",
    7: "Warts",
}
CLASS_NAMES = [LABELS[i] for i in range(len(LABELS))]
N = len(CLASS_NAMES)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
EVAL_VERSION = "3 (supports --map and --exclude-overlap)"

_log_lines: list[str] = []


def log(text: str = "") -> None:
    print(text)
    _log_lines.append(text)


def _norm(name: str) -> str:
    """'Skin Cancer', 'skin_cancer' and 'SkinCancer' all become 'skincancer'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _pct(x) -> str:
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:5.1f}%"


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b != 0)
    return out


# ─────────────────────────────────────────────
# Data + model
# ─────────────────────────────────────────────
def parse_mapping(pairs):
    """['Acne and Rosacea Photos=Acne', ...] -> {'acneandrosaceaphotos': 0, ...}"""
    name_to_idx = {_norm(n): i for i, n in enumerate(CLASS_NAMES)}
    mapping = {}
    for pair in pairs or []:
        if "=" not in pair:
            sys.exit(f"❌ --map needs the form \"folder name=Class\", got: {pair}")
        folder, cls = pair.rsplit("=", 1)
        if _norm(cls) not in name_to_idx:
            sys.exit(f"❌ Unknown class in --map: '{cls}'. Use one of: {', '.join(CLASS_NAMES)}")
        mapping[_norm(folder)] = name_to_idx[_norm(cls)]
    return mapping


def collect_images(data_dir: str, limit: int | None, mapping: dict | None = None):
    """Return [(path, true_class_index)] from class sub-folders."""
    name_to_idx = {_norm(n): i for i, n in enumerate(CLASS_NAMES)}
    name_to_idx.update(mapping or {})
    items = []
    if not os.path.isdir(data_dir):
        sys.exit(f"❌ Folder not found: {data_dir}")

    for folder in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, folder)
        if not os.path.isdir(full):
            continue
        idx = name_to_idx.get(_norm(folder))
        if idx is None:
            print(f"⚠️  Skipping folder '{folder}' (not one of: {', '.join(CLASS_NAMES)})")
            continue
        files = sorted(f for f in os.listdir(full) if f.lower().endswith(IMAGE_EXTS))
        if limit:
            files = files[:limit]
        items.extend((os.path.join(full, f), idx) for f in files)

    if not items:
        sys.exit("❌ No images found. Check the folder layout in the header of evaluate.py.")
    return items


def apply_exclusions(items, csv_path: str, max_dist: int):
    """Drop images that check_overlap.py matched to a TRAIN image (distance <= max_dist)."""
    if not os.path.exists(csv_path):
        sys.exit(f"❌ Overlap file not found: {csv_path}")
    key = lambda p: os.path.normcase(os.path.abspath(p))
    bad = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split_b"] == "train" and row["split_a"] != "train" and int(row["distance"]) <= max_dist:
                bad.add(key(row["path_a"]))
    kept = [(p, t) for p, t in items if key(p) not in bad]
    removed = len(items) - len(kept)
    log(f"Cleaned set: excluded {removed} of {len(items)} images that have a near-duplicate in the "
        f"training split (hash distance <= {max_dist}); {len(kept)} images remain.")
    if removed == 0:
        log("   (nothing matched: check that --data points to the same valid/test folder you ran check_overlap.py on)")
    if not kept:
        sys.exit("❌ Every image was excluded.")
    return kept


def load_keras():
    from keras.models import load_model
    try:
        from keras.preprocessing.image import img_to_array, load_img
    except ImportError:
        from keras.utils import img_to_array, load_img
    return load_model, img_to_array, load_img


def predict_all(model_path: str, items, batch: int):
    load_model, img_to_array, load_img = load_keras()
    if not os.path.exists(model_path):
        sys.exit(f"❌ Model file not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    size = (224, 224)
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, tuple) and len(shape) == 4 and shape[1] and shape[2]:
        size = (int(shape[1]), int(shape[2]))
    print(f"Image size: {size[0]}x{size[1]}  |  images: {len(items)}")

    probs, kept, skipped = [], [], []
    start_time = time.time()
    for start in range(0, len(items), batch):
        arrays, ok = [], []
        for path, true_idx in items[start:start + batch]:
            try:
                # Same preprocessing as app.py: resize, then divide by 255
                arrays.append(img_to_array(load_img(path, target_size=size)) / 255.0)
                ok.append((path, true_idx))
            except Exception as e:  # unreadable / corrupt image
                skipped.append((path, str(e)[:80]))
        if not arrays:
            continue
        p = np.asarray(model.predict(np.stack(arrays), verbose=0), dtype=float)
        if p.shape[1] != N:
            sys.exit(f"❌ Model outputs {p.shape[1]} classes but LABELS has {N}. Update LABELS in evaluate.py.")
        probs.append(p)
        kept.extend(ok)
        done = min(start + batch, len(items))
        print(f"\r  predicted {done}/{len(items)}", end="", flush=True)
    print(f"\r  done in {time.time() - start_time:.1f}s" + " " * 20)

    if skipped:
        print(f"⚠️  Skipped {len(skipped)} unreadable images (first: {skipped[0][0]})")
    if not probs:
        sys.exit("❌ No image could be read.")
    return np.vstack(probs), kept


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def confusion_matrix(y_true, y_pred):
    cm = np.zeros((N, N), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1, support


def expected_calibration_error(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc, avg_conf = float(correct[mask].mean()), float(conf[mask].mean())
        ece += (n / len(conf)) * abs(acc - avg_conf)
        rows.append((lo, hi, n, avg_conf, acc))
    return ece, rows


# ─────────────────────────────────────────────
# Triage / safety analysis (uses your real triage_rules.py)
# ─────────────────────────────────────────────
def triage_levels(probs, low_conf, floor=None, margin=None):
    """Urgency level (no symptom answers) for every image under the given settings."""
    import triage_rules as tr
    old = (tr.RISK_PROB_FLOOR, tr.CLOSE_MARGIN)
    if floor is not None:
        tr.RISK_PROB_FLOOR = floor
    if margin is not None:
        tr.CLOSE_MARGIN = margin
    try:
        levels = []
        for p in probs:
            top = list(zip(CLASS_NAMES, p))
            pred = CLASS_NAMES[int(np.argmax(p))]
            levels.append(tr.assess(pred, float(np.max(p)), top, None, low_conf)["level"])
        return np.array(levels)
    finally:
        tr.RISK_PROB_FLOOR, tr.CLOSE_MARGIN = old


def flagged(levels, mask):
    """Share of images in `mask` whose urgency is Medium or High."""
    n = int(mask.sum())
    if n == 0:
        return None
    return float(np.isin(levels[mask], ["medium", "high"]).sum() / n)


def safety_section(probs, y_true, y_pred, low_conf):
    try:
        import triage_rules as tr
    except ImportError:
        log("\n(triage_rules.py not found next to evaluate.py: skipping the triage analysis)")
        return

    watch = [c for c in CLASS_NAMES if c in tr.WATCH_CLASSES]
    low_risk = [c for c in CLASS_NAMES if tr.CLASS_INFO.get(c, {}).get("level") == "low"]
    watch_idx = [CLASS_NAMES.index(c) for c in watch]
    low_idx = [CLASS_NAMES.index(c) for c in low_risk]
    watch_mask = np.isin(y_true, watch_idx)
    low_mask = np.isin(y_true, low_idx)

    log("\n" + "=" * 64)
    log("SAFETY: does the triage safety net catch serious cases?")
    log("=" * 64)
    log(f"Current settings: RISK_PROB_FLOOR={tr.RISK_PROB_FLOOR}, CLOSE_MARGIN={tr.CLOSE_MARGIN}, low-conf={low_conf}")
    log("'Flagged' = urgency Medium or High (before any symptom answers).\n")

    levels = triage_levels(probs, low_conf)
    log(f"{'True class':<20}{'n':>6}{'CNN correct':>13}{'Flagged M+':>12}{'High':>8}")
    for c in watch + low_risk:
        i = CLASS_NAMES.index(c)
        m = y_true == i
        n = int(m.sum())
        if n == 0:
            continue
        log(f"{c:<20}{n:>6}{_pct((y_pred[m] == i).mean()):>13}{_pct(flagged(levels, m)):>12}"
            f"{_pct((levels[m] == 'high').mean()):>8}")

    # Where do the serious cases that the CNN gets wrong end up?
    for c in ("SkinCancer",):
        if c not in CLASS_NAMES:
            continue
        i = CLASS_NAMES.index(c)
        m = (y_true == i) & (y_pred != i)
        if m.sum():
            log(f"\nMissed {c} images ({int(m.sum())}) were predicted as:")
            for j in np.argsort(-np.bincount(y_pred[m], minlength=N)):
                cnt = int((y_pred[m] == j).sum())
                if cnt:
                    log(f"   {CLASS_NAMES[j]:<20}{cnt:>5}   (still flagged Medium+ by triage: "
                        f"{int(np.isin(levels[m & (y_pred == j)], ['medium', 'high']).sum())})")

    log(f"\nFalse alarms: low-risk classes flagged Medium+ = {_pct(flagged(levels, low_mask))}")

    # Threshold sweeps (this is how you choose the numbers, instead of guessing)
    log("\nRISK_PROB_FLOOR sweep (higher = fewer false alarms, but more serious cases missed)")
    log(f"{'floor':>7}{'SkinCancer flagged':>21}{'watch classes flagged':>24}{'false alarms':>15}")
    sc_mask = y_true == CLASS_NAMES.index("SkinCancer") if "SkinCancer" in CLASS_NAMES else np.zeros(len(y_true), bool)
    for floor in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        lv = triage_levels(probs, low_conf, floor=floor)
        log(f"{floor:>7.2f}{_pct(flagged(lv, sc_mask)):>21}{_pct(flagged(lv, watch_mask)):>24}{_pct(flagged(lv, low_mask)):>15}")

    log("\nCLOSE_MARGIN sweep (top-1 vs top-2 gap that counts as 'unsure')")
    log(f"{'margin':>7}{'SkinCancer flagged':>21}{'watch classes flagged':>24}{'false alarms':>15}")
    for margin in (0.05, 0.10, 0.20, 0.30, 0.40):
        lv = triage_levels(probs, low_conf, margin=margin)
        log(f"{margin:>7.2f}{_pct(flagged(lv, sc_mask)):>21}{_pct(flagged(lv, watch_mask)):>24}{_pct(flagged(lv, low_mask)):>15}")

    log("\nHow to read this: pick the settings where SkinCancer flagged is (close to) 100%")
    log("while false alarms stay at a level you can defend. Then edit them in triage_rules.py.")


# ─────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────
def save_confusion_png(cm, path, normalize):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    data = cm.astype(float)
    if normalize:
        data = np.nan_to_num(data / np.maximum(cm.sum(axis=1, keepdims=True), 1))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised = recall)" if normalize else " (counts)"))
    thresh = data.max() / 2 if data.max() else 0
    for i in range(N):
        for j in range(N):
            txt = f"{data[i, j]:.2f}" if normalize else str(cm[i, j])
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the skin CNN on a held-out test folder.")
    ap.add_argument("--data", required=True, help="test folder with one sub-folder per class")
    ap.add_argument("--model", default="skin_major_model.keras")
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--limit", type=int, default=None, help="max images per class (quick trial)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--low-conf", type=float, default=0.55)
    ap.add_argument("--exclude-overlap", default=None, metavar="CSV",
                    help="overlap_pairs.csv from check_overlap.py: skip images that duplicate a training image")
    ap.add_argument("--exclude-dist", type=int, default=5, help="max hash distance to exclude (default 5)")
    ap.add_argument("--map", action="append", default=[], metavar='"FOLDER=CLASS"',
                    help='map a differently named folder to one of your classes; repeat as needed')
    args = ap.parse_args(argv)
    print(f"evaluate.py version {EVAL_VERSION}")

    items = collect_images(args.data, args.limit, parse_mapping(args.map))
    if args.exclude_overlap:
        items = apply_exclusions(items, args.exclude_overlap, args.exclude_dist)
    probs, kept = predict_all(args.model, items, args.batch)
    y_true = np.array([t for _, t in kept])
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (y_pred == y_true)
    os.makedirs(args.out, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, support = per_class_metrics(cm)
    present = support > 0

    log("\n" + "=" * 64)
    log(f"EVALUATION  |  {len(y_true)} images  |  {args.data}")
    log("=" * 64)
    log("Reminder: these numbers are only meaningful if the test images were NOT used in training.\n")

    log(f"{'Class':<20}{'Precision':>10}{'Recall':>9}{'F1':>8}{'Support':>9}")
    for i, name in enumerate(CLASS_NAMES):
        log(f"{name:<20}{_pct(precision[i]):>10}{_pct(recall[i]):>9}{_pct(f1[i]):>8}{int(support[i]):>9}")

    top3 = np.argsort(-probs, axis=1)[:, :3]
    top3_acc = float(np.mean([t in row for t, row in zip(y_true, top3)]))
    log("\nOverall")
    log(f"  Accuracy            : {_pct(correct.mean())}")
    log(f"  Balanced accuracy   : {_pct(np.nanmean(recall[present]))}   (mean recall; fairer if classes are imbalanced)")
    log(f"  Macro F1            : {_pct(np.nanmean(f1[present]))}")
    log(f"  Weighted F1         : {_pct(np.nansum(f1[present] * support[present]) / support[present].sum())}")
    log(f"  Top-3 accuracy      : {_pct(top3_acc)}   (true class is among the 3 shown on the result page)")

    if "SkinCancer" in CLASS_NAMES:
        i = CLASS_NAMES.index("SkinCancer")
        log(f"\n  >> SkinCancer recall (most important safety number): {_pct(recall[i])}  "
            f"({int(cm[i, i])} of {int(support[i])} found)")

    # Confidence behaviour
    ece, rows = expected_calibration_error(conf, correct.astype(float))
    log("\nConfidence")
    if correct.any():
        log(f"  Mean confidence when CORRECT : {_pct(conf[correct].mean())}")
    if (~correct).any():
        log(f"  Mean confidence when WRONG   : {_pct(conf[~correct].mean())}   "
            "(if this is also high, the model is over-confident)")
    log(f"  Calibration error (ECE)      : {ece:.3f}   (0 = perfectly calibrated; lower is better)")
    log(f"  {'bin':<12}{'images':>7}{'avg conf':>10}{'accuracy':>10}")
    for lo, hi, n, avg_conf, acc in rows:
        log(f"  {lo:.1f}-{hi:.1f}{'':<5}{n:>7}{_pct(avg_conf):>10}{_pct(acc):>10}")

    log(f"\n  Accuracy if we only trust predictions above a confidence threshold:")
    log(f"  {'threshold':>10}{'coverage':>10}{'accuracy':>10}")
    for th in (0.0, 0.5, args.low_conf, 0.7, 0.85, 0.95):
        m = conf >= th
        log(f"  {th:>10.2f}{_pct(m.mean()):>10}{_pct(correct[m].mean() if m.any() else None):>10}")

    safety_section(probs, y_true, y_pred, args.low_conf)

    # ── save files ──
    with open(os.path.join(args.out, "metrics_per_class.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for i, name in enumerate(CLASS_NAMES):
            w.writerow([name] + [("" if np.isnan(v) else round(float(v), 4)) for v in (precision[i], recall[i], f1[i])] + [int(support[i])])
    with open(os.path.join(args.out, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            w.writerow([name] + [int(v) for v in cm[i]])
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "predicted", "confidence", "correct", "top3"])
        for (path, t), p, row in zip(kept, probs, top3):
            w.writerow([path, CLASS_NAMES[t], CLASS_NAMES[int(p.argmax())], round(float(p.max()), 4),
                        int(p.argmax() == t), " | ".join(f"{CLASS_NAMES[j]} {p[j]:.2f}" for j in row)])

    made_png = save_confusion_png(cm, os.path.join(args.out, "confusion_matrix.png"), normalize=False)
    save_confusion_png(cm, os.path.join(args.out, "confusion_matrix_normalized.png"), normalize=True)
    if not made_png:
        log("\n(matplotlib not installed: the PNG confusion matrices were skipped. `python -m pip install matplotlib`)")

    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")
    print(f"\n✅ Reports saved in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
