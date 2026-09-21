"""
fix_model_compat.py
-------------------
Fixes this error when loading a model that was trained on Colab:

    ValueError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}

Cause: Colab has a NEWER Keras than your laptop. The newer Keras writes a few extra
settings into the model file that your older Keras does not know. The learned weights are
fine; only the "description" part of the file (config.json) contains the unknown keys.

This script makes a COPY of the model without those keys (your original file is not changed):

    python fix_model_compat.py --model skin_model_v2.keras
    -> writes skin_model_v2_compat.keras

Then use the *_compat.keras file with evaluate.py and app.py.

How it works: it removes the known extra keys, tries to load the copy, and if your Keras
still complains about another unknown key, it removes that key too and tries again.
It also drops the training settings (optimizer etc.), which are not needed for predictions.
"""

import argparse
import json
import os
import re
import sys
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

KNOWN_EXTRA_KEYS = {"quantization_config"}       # first thing newer Keras adds to Dense layers
MAX_ROUNDS = 30


def strip_keys(obj, keys):
    """Return a copy of a JSON-like object without any dict key in `keys`."""
    if isinstance(obj, dict):
        return {k: strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [strip_keys(v, keys) for v in obj]
    return obj


def unknown_keys_from_error(text: str) -> set:
    """Read the names of settings that this Keras version rejected out of its error message."""
    found = set()
    for block in re.findall(r"Unrecognized keyword arguments passed to \w+: \{([^}]*)\}", text):
        found |= set(re.findall(r"'(\w+)'\s*:", block))
    found |= set(re.findall(r"got an unexpected keyword argument '(\w+)'", text))
    return found


def write_copy(src: str, dst: str, keys: set) -> None:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                cfg = json.loads(data.decode("utf-8"))
                cfg.pop("compile_config", None)                   # training settings: not needed
                if isinstance(cfg.get("config"), dict):
                    cfg["config"].pop("compile_config", None)
                data = json.dumps(strip_keys(cfg, keys)).encode("utf-8")
            zout.writestr(item, data)                              # keeps weights + metadata untouched


def try_load(path: str):
    from keras.models import load_model
    return load_model(path, compile=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make a Colab-trained .keras model loadable by an older Keras.")
    ap.add_argument("--model", required=True, help="the model file downloaded from Colab")
    ap.add_argument("--out", default=None, help="output file (default: <model>_compat.keras)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.model):
        sys.exit(f"❌ File not found: {args.model}")
    out = args.out or os.path.splitext(args.model)[0] + "_compat.keras"

    import keras
    print(f"Keras on this computer: {keras.__version__}")
    try:
        with zipfile.ZipFile(args.model) as z:
            saved_with = json.loads(z.read("metadata.json")).get("keras_version", "unknown")
        print(f"Model file was saved with Keras: {saved_with}")
    except Exception:
        pass

    try:
        try_load(args.model)
        print("✅ The original file already loads fine here. No fix needed; use it as it is.")
        return
    except Exception as e:
        first_error = str(e).strip().splitlines()[-1][:200]
        print(f"The original file does not load here ({first_error}).\nCreating a compatible copy...")

    keys = set(KNOWN_EXTRA_KEYS)
    for round_no in range(1, MAX_ROUNDS + 1):
        write_copy(args.model, out, keys)
        try:
            model = try_load(out)
            break
        except Exception as e:
            new_keys = unknown_keys_from_error(str(e)) - keys
            if not new_keys:
                print("\n❌ Still cannot load the copy, and the error does not name a setting I can remove:")
                print("   " + str(e).strip().splitlines()[-1][:300])
                print("   Copy this whole message and send it to me.")
                if os.path.exists(out):
                    os.remove(out)
                sys.exit(1)
            keys |= new_keys
    else:
        sys.exit("❌ Gave up after too many rounds. Send me the list above.")

    print(f"Removed these settings your Keras does not know: {sorted(keys)}")

    # sanity check: one blank image must give 8 probabilities that add up to 1
    import numpy as np
    probs = model.predict(np.zeros((1, 224, 224, 3), dtype="float32"), verbose=0)[0]
    print(f"Check: {len(probs)} outputs, sum of probabilities = {probs.sum():.4f} (should be 1.0000)")
    if abs(probs.sum() - 1) > 1e-3:
        sys.exit("❌ The loaded model gives strange output. Do not use it and tell me.")
    print(f"\n✅ Done. Use this file from now on: {out}")


if __name__ == "__main__":
    main()
