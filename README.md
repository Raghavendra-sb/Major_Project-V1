# DermaAI: Skin Condition Screening Assistant

DermaAI is a web app that looks at a photo of a skin problem and returns three things:

1. **A prediction** from a deep-learning image classifier (top 3 most likely conditions, out of 8).
2. **An urgency check** from transparent, rule-based triage (Low / Medium / High) that tells the user how soon to see a doctor.
3. **A plain-language explanation**, written by an LLM (Google Gemini) that is grounded in retrieved medical reference text (RAG).

The user can also answer a few multiple-choice follow-up questions (how long, itchy or painful, bleeding, changing, fever...) and the system re-runs with those answers.

> ⚠️ **This is a student screening aid, not a medical device.** It cannot diagnose anything and has not been clinically validated. If you are worried about your skin, see a doctor, whatever this app says. See [Limitations](#limitations).

<!-- Add screenshots here, for example:
![Result page](docs/screenshots/result.png)
-->

---

## Contents
- [Features](#features)
- [How it works](#how-it-works)
- [The classifier and its results](#the-classifier-and-its-results)
- [Triage rules](#triage-rules)
- [Limitations](#limitations)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Evaluating a model](#evaluating-a-model)
- [Retraining the model](#retraining-the-model)
- [Troubleshooting](#troubleshooting)
- [Privacy](#privacy)
- [Credits](#credits)

---

## Features
- Image upload (JPG / PNG / WEBP, up to 10 MB) with a dark, single-page UI and login screen (demo login).
- **Top-3 predictions** with probabilities.
- **Urgency badge** with the reasons behind it and "see a doctor if..." warning signs, produced by rules and not by the LLM.
- **RAG explanation**: retrieval from a Qdrant vector database of dermatology reference documents, then an explanation from Gemini with an automatic reviewer step that can reject and regenerate weak answers.
- **Follow-up questionnaire** (4 multiple-choice questions chosen by the predicted class) that can raise the urgency level and steer retrieval.
- **Graceful failures**: if every LLM call fails or a quota is hit, the page still shows the prediction and the full triage result.
- Evaluation tooling: confusion matrices, per-class metrics, calibration, a near-duplicate (data leakage) checker, and a triage-threshold analysis.

## How it works

```
Photo ─► CNN (EfficientNetV2-S, 288 px) ─► class probabilities
                                             │
                                             ▼
                              LangGraph pipeline (rag_graph.py)
   check_confidence ─► triage (rules, no API) ─┬─► low confidence: ask for a clearer photo
                                               │
                                               └─► retrieve_context (Qdrant)
                                                     ─► generate_explanation (Gemini)
                                                     ─► validate_response (length check + Gemini reviewer)
                                                          ├─ weak answer: retry (max 2) with the reviewer's feedback
                                                          └─ good answer ─► final response
                                             │
                                             ▼
                        result page: top-3 · urgency · explanation · follow-up questions
                                             │  (optional) answers ─► POST /refine ─► pipeline runs again
```

**Models used per step** (all configurable in `.env`, each tried in order until one works, which also spreads free-tier quota):

| Step | Models |
|---|---|
| Explanation | `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → `gemini-3.6-flash` |
| Reviewer | `gemini-3.1-flash-lite` → `gemini-3.5-flash-lite` |
| Embeddings (retrieval) | `sentence-transformers/all-MiniLM-L6-v2`, run locally |

The LLM never sees the photo. It only receives the predicted label, the confidence, the follow-up answers, and the retrieved reference text.

## The classifier and its results

**Classes (8):** Acne, Actinic_Keratosis, Benign_tumors, Eczema, Lupus, SkinCancer, Vasculitis, Warts.

**Model:** EfficientNetV2-S (ImageNet-pretrained) → global average pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(8, softmax). Input is a 288 × 288 photo scaled to 0–1 (the model rescales internally to what the backbone expects).

**Training** (Google Colab, T4 GPU, see [Retraining the model](#retraining-the-model)):
- Phase 1: train the new head with the backbone frozen (Adam, lr 1e-3).
- Phase 2: fine-tune the last 40% of the backbone layers, BatchNorm frozen (AdamW, lr 2e-5), with early stopping on validation loss.
- Class weights (balanced), augmentation (flip, ±20° rotation, zoom, contrast, brightness), mixed-precision training converted back to float32 before saving.
- The `valid` split picks the best epoch. The `test` split was never used for any decision.

**Data:** the Kaggle dataset *Human Skin Diseases (Image)* (`youssefmohmmed/human-skin-diseases-image`), using 8 of its 21 classes and its own train / valid / test split. Check the dataset's licence on Kaggle before redistributing anything derived from it.

| Class | train | valid | test |
|---|---:|---:|---:|
| Acne | 450 | 132 | 65 |
| Actinic_Keratosis | 1196 | 300 | 167 |
| Benign_tumors | 886 | 228 | 121 |
| Eczema | 762 | 228 | 112 |
| Lupus | 218 | 84 | 34 |
| SkinCancer | 769 | 156 | 77 |
| Vasculitis | 373 | 84 | 52 |
| Warts | 495 | 72 | 64 |
| **Total** | **5149** | **1284** | **692** |

(94 training images that were near-copies of validation/test images were removed; see [Evaluating a model](#evaluating-a-model).)

### Results (measured with `evaluate.py` on the laptop, final model)

| | valid (1284 images) | test (692 images) |
|---|---:|---:|
| Accuracy | 65.4% | 71.7% |
| Balanced accuracy | 59.7% | 67.8% |
| Macro F1 | 59.5% | 67.6% |
| Top-3 accuracy | 93.1% | 94.1% |
| SkinCancer recall | 74.4% (116 of 156) | 57.1% (44 of 77) |
| Calibration error (ECE) | 0.140 | 0.079 |

**Per class on the test split**

| Class | Precision | Recall | F1 | Images |
|---|---:|---:|---:|---:|
| Acne | 82.0% | 76.9% | 79.4% | 65 |
| Actinic_Keratosis | 91.4% | 82.6% | 86.8% | 167 |
| Benign_tumors | 62.9% | 78.5% | 69.9% | 121 |
| Eczema | 76.0% | 67.9% | 71.7% | 112 |
| Lupus | 38.5% | 44.1% | 41.1% | 34 |
| SkinCancer | 64.7% | 57.1% | 60.7% | 77 |
| Vasculitis | 56.1% | 71.2% | 62.7% | 52 |
| Warts | 73.2% | 64.1% | 68.3% | 64 |

Recall per class on the validation split: Acne 54.5%, Actinic_Keratosis 86.0%, Benign_tumors 71.5%, Eczema 52.6%, Lupus 45.2%, SkinCancer 74.4%, Vasculitis 46.4%, Warts 47.2%. Confusion matrices are produced by `evaluate.py` (`confusion_matrix.png`).

### Comparison with earlier models

The project started with a MobileNetV2 model provided by a senior, with **no record of how or on what data it was trained**. It was replaced by a documented retrain, and the intermediate version is kept for comparison.

| | Provided model | v2 (MobileNetV2, 224 px, retrained) | **v3 (EfficientNetV2-S, 288 px)** |
|---|---:|---:|---:|
| test accuracy | 62.7% | 61.4% | **71.7%** |
| test balanced accuracy | 57.6% | 57.9% | **67.8%** |
| test SkinCancer recall | 46.8% | 57.1% | **57.1%** |
| valid accuracy | 67.1%* | 56.3% | 65.4% |
| valid SkinCancer recall | 17.9%* | 66.0% | **74.4%** |

\* The provided model's validation numbers are probably inflated: it may have seen those images in training, which cannot be verified. Compare models on the **test** split.

### Confidence is informative but somewhat over-optimistic
Predictions above 90% confidence make up about 45% of all images and were correct 83.5% of the time on `valid` and 92.9% on `test`. The result page therefore shows "High confidence" only above 0.90. A calibration step (temperature scaling) is a suggested improvement.

## Triage rules

Triage is deliberately **not** done by the LLM. It lives in [`triage_rules.py`](triage_rules.py) as readable rules.

1. **Baseline urgency per class:** Low: Acne, Eczema, Warts. Medium: Benign_tumors, Actinic_Keratosis. High: SkinCancer, Lupus, Vasculitis.
2. **Escalation to at least Medium** when: the confidence is below 0.55; the top two classes are within 0.20 of each other; or a "watch" class (SkinCancer, Actinic_Keratosis, Lupus, Vasculitis) gets at least 0.10 probability without being the top result.
3. **Follow-up answers:** a "yes" to bleeding / oozing / not healing, recent change or spreading, or fever / joint pain / tiredness is a red flag. For lesion-type classes any red flag raises the level to High; for other classes it raises it to at least Medium, and two or more red flags raise it to High.
4. Every result also lists "see a doctor if..." warning signs for the predicted class.

**Measured behaviour (no follow-up answers given):**

| | valid | test |
|---|---:|---:|
| True SkinCancer images flagged Medium or High | 100% | 98.7% |
| True SkinCancer images the CNN alone got right | 74.4% | 57.1% |
| Low-risk-class images flagged Medium or High (false alarms) | 60.0% | 45.6% |

The triage catches most cases the classifier misses, at the price of many false alarms. That is a deliberate safety-first trade-off. The thresholds (`RISK_PROB_FLOOR`, `CLOSE_MARGIN`) are engineering settings chosen from these measurements, and `evaluate.py` prints sweeps so they can be re-tuned.

> The class baselines and the "see a doctor if..." texts are conservative general guidance written by the project author, **not** reviewed by a clinician. They should be checked against sources such as the AAD or NHS patient pages before any real-world use.

## Limitations

- **Only 8 conditions.** The model has no "healthy / other" class. A photo of healthy skin, or of a condition it does not know (psoriasis, ringworm, vitiligo...), is still labelled as one of the 8, sometimes confidently.
- **Modest accuracy.** About 72% on the test split. Lupus and Vasculitis are the weakest classes (precision 34% to 56%, recall about 45%). Benign tumours, skin cancer and actinic keratosis are often confused with each other.
- **The classifier misses many skin cancers on its own** (57% to 74% recall, depending on the split). The triage layer exists to compensate, and it is itself not clinically validated.
- **Dataset quality.** Web-sourced images, some label noise (a few images appear under two labels) and repeated / near-duplicate images (about 17% of training images have a near-duplicate in the training set).
- **Skin-tone coverage is unknown.** The dataset has no skin-tone annotations, so performance across skin tones was not measured and may differ.
- **Confidence is somewhat over-optimistic** (see above).
- **LLM explanations can contain errors** and are only as good as the retrieved reference text. A reviewer step reduces but does not remove this.
- **Demo-grade web app:** hard-coded `admin` / `admin` login, a single shared temporary upload file (one user at a time), no HTTPS.

## Project structure

```
app.py                  Flask app (routes: /, /login, /predict, /refine, /logout)
rag_graph.py            LangGraph pipeline: triage, RAG retrieval, Gemini explanation, reviewer
triage_rules.py         Rule-based urgency levels and follow-up questions
ingest.py               Builds the Qdrant collection from the reference documents (run once)
templates/              index.html, login.html, result.html
static/                 styles and the temporary upload folder

evaluate.py             Metrics, confusion matrix, calibration, triage analysis for any model
check_overlap.py        Finds near-duplicate images across train / valid / test (leakage check)
prepare_subset.py       Extracts the 8 classes from the Kaggle dataset and zips them for Colab
fix_model_compat.py     Makes a model saved on Colab loadable by an older local Keras
skin_model_retrain.ipynb      Colab notebook for the v2 model (MobileNetV2, 224 px)
skin_model_retrain_v3.ipynb   Colab notebook for the v3 model (EfficientNetV2-S / ConvNeXt-Tiny, 288 px)
train_major_project.py  The ORIGINAL training script of the provided model (kept for reference only)

skin_model_v3_efficientnetv2s_288_compat.keras   Model used by the app
skin_model_v2*.keras, skin_major_model.keras     Earlier models (comparison only)
class_names.json        Class order of the model (must match LABELS in app.py)
eval_*/                 Saved evaluation reports (summary.txt, confusion matrices, predictions.csv)
```

## Setup

**Requirements:** Python 3.10, a Gemini API key (free tier works), and a running Qdrant instance. Tested with TensorFlow 2.20 / Keras 3.12 and `google-genai` 2.24 on Windows.

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1            # macOS / Linux: source venv/bin/activate

# 2. Install dependencies
python -m pip install -r requirements.txt
python -m pip install -U google-genai matplotlib   # if they are not in requirements.txt yet
```

Key packages: `flask`, `tensorflow` (includes Keras), `numpy`, `pillow`, `python-dotenv`, `langgraph`, `langchain-huggingface`, `langchain-qdrant`, `qdrant-client`, `sentence-transformers`, `google-genai`, and `matplotlib` (for the evaluation charts). If you change your environment, refresh the file with `pip freeze > requirements.txt`.

```powershell
# 3. Start Qdrant (for example with Docker)
docker run -p 6333:6333 qdrant/qdrant

# 4. Create .env in the project folder (see Configuration)

# 5. Load the reference documents into Qdrant, once
python ingest.py

# 6. Put the model file in the project folder, then start the app
python app.py
```

Open <http://127.0.0.1:5000> and sign in with the demo login (`admin` / `admin`). On startup you should see `✅ Model ready: ... (input 288x288)`.

**About the `_compat` model file.** The model was trained on Colab, whose Keras is newer than a typical local install, so the raw file may fail to load with `Unrecognized keyword arguments ... quantization_config`. Run `python fix_model_compat.py --model <file>.keras` once. It writes a `_compat.keras` copy that loads locally (the original is untouched). If your local Keras is new enough, the raw file loads as it is.

## Configuration

Create a `.env` file next to `app.py` (never commit it):

```dotenv
GEMINI_API_KEY=your_key_here
```

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Gemini API key. Without it the app still runs but shows only the fallback text and the triage result. |
| `GEMINI_GEN_MODELS` | `gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash` | Comma-separated models for the explanation, tried in order. |
| `GEMINI_JUDGE_MODELS` | `gemini-3.1-flash-lite,gemini-3.5-flash-lite` | Models for the reviewer step. |
| `USE_LLM_VALIDATION` | `true` | `false` uses only the free length check and skips the Gemini reviewer (saves quota). |
| `FLASK_SECRET_KEY` | `your_secret_key` | Session signing key. Change it for anything beyond a local demo. |

Other settings live at the top of the code: `MODEL_PATH` in `app.py`; the confidence threshold (0.55), retry limit and Qdrant address in `rag_graph.py`; urgency levels, thresholds and questions in `triage_rules.py`.

## Evaluating a model

`evaluate.py` expects one folder per class (folder names match the class names):

```powershell
$ds = "C:\path\to\SkinDisease"
python evaluate.py --data "$ds\test"  --model skin_model_v3_efficientnetv2s_288_compat.keras --out eval_v3_test
python evaluate.py --data "$ds\valid" --model skin_model_v3_efficientnetv2s_288_compat.keras --out eval_v3_valid
```

Each run writes to its `--out` folder: `summary.txt` (accuracy, per-class metrics, confidence and calibration, and the triage safety analysis with threshold sweeps), `confusion_matrix.png` / `confusion_matrix_normalized.png` / `confusion_matrix.csv`, `metrics_per_class.csv`, and `predictions.csv` (one row per image). `--limit N` uses at most N images per class for a quick trial, and `--map "Folder name=Class"` handles datasets with different folder names.

**Data-leakage check.** Web-scraped image datasets often contain near-duplicates. `check_overlap.py` compares perceptual hashes of the images across train / valid / test (and inside each split), also catching flipped, resized or re-compressed copies:

```powershell
python check_overlap.py --root "C:\path\to\SkinDisease"
python evaluate.py --data "$ds\test" --exclude-overlap overlap_results\overlap_pairs.csv --exclude-dist 2 --out eval_test_clean2
```

On this dataset about 4% of valid/test images had a near-identical copy in the training split (about 14% to 18% at a looser threshold, mostly look-alikes). Excluding them did not change the scores of the provided model, and for the retrained models the 94 closest matches were removed from the training set.

## Retraining the model

Training runs on Google Colab (free T4 GPU, roughly 30 to 60 minutes).

1. **Prepare the data on your laptop.** Extract only the 8 classes, drop training images that duplicate valid/test images, and make a zip:
   ```powershell
   python prepare_subset.py --root "C:\path\to\SkinDisease" --out "C:\path\to\skin8" --drop-overlap overlap_results\overlap_pairs.csv --zip
   ```
2. Upload `skin8.zip` to Google Drive at `My Drive/skin_project/skin8.zip`.
3. Open `skin_model_retrain_v3.ipynb` in Colab, choose **Runtime → Change runtime type → T4 GPU**, and run the cells top to bottom. The backbone is chosen in Step 3 (`efficientnetv2s`, `convnexttiny` or `mobilenetv2`).
4. Download the `.keras` model and `class_names.json` (Step 13), run `fix_model_compat.py`, and evaluate with `evaluate.py`.
5. Only if it is clearly better, point `MODEL_PATH` in `app.py` at it. The app reads the input size from the model automatically.

Good practice built into the notebooks: the class order is fixed and saved, the `valid` split picks the best epoch, and the `test` split stays untouched until the final report.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Unrecognized keyword arguments ... quantization_config` when loading the model | Run `python fix_model_compat.py --model <file>.keras` and use the `_compat.keras` copy. |
| `ImportError: cannot import name 'genai' from 'google'` | `python -m pip install -U google-genai` inside the virtual environment. |
| "AI explanation temporarily unavailable" on the result page | The Gemini call failed. Check `GEMINI_API_KEY` in `.env` and free-tier quota; the terminal log shows the exact error per model. The triage result is still valid. |
| Qdrant connection refused | Start Qdrant on `localhost:6333` and make sure `ingest.py` was run once. |
| Shape error mentioning 224 or 288 | An old `app.py` is still passing a fixed picture size. Use the current one, which reads the size from the model. |
| `evaluate.py: unrecognized arguments: --exclude-overlap` | An old copy of `evaluate.py`. The current one prints `evaluate.py version 3` on its first line. |
| Very slow first prediction | The model warms up at startup now; a bigger model also means about 1 to 3 seconds per image on CPU. |

## Privacy

- The uploaded photo is stored temporarily as `static/uploads/temp.jpg` and **deleted when the user returns to the home page or signs out**.
- The photo is processed locally by the classifier and is **not sent** to any external service. Gemini receives only the predicted label, the confidence, the follow-up answers and the retrieved reference text.
- The last prediction (label and probabilities) is kept in a signed session cookie until the user goes home or signs out.

## Credits

- **Dataset:** *Human Skin Diseases (Image)* on Kaggle by Youssef Mohamed (`youssefmohmmed/human-skin-diseases-image`).
- **Models and libraries:** EfficientNetV2 (Tan & Le, 2021) with ImageNet weights via Keras Applications, TensorFlow / Keras, LangGraph, Qdrant, Sentence-Transformers (`all-MiniLM-L6-v2`), Google Gemini, Flask.
- **Reference documents for the knowledge base:** _list the documents ingested by `ingest.py` here._
- **Author:** Raghavendra SB , Sarvesh S Gounder , Suhas A, Vijay Kumar , UVCE._

## Disclaimer

DermaAI is an educational project. Its output is informational only and is **not** a diagnosis, treatment advice or a substitute for a qualified dermatologist or doctor. Do not use it to decide whether or not to seek medical care.