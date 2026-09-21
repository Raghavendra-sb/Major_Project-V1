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

* [Features](#features)
* [How it works](#how-it-works)
* [The classifier and its results](#the-classifier-and-its-results)
* [Triage rules](#triage-rules)
* [Limitations](#limitations)
* [Project structure](#project-structure)
* [Setup](#setup)
* [Configuration](#configuration)
* [Evaluating a model](#evaluating-a-model)
* [Retraining the model](#retraining-the-model)
* [Troubleshooting](#troubleshooting)
* [Privacy](#privacy)
* [Credits](#credits)

---

## Features

* Image upload (JPG / PNG / WEBP, up to 10 MB) with a dark, single-page UI and login screen (demo login).
* **Top-3 predictions** with probabilities.
* **Urgency badge** with the reasons behind it and "see a doctor if..." warning signs, produced by rules and not by the LLM.
* **RAG explanation**: retrieval from a Qdrant vector database of dermatology reference documents, then an explanation from Gemini with an automatic reviewer step that can reject and regenerate weak answers.
* **Follow-up questionnaire** (4 multiple-choice questions chosen by the predicted class) that can raise the urgency level and steer retrieval.
* **Graceful failures**: if every LLM call fails or a quota is hit, the page still shows the prediction and the full triage result.
* **Evaluation tooling**: confusion matrices, per-class metrics, calibration, a near-duplicate (data leakage) checker, and a triage-threshold analysis.

---

## How it works

```text
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
                                             │
                                             │ (optional) answers
                                             ▼
                                      POST /refine
                                             │
                                             ▼
                                      pipeline runs again
```

### Models used per step

All models are configurable in `.env`. Each model is tried in order until one works, which also helps spread free-tier quota.

| Step                   | Models                                                                 |
| ---------------------- | ---------------------------------------------------------------------- |
| Explanation            | `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → `gemini-3.6-flash` |
| Reviewer               | `gemini-3.1-flash-lite` → `gemini-3.5-flash-lite`                      |
| Embeddings (retrieval) | `sentence-transformers/all-MiniLM-L6-v2`, run locally                  |

The LLM never sees the photo. It only receives the predicted label, the confidence, the follow-up answers, and the retrieved reference text.

---

## The classifier and its results

### Classes

The model predicts 8 classes:

* Acne
* Actinic_Keratosis
* Benign_tumors
* Eczema
* Lupus
* SkinCancer
* Vasculitis
* Warts

### Model

**EfficientNetV2-S** (ImageNet-pretrained) → Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(8, softmax).

Input is a **288 × 288** photo scaled to 0–1. The model rescales internally to what the backbone expects.

### Training

Training was performed using Google Colab with a T4 GPU.

* **Phase 1:** train the new head with the backbone frozen using Adam with learning rate `1e-3`.
* **Phase 2:** fine-tune the last 40% of the backbone layers while keeping BatchNorm frozen, using AdamW with learning rate `2e-5`.
* Early stopping was applied using validation loss.
* Class weights were balanced.
* Data augmentation included:

  * Horizontal/vertical flips
  * ±20° rotation
  * Zoom
  * Contrast adjustment
  * Brightness adjustment
* Mixed-precision training was used and the model was converted back to float32 before saving.
* The `valid` split was used to select the best epoch.
* The `test` split was never used for training decisions.

### Dataset

The project uses the Kaggle dataset **Human Skin Diseases (Image)**:

`youssefmohmmed/human-skin-diseases-image`

Eight of its 21 classes are used along with its original train / valid / test split.

> Check the dataset's license on Kaggle before redistributing anything derived from it.

### Dataset distribution

| Class             |    Train |    Valid |    Test |
| ----------------- | -------: | -------: | ------: |
| Acne              |      450 |      132 |      65 |
| Actinic_Keratosis |     1196 |      300 |     167 |
| Benign_tumors     |      886 |      228 |     121 |
| Eczema            |      762 |      228 |     112 |
| Lupus             |      218 |       84 |      34 |
| SkinCancer        |      769 |      156 |      77 |
| Vasculitis        |      373 |       84 |      52 |
| Warts             |      495 |       72 |      64 |
| **Total**         | **5149** | **1284** | **692** |

94 training images that were near-copies of validation/test images were removed. See [Evaluating a model](#evaluating-a-model).

---

### Results

Measured using `evaluate.py` on the final model.

| Metric                  | Valid (1284 images) | Test (692 images) |
| ----------------------- | ------------------: | ----------------: |
| Accuracy                |               65.4% |         **71.7%** |
| Balanced accuracy       |               59.7% |         **67.8%** |
| Macro F1                |               59.5% |         **67.6%** |
| Top-3 accuracy          |               93.1% |         **94.1%** |
| SkinCancer recall       |  74.4% (116 of 156) |  57.1% (44 of 77) |
| Calibration error (ECE) |               0.140 |             0.079 |

### Per-class results on the test split

| Class             | Precision | Recall |    F1 | Images |
| ----------------- | --------: | -----: | ----: | -----: |
| Acne              |     82.0% |  76.9% | 79.4% |     65 |
| Actinic_Keratosis |     91.4% |  82.6% | 86.8% |    167 |
| Benign_tumors     |     62.9% |  78.5% | 69.9% |    121 |
| Eczema            |     76.0% |  67.9% | 71.7% |    112 |
| Lupus             |     38.5% |  44.1% | 41.1% |     34 |
| SkinCancer        |     64.7% |  57.1% | 60.7% |     77 |
| Vasculitis        |     56.1% |  71.2% | 62.7% |     52 |
| Warts             |     73.2% |  64.1% | 68.3% |     64 |

Recall per class on the validation split:

* Acne: 54.5%
* Actinic_Keratosis: 86.0%
* Benign_tumors: 71.5%
* Eczema: 52.6%
* Lupus: 45.2%
* SkinCancer: 74.4%
* Vasculitis: 46.4%
* Warts: 47.2%

Confusion matrices are produced by `evaluate.py` as `confusion_matrix.png`.

---

## Comparison with earlier models

The project started with a MobileNetV2 model provided by a senior, with **no record of how or on what data it was trained**.

It was replaced by a documented retrain, and the intermediate version is kept for comparison.

| Metric                  | Provided model | v2 MobileNetV2, 224 px, retrained | **v3 EfficientNetV2-S, 288 px** |
| ----------------------- | -------------: | --------------------------------: | ------------------------------: |
| Test accuracy           |          62.7% |                             61.4% |                       **71.7%** |
| Test balanced accuracy  |          57.6% |                             57.9% |                       **67.8%** |
| Test SkinCancer recall  |          46.8% |                             57.1% |                       **57.1%** |
| Valid accuracy          |         67.1%* |                             56.3% |                           65.4% |
| Valid SkinCancer recall |         17.9%* |                             66.0% |                       **74.4%** |

* The provided model's validation numbers are probably inflated because it may have seen those images during training. This cannot be verified. Models should therefore be compared using the **test split**.

---

## Confidence calibration

Predictions above 90% confidence make up about 45% of all images.

These predictions were correct:

* **83.5%** of the time on the validation split.
* **92.9%** of the time on the test split.

The result page therefore shows **"High confidence"** only above `0.90`.

A calibration step such as **temperature scaling** is a suggested future improvement.

---

## Triage rules

Triage is deliberately **not performed by the LLM**.

It lives in [`triage_rules.py`](triage_rules.py) as readable, deterministic rules.

### 1. Baseline urgency per class

| Urgency | Classes                          |
| ------- | -------------------------------- |
| Low     | Acne, Eczema, Warts              |
| Medium  | Benign_tumors, Actinic_Keratosis |
| High    | SkinCancer, Lupus, Vasculitis    |

### 2. Escalation to at least Medium

Urgency is raised to at least Medium when:

* Confidence is below `0.55`.
* The top two classes are within `0.20` of each other.
* A "watch" class gets at least `0.10` probability without being the top result.

Watch classes:

* SkinCancer
* Actinic_Keratosis
* Lupus
* Vasculitis

### 3. Follow-up answers

The following answers are treated as red flags:

* Bleeding / oozing / not healing
* Recent change or spreading
* Fever / joint pain / tiredness

For lesion-type classes, any red flag raises the level to **High**.

For other classes, a red flag raises the level to at least **Medium**.

Two or more red flags raise the level to **High**.

### 4. Warning signs

Every result also lists **"see a doctor if..."** warning signs for the predicted class.

---

## Measured triage behaviour

No follow-up answers were provided for this analysis.

| Metric                                                   | Valid |  Test |
| -------------------------------------------------------- | ----: | ----: |
| True SkinCancer images flagged Medium or High            |  100% | 98.7% |
| True SkinCancer images correctly classified by CNN alone | 74.4% | 57.1% |
| Low-risk-class images flagged Medium or High             | 60.0% | 45.6% |

The triage system flags most SkinCancer cases, including many that the classifier itself misses, at the cost of a significant number of false alarms.

The thresholds `RISK_PROB_FLOOR` and `CLOSE_MARGIN` are engineering settings selected from these measurements. `evaluate.py` provides threshold sweeps so they can be re-tuned.

> The class baselines and "see a doctor if..." texts are conservative general guidance written by the project author, **not reviewed by a clinician**. They should be checked against authoritative medical sources such as the AAD or NHS before any real-world use.

---

## Limitations

* **Only 8 conditions.** The model has no "healthy / other" class. A photo of healthy skin or an unknown condition such as psoriasis, ringworm, or vitiligo may still be classified as one of the 8 known classes, sometimes with high confidence.
* **Modest accuracy.** Test accuracy is about 72%. Lupus and Vasculitis are the weakest classes, with precision ranging from approximately 34% to 56% and recall around 45%.
* **Class confusion.** Benign tumors, skin cancer, and actinic keratosis are often confused with each other.
* **Skin cancer misses.** The classifier alone misses a substantial number of SkinCancer cases, with recall ranging from 57% to 74% depending on the split.
* **Triage is not clinically validated.** The triage layer is an engineering safety mechanism and has not been clinically validated.
* **Dataset quality.** The dataset contains web-sourced images, some label noise, and repeated / near-duplicate images.
* **Skin-tone coverage is unknown.** The dataset has no skin-tone annotations, so performance across different skin tones was not measured and may differ.
* **Confidence is somewhat over-optimistic.**
* **LLM explanations can contain errors.** They depend on the retrieved reference text. The reviewer step can reduce weak responses but cannot guarantee correctness.
* **Demo-grade web app.** The current application uses a hard-coded `admin` / `admin` login, a single shared temporary upload file, and no HTTPS.

---

## Project structure

```text
app.py
├── Flask application
├── Routes: /, /login, /predict, /refine, /logout

rag_graph.py
└── LangGraph pipeline
    ├── Triage
    ├── Qdrant retrieval
    ├── Gemini explanation
    └── Reviewer / retry logic

triage_rules.py
└── Rule-based urgency levels and follow-up questions

ingest.py
└── Builds the Qdrant collection from reference documents

templates/
├── index.html
├── login.html
└── result.html

static/
├── styles
└── temporary upload folder

evaluate.py
└── Metrics, confusion matrix, calibration, triage analysis

check_overlap.py
└── Near-duplicate / data leakage detection

prepare_subset.py
└── Extracts the 8 classes from the Kaggle dataset

fix_model_compat.py
└── Converts models for compatibility with older local Keras

skin_model_retrain.ipynb
└── Colab notebook for v2 MobileNetV2 model

skin_model_retrain_v3.ipynb
└── Colab notebook for v3 EfficientNetV2-S / ConvNeXt-Tiny

train_major_project.py
└── Original training script for the provided model

skin_model_v3_efficientnetv2s_288_compat.keras
└── Model used by the application

skin_model_v2/*.keras
└── Earlier models

skin_major_model.keras
└── Original provided model

class_names.json
└── Class order used by the model

eval_*/
└── Saved evaluation reports
    ├── summary.txt
    ├── confusion matrices
    └── predictions.csv
```

---

## Setup

### Requirements

* Python 3.10
* Gemini API key
* Running Qdrant instance

Tested with:

* TensorFlow 2.20
* Keras 3.12
* `google-genai` 2.24
* Windows

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

For macOS / Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -U google-genai matplotlib
```

The second command is only needed if these packages are not already included in `requirements.txt`.

### Key packages

* `flask`
* `tensorflow`
* `numpy`
* `pillow`
* `python-dotenv`
* `langgraph`
* `langchain-huggingface`
* `langchain-qdrant`
* `qdrant-client`
* `sentence-transformers`
* `google-genai`
* `matplotlib`

If you change the environment, refresh the dependency file:

```powershell
pip freeze > requirements.txt
```

### 3. Start Qdrant

For example, using Docker:

```powershell
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Create `.env`

Create a `.env` file in the project folder. See [Configuration](#configuration).

### 5. Load the reference documents

Run once:

```powershell
python ingest.py
```

### 6. Start the application

Place the model file in the project folder and run:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Demo login:

```text
Username: admin
Password: admin
```

On startup you should see:

```text
✅ Model ready: ... (input 288x288)
```

---

## Model compatibility

The `_compat` model exists because the model was trained in Google Colab using a newer Keras version than may be installed locally.

If the raw `.keras` file produces:

```text
Unrecognized keyword arguments ... quantization_config
```

run:

```powershell
python fix_model_compat.py --model <file>.keras
```

This creates a `_compat.keras` copy.

The original model remains untouched.

If your local Keras version is new enough, the raw model may load without conversion.

---

## Configuration

Create a `.env` file next to `app.py`.

```dotenv
GEMINI_API_KEY=your_key_here
```

> Never commit `.env` to Git.

### Environment variables

| Variable              | Default                                                        | Purpose                              |
| --------------------- | -------------------------------------------------------------- | ------------------------------------ |
| `GEMINI_API_KEY`      | Required                                                       | Gemini API key                       |
| `GEMINI_GEN_MODELS`   | `gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash` | Explanation models tried in order    |
| `GEMINI_JUDGE_MODELS` | `gemini-3.1-flash-lite,gemini-3.5-flash-lite`                  | Reviewer models                      |
| `USE_LLM_VALIDATION`  | `true`                                                         | Enables/disables the Gemini reviewer |
| `FLASK_SECRET_KEY`    | `your_secret_key`                                              | Flask session signing key            |

Without `GEMINI_API_KEY`, the application can still run but will show fallback text instead of the generated explanation.

Other settings are located in the source code:

* `MODEL_PATH` → `app.py`
* Confidence threshold → `rag_graph.py`
* Retry limit → `rag_graph.py`
* Qdrant address → `rag_graph.py`
* Urgency thresholds → `triage_rules.py`
* Follow-up questions → `triage_rules.py`

---

## Evaluating a model

`evaluate.py` expects one folder per class, with folder names matching the class names.

Example:

```powershell
$ds = "C:\path\to\SkinDisease"

python evaluate.py `
  --data "$ds\test" `
  --model skin_model_v3_efficientnetv2s_288_compat.keras `
  --out eval_v3_test

python evaluate.py `
  --data "$ds\valid" `
  --model skin_model_v3_efficientnetv2s_288_compat.keras `
  --out eval_v3_valid
```

Each run produces:

```text
summary.txt
confusion_matrix.png
confusion_matrix_normalized.png
confusion_matrix.csv
metrics_per_class.csv
predictions.csv
```

The summary includes:

* Accuracy
* Balanced accuracy
* Per-class precision
* Per-class recall
* F1 scores
* Confidence statistics
* Calibration
* Triage analysis
* Threshold sweeps

For a quick test, use:

```text
--limit N
```

This evaluates at most `N` images per class.

If dataset folder names differ from the model class names, use:

```text
--map "Folder name=Class"
```

---

## Data leakage check

Web-sourced image datasets can contain near-duplicate images.

`check_overlap.py` compares perceptual hashes across:

* Train
* Validation
* Test

It can also detect:

* Flipped copies
* Resized copies
* Re-compressed copies
* Similar images within a split

Run:

```powershell
python check_overlap.py --root "C:\path\to\SkinDisease"
```

You can then exclude overlapping images during evaluation:

```powershell
python evaluate.py `
  --data "$ds\test" `
  --exclude-overlap overlap_results\overlap_pairs.csv `
  --exclude-dist 2 `
  --out eval_test_clean2
```

On this dataset:

* About 4% of validation/test images had a near-identical copy in training at a strict threshold.
* Approximately 14%–18% were identified at a looser threshold, mostly look-alikes.
* 94 of the closest matches were removed from the trai

