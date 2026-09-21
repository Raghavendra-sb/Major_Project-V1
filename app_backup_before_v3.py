"""
app.py  (Updated: triage + symptom follow-ups)
----------------------------------------------
Flask app for Skin Disease AI.
CNN prediction is enriched by the LangGraph + RAG pipeline.

Flow:
    User uploads image
         ↓
    CNN predicts disease label (+ all class probabilities)
         ↓
    LangGraph pipeline (rag_graph.py):
        → confidence check
        → rule-based triage (urgency level)
        → RAG retrieval from Qdrant
        → Gemini LLM explanation
        → response validation
         ↓
    Flask renders result.html: prediction, top-3, urgency, explanation,
    and a short follow-up questionnaire
         ↓ (optional)
    User answers the questions → POST /refine → pipeline re-runs with the answers
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# ── Import the LangGraph pipeline ──
from rag_graph import run_skin_disease_graph, get_followup_questions

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "your_secret_key"  # change in production

# ── Load CNN model once at startup ──
MODEL_PATH = "skin_major_model.keras"
model = load_model(MODEL_PATH)

UPLOAD_DIR = os.path.join("static", "uploads")
SAVED_PATH = os.path.join(UPLOAD_DIR, "temp.jpg")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


def delete_uploaded_image():
    """Remove the last uploaded image (called when the user goes home or signs out)."""
    try:
        os.remove(SAVED_PATH)
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"⚠️ Could not delete uploaded image: {e}")


def render_result(graph_result, label, score, answers=None, refined=False):
    """Build result.html from a LangGraph result (used by both /predict and /refine)."""
    image_url = url_for("static", filename="uploads/temp.jpg") if os.path.exists(SAVED_PATH) else None
    return render_template(
        "result.html",
        image_url=image_url,
        label=label,
        score=score,
        llm_explanation=graph_result.get("llm_explanation", "No explanation available."),
        warning=graph_result.get("warning"),          # "low_confidence" or None
        model_used=graph_result.get("model_used"),
        triage=graph_result.get("triage"),
        top_predictions=graph_result.get("top_predictions") or [],
        questions=get_followup_questions(label),
        answers=answers or {},
        refined=refined,
    )


# class index → label (unchanged from your original)
LABELS = {
    0: "Acne",
    1: "Actinic_Keratosis",
    2: "Benign_tumors",
    3: "Eczema",
    4: "Lupus",
    5: "SkinCancer",
    6: "Vasculitis",
    7: "Warts"
}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    # Coming back home = done with the previous image.
    delete_uploaded_image()
    session.pop("last_prediction", None)
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == "admin" and password == "admin":
            session["username"] = username
            return redirect(url_for("index"))
        flash("Invalid credentials. Try admin / admin.", "error")
        return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    delete_uploaded_image()
    session.pop("last_prediction", None)
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))

    if "file" not in request.files:
        flash("No file provided.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    # ── Save uploaded image ──
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file.save(SAVED_PATH)

    # ── CNN Prediction ──
    image = preprocess_image(SAVED_PATH, target_size=(224, 224))
    preds = model.predict(image)
    predicted_idx = int(np.argmax(preds, axis=1)[0])
    predicted_label = LABELS.get(predicted_idx, "Unknown")
    confidence_score = float(np.max(preds))

    # All class probabilities as fractions (0-1), e.g. [("Acne", 0.929), ...]
    all_probs = [(LABELS.get(i, f"Class {i}"), float(p)) for i, p in enumerate(preds[0])]
    top3 = sorted(all_probs, key=lambda x: x[1], reverse=True)[:3]

    # ── Print all probabilities to terminal (kept from your original) ──
    print("\n===== CNN Prediction Results =====")
    for disease, prob in sorted(all_probs, key=lambda x: x[1], reverse=True):
        print(f"  {disease}: {prob * 100:.2f}%")
    print(f"  Predicted class: {predicted_label}")
    print(f"  Confidence: {confidence_score * 100:.4f}%")
    print("==================================\n")

    # Remember this prediction so /refine can re-use it (session cookie is signed by secret_key)
    session["last_prediction"] = {
        "label": predicted_label,
        "score": confidence_score,
        "top": [[name, prob] for name, prob in top3],
    }

    # ── LangGraph + RAG Pipeline ──
    print("🚀 Handing off to LangGraph pipeline...")
    graph_result = run_skin_disease_graph(
        disease_label=predicted_label,
        confidence_score=confidence_score,
        top_predictions=all_probs,
    )

    return render_result(graph_result, predicted_label, confidence_score)


@app.route("/refine", methods=["POST"])
def refine():
    """Re-run the pipeline for the SAME image, now with the user's follow-up answers."""
    if "username" not in session:
        return redirect(url_for("login"))

    last = session.get("last_prediction")
    if not last:
        flash("Please upload an image first.", "error")
        return redirect(url_for("index"))

    label = last["label"]
    score = float(last["score"])
    top = [(name, prob) for name, prob in last["top"]]

    print(f"🗣️  Refining {label} with follow-up answers: {dict(request.form)}")
    graph_result = run_skin_disease_graph(
        disease_label=label,
        confidence_score=score,
        top_predictions=top,
        symptoms=request.form,      # unknown fields / values are dropped inside the graph
    )

    return render_result(
        graph_result, label, score,
        answers=graph_result.get("symptoms") or {},
        refined=True,
    )


if __name__ == "__main__":
    app.run(debug=True)