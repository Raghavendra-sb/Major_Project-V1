"""
rag_graph.py  (Gemini + triage + symptom follow-ups)
----------------------------------------------------
LangGraph workflow for the Skin Disease AI system.

Pipeline:
    CNN label + confidence + top predictions (+ optional symptom answers)
         ↓
    Node 1: check_confidence
         ↓
    Node 2: triage                  (rule-based urgency, NO API call, always runs)
         ↓ (low confidence → END with warning)
    Node 3: retrieve_context        (RAG from Qdrant, local MiniLM embeddings)
         ↓
    Node 4: generate_explanation    (Gemini Flash-Lite primary, Flash as quality fallback)
         ↓
    Node 5: validate_response       (word-count check + Gemini Flash-Lite as reviewer)
         ↓ (insufficient → increment_retry → re-retrieve + regenerate with feedback, max 2 retries)
         ↓ (all Gemini calls failed → skip retries, go straight to final response)
    Node 6: build_final_response → END

Models per step (all configurable from .env, see the CONFIG section):
    generate_explanation : gemini-3.5-flash-lite  → gemini-3.1-flash-lite → gemini-3.6-flash
    validate_response    : gemini-3.1-flash-lite  → gemini-3.5-flash-lite

Why this order: on the free tier the Flash-Lite models allow ~500 requests/day,
while the Flash models allow only ~20/day. Each model has its OWN quota bucket,
so if one is rate-limited (HTTP 429) the code simply moves on to the next one.

Public helpers for app.py:
    run_skin_disease_graph(label, confidence, top_predictions=None, symptoms=None)
    quick_triage(label, confidence, top_predictions=None, symptoms=None)   # instant, no LLM
    get_followup_questions(label)                                         # for the HTML form
"""

import os
import json
import time
from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from google.genai import types

from triage_rules import (
    assess,
    clean_symptoms,
    describe_symptoms,
    get_questions,
    normalize_top_predictions,
    symptom_search_terms,
)


load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def _model_list(env_name: str, default: str) -> list[str]:
    """Read a comma-separated model list from .env, falling back to the default."""
    raw = os.getenv(env_name, default)
    return [m.strip() for m in raw.split(",") if m.strip()]


# Tried in order; the next model is used if the previous one errors or is rate-limited.
GENERATION_MODELS = _model_list(
    "GEMINI_GEN_MODELS",
    "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash",
)
JUDGE_MODELS = _model_list(
    "GEMINI_JUDGE_MODELS",
    "gemini-3.1-flash-lite,gemini-3.5-flash-lite",
)

# Set USE_LLM_VALIDATION=false in .env to use only the free word-count check.
USE_LLM_VALIDATION = os.getenv("USE_LLM_VALIDATION", "true").strip().lower() == "true"

MIN_WORDS = 80                 # explanations shorter than this are rejected
CONFIDENCE_THRESHOLD = 0.55    # below this → "upload a clearer image" response
MAX_RETRIES = 2


# ─────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("⚠️  GEMINI_API_KEY not found in .env — explanations will use the fallback text.")


def _call_gemini(
    prompt: str,
    models: list[str],
    system_instruction: str | None = None,
    json_mode: bool = False,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """
    Call Gemini, trying each model in `models` until one returns text.
    Returns (text, model_that_worked). Raises RuntimeError if every model fails.

    NOTE: temperature / top_p / top_k are deliberately NOT set — the Gemini 3.5+
    models ignore or discourage custom values for them.
    max_tokens is generous because "thinking" tokens can count toward the limit.
    """
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is missing")

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else None,
    )

    last_error = None
    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                text = (response.text or "").strip()
                if not text:
                    raise ValueError("empty response (blocked or ran out of tokens)")
                return text, model

            except Exception as e:  # noqa: BLE001 - we want to catch every API failure
                last_error = e
                code = getattr(e, "code", None)
                print(f"⚠️  {model} failed (code={code}): {str(e)[:160]}")
                # Temporary server-side trouble → one quick retry on the same model.
                if code in (500, 502, 503, 504) and attempt == 0:
                    time.sleep(2)
                    continue
                break  # 429 / 404 / bad request / empty → move to the next model

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


# ─────────────────────────────────────────────
# Qdrant + Embeddings
# (The collection was built with MiniLM, so queries MUST use the same embedding
#  model. Don't switch to Gemini embeddings without re-running ingest.py,
#  because the vector sizes differ.)
# ─────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="skin_disease_vectors",
    embedding=embedding_model,
)


# ─────────────────────────────────────────────
# State definition
# NOTE: LangGraph drops keys that are not declared here,
#       so every field used by a node must be listed.
# ─────────────────────────────────────────────
class SkinDiseaseState(TypedDict):
    disease_label: str
    confidence_score: float
    top_predictions: list[dict] | None  # [{"label", "probability"}] top 3, high → low
    symptoms: dict | None               # cleaned follow-up answers, e.g. {"bleeding": "yes"}
    triage: dict | None                 # output of triage_rules.assess()
    retrieved_context: str | None
    llm_explanation: str | None
    model_used: str | None              # which Gemini model wrote the explanation
    generation_failed: bool | None      # True if every Gemini model failed
    validation_feedback: str | None     # reviewer's complaint, fed into the retry
    is_sufficient: bool | None
    retry_count: int
    final_response: dict | None


def _pretty(label: str) -> str:
    """'Actinic_Keratosis' → 'Actinic Keratosis' (better for search + prompts)."""
    return label.replace("_", " ")


# ─────────────────────────────────────────────
# Node 1: Check CNN confidence score
# ─────────────────────────────────────────────
def check_confidence(state: SkinDiseaseState) -> SkinDiseaseState:
    print(f"⚙️  Checking confidence for: {state['disease_label']} ({state['confidence_score']:.2%})")
    return state


# ─────────────────────────────────────────────
# Node 2: Rule-based triage (no API call, always runs, even for low confidence)
# ─────────────────────────────────────────────
def run_triage(state: SkinDiseaseState) -> SkinDiseaseState:
    result = assess(
        label=state["disease_label"],
        confidence=state["confidence_score"],
        top_predictions=state.get("top_predictions"),
        symptoms=state.get("symptoms"),
        low_conf_threshold=CONFIDENCE_THRESHOLD,
    )
    state["triage"] = result
    print(f"🚦 Triage: {result['level'].upper()}")
    for reason in result["reasons"]:
        print(f"   • {reason}")
    return state


def route_confidence(state: SkinDiseaseState) -> Literal["retrieve_context", "low_confidence_response"]:
    if state["confidence_score"] < CONFIDENCE_THRESHOLD:
        print("⚠️  Low confidence — routing to fallback response")
        return "low_confidence_response"
    print("✅  Confidence OK — routing to RAG retrieval")
    return "retrieve_context"


# ─────────────────────────────────────────────
# Node 3a: Low confidence fallback
# ─────────────────────────────────────────────
def low_confidence_response(state: SkinDiseaseState) -> SkinDiseaseState:
    print("💬 Generating low-confidence fallback response...")
    state["final_response"] = {
        "disease_label": state["disease_label"],
        "confidence_score": state["confidence_score"],
        "llm_explanation": (
            f"The model detected a possible case of {_pretty(state['disease_label'])} "
            f"but with low confidence ({state['confidence_score']:.1%}). "
            "Please upload a clearer, well-lit image of the affected skin area "
            "for a more accurate analysis. Consult a dermatologist for professional diagnosis."
        ),
        "retrieved_context": None,
        "model_used": None,
        "triage": state.get("triage"),
        "top_predictions": state.get("top_predictions"),
        "symptoms": state.get("symptoms"),
        "warning": "low_confidence",
    }
    return state


# ─────────────────────────────────────────────
# Node 3b: RAG retrieval from Qdrant
# ─────────────────────────────────────────────
def retrieve_context(state: SkinDiseaseState) -> SkinDiseaseState:
    disease = _pretty(state["disease_label"])
    retry = state.get("retry_count", 0)
    extra = symptom_search_terms(state.get("symptoms"))   # what the user reported

    if retry > 0:
        query = f"skin disease symptoms treatment prevention general dermatology {disease} {extra}"
        print(f"🔄 Retry {retry}: broader query for {disease}")
    else:
        query = f"{disease} skin disease causes symptoms treatments prevention when to see a doctor {extra}"
        print(f"🔍 RAG retrieval for: {disease}" + (f" (+ symptoms: {extra})" if extra else ""))

    search_results = vector_db.similarity_search(query=query.strip(), k=4)

    context = "\n\n".join([
        f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}"
        for result in search_results
    ])

    state["retrieved_context"] = context
    print(f"📄 Retrieved {len(search_results)} chunks from Qdrant")
    return state


# ─────────────────────────────────────────────
# Node 4: Explanation via Gemini
# ─────────────────────────────────────────────
def generate_explanation(state: SkinDiseaseState) -> SkinDiseaseState:
    print(f"🤖 Generating explanation via Gemini (models: {', '.join(GENERATION_MODELS)})...")

    disease    = _pretty(state["disease_label"])
    context    = state["retrieved_context"]
    confidence = state["confidence_score"]
    feedback   = state.get("validation_feedback")
    triage     = state.get("triage") or {}
    symptom_lines = describe_symptoms(state.get("symptoms"))

    system_instruction = (
        "You are a medical information assistant specializing in dermatology. "
        "Use ONLY the CONTEXT below for sections 1 to 5. If the context does not cover a point, "
        "say that it is not covered instead of guessing. "
        "For section 6 (when to seek medical attention) use the SAFETY GUIDANCE in the user message, "
        "even if the context does not cover it. "
        "This is educational information, not a diagnosis: never state a definitive diagnosis "
        "and never give specific drug doses. "
        "Always recommend consulting a qualified dermatologist. "
        "Write in plain text only: no Markdown symbols (no asterisks, no # headings). "
        "Put each numbered section title on its own line and use '-' for short bullet points. "
        "Be empathetic and easy to understand, and keep the whole answer under about 350 words.\n\n"
        f"CONTEXT:\n{context}"
    )

    prompt = (
        f"A skin-image classifier predicted: {disease} (confidence {confidence:.1%}). "
        f"The classifier can be wrong, so keep the tone informational.\n\n"
        f"Please provide:\n"
        f"1. What {disease} is\n"
        f"2. Common causes\n"
        f"3. Key symptoms\n"
        f"4. Recommended treatments\n"
        f"5. Precautions and prevention\n"
        f"6. When to seek medical attention\n"
    )

    if symptom_lines:
        prompt += "\nThe user also reported:\n" + "\n".join(f"- {line}" for line in symptom_lines) + "\n"
        prompt += "Relate these to the condition only where the context supports it.\n"

    if triage:
        prompt += (
            f"\nSAFETY GUIDANCE (written by safety rules, never contradict it):\n"
            f"Urgency: {triage['level'].upper()}. {triage['message']}\n"
            f"{triage['watch_for_title']}\n"
            + "\n".join(f"- {item}" for item in triage["watch_for"])
            + "\n"
        )
        if triage["level"] == "high":
            prompt += "Do not reassure the user that this is harmless. Clearly advise seeing a doctor soon.\n"
        elif triage["level"] == "medium":
            prompt += "Clearly advise having this checked by a doctor or dermatologist.\n"

    if feedback:
        prompt += f"\nA reviewer rejected the previous attempt because: {feedback}\nFix this in your new answer.\n"

    try:
        text, used = _call_gemini(
            prompt=prompt,
            models=GENERATION_MODELS,
            system_instruction=system_instruction,
            max_tokens=4096,
        )
        state["llm_explanation"] = text
        state["model_used"] = used
        state["generation_failed"] = False
        print(f"✅ Explanation generated by {used}.")

    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Gemini error: {e}")
        state["generation_failed"] = True
        state["model_used"] = None
        state["llm_explanation"] = (
            f"AI explanation temporarily unavailable.\n\n"
            f"Detected: {disease} — Confidence: {confidence:.1%}\n\n"
            f"Please consult a qualified dermatologist for proper diagnosis."
        )

    return state


# ─────────────────────────────────────────────
# Node 5: Validate LLM response quality
#   Step 1: free word-count check
#   Step 2: Gemini Flash-Lite reviews completeness + safety
# ─────────────────────────────────────────────
def _judge_explanation(disease: str, urgency: str, context: str, explanation: str) -> tuple[bool, str]:
    """Ask a cheap Gemini model to review the answer. Returns (is_ok, reason)."""
    prompt = (
        "You are a strict reviewer of a dermatology information answer.\n\n"
        f"Condition: {disease}\n"
        f"Urgency decided by safety rules: {urgency.upper()}\n\n"
        f"RETRIEVED CONTEXT (truncated):\n{(context or '')[:6000]}\n\n"
        f"ANSWER TO REVIEW:\n{explanation}\n\n"
        "Check ALL of these:\n"
        "1. It covers: what the condition is, causes, symptoms, treatments, prevention, "
        "and when to seek medical attention.\n"
        "2. It does not present a definitive diagnosis and gives no specific drug doses.\n"
        "3. It recommends consulting a qualified dermatologist.\n"
        "4. It does not contradict the retrieved context.\n"
        "5. Its advice on seeing a doctor is consistent with the urgency level above "
        "(for MEDIUM or HIGH it must not say the condition is harmless or that a visit is unnecessary).\n\n"
        'Reply with JSON only, in this exact shape: {"sufficient": true, "reason": "one short sentence"}'
    )

    text, used = _call_gemini(prompt=prompt, models=JUDGE_MODELS, json_mode=True, max_tokens=1024)

    # Be tolerant if the model wraps the JSON in ``` fences.
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    if isinstance(data, list) and data:
        data = data[0]
    ok = bool(data.get("sufficient"))
    reason = str(data.get("reason", "")).strip()
    print(f"🧑‍⚖️ Reviewer ({used}): {'PASS' if ok else 'FAIL'} — {reason}")
    return ok, reason


def validate_response(state: SkinDiseaseState) -> SkinDiseaseState:
    print("🔎 Validating response quality...")
    explanation = state["llm_explanation"] or ""
    word_count = len(explanation.split())

    # If every Gemini model failed, there is nothing worth reviewing.
    if state.get("generation_failed"):
        state["is_sufficient"] = False
        state["validation_feedback"] = None
        print("⚠️ Skipping validation — generation failed.")
        return state

    # Step 1: cheap length check (saves an API call when the answer is clearly too short)
    if word_count <= MIN_WORDS:
        state["is_sufficient"] = False
        state["validation_feedback"] = (
            f"The answer was too short ({word_count} words). Cover all six topics in more detail."
        )
        print(f"📊 Too short ({word_count} words).")
        return state

    print(f"📊 Length OK — {word_count} words.")

    # Step 2: LLM reviewer (optional)
    if not USE_LLM_VALIDATION:
        state["is_sufficient"] = True
        state["validation_feedback"] = None
        return state

    try:
        ok, reason = _judge_explanation(
            _pretty(state["disease_label"]),
            (state.get("triage") or {}).get("level", "low"),
            state["retrieved_context"] or "",
            explanation,
        )
        state["is_sufficient"] = ok
        state["validation_feedback"] = None if ok else (reason or "The answer was incomplete or unsafe.")
    except Exception as e:  # noqa: BLE001
        # Reviewer unavailable (quota, bad JSON, ...) → trust the length check.
        print(f"⚠️ Reviewer unavailable ({str(e)[:120]}); accepting based on length check.")
        state["is_sufficient"] = True
        state["validation_feedback"] = None

    return state


# ─────────────────────────────────────────────
# Router: after validate_response
# NOTE: State mutation here is NOT saved by LangGraph.
#       Increment happens inside the increment_retry NODE instead.
# ─────────────────────────────────────────────
def route_validation(state: SkinDiseaseState) -> Literal["build_final_response", "increment_retry"]:
    # If every Gemini model failed, retrying immediately won't help.
    if state.get("generation_failed"):
        print("⚠️ Generation failed — skipping retries.")
        return "build_final_response"

    retry = state.get("retry_count", 0)
    if not state["is_sufficient"] and retry < MAX_RETRIES:
        print(f"🔄 Response insufficient (retry_count={retry}), will retry...")
        return "increment_retry"
    print("✅ Moving to final response.")
    return "build_final_response"


# ─────────────────────────────────────────────
# Node 5b: Increment retry counter (must be a NODE, not done in router)
# LangGraph only persists state changes made inside nodes,
# not inside routing functions.
# ─────────────────────────────────────────────
def increment_retry(state: SkinDiseaseState) -> SkinDiseaseState:
    state["retry_count"] = state.get("retry_count", 0) + 1
    print(f"🔁 Retry attempt {state['retry_count']}")
    return state


# ─────────────────────────────────────────────
# Node 6: Build final structured response
# ─────────────────────────────────────────────
def build_final_response(state: SkinDiseaseState) -> SkinDiseaseState:
    print("📦 Building final response...")
    state["final_response"] = {
        "disease_label": state["disease_label"],
        "confidence_score": state["confidence_score"],
        "llm_explanation": state["llm_explanation"],
        "retrieved_context": state["retrieved_context"],
        "model_used": state.get("model_used"),
        "triage": state.get("triage"),
        "top_predictions": state.get("top_predictions"),
        "symptoms": state.get("symptoms"),
        "warning": None,
    }
    return state


# ─────────────────────────────────────────────
# Build the LangGraph
# ─────────────────────────────────────────────
graph_builder = StateGraph(SkinDiseaseState)

# Register all nodes
graph_builder.add_node("check_confidence",        check_confidence)
graph_builder.add_node("triage",                  run_triage)
graph_builder.add_node("low_confidence_response", low_confidence_response)
graph_builder.add_node("retrieve_context",        retrieve_context)
graph_builder.add_node("generate_explanation",    generate_explanation)
graph_builder.add_node("validate_response",       validate_response)
graph_builder.add_node("increment_retry",         increment_retry)
graph_builder.add_node("build_final_response",    build_final_response)

# Wire edges
graph_builder.add_edge(START,                      "check_confidence")
graph_builder.add_edge("check_confidence",         "triage")
graph_builder.add_conditional_edges("triage",      route_confidence)
graph_builder.add_edge("low_confidence_response",  END)
graph_builder.add_edge("retrieve_context",         "generate_explanation")
graph_builder.add_edge("generate_explanation",     "validate_response")
graph_builder.add_conditional_edges("validate_response", route_validation)
graph_builder.add_edge("increment_retry",          "retrieve_context")  # loops back
graph_builder.add_edge("build_final_response",     END)

graph = graph_builder.compile()


# ─────────────────────────────────────────────
# Public helpers called by Flask app.py
# ─────────────────────────────────────────────
def get_followup_questions(disease_label: str) -> list[dict]:
    """The 4 multiple-choice questions for this class: [{id, text, options:[{value,label}]}]."""
    return get_questions(disease_label)


def quick_triage(
    disease_label: str,
    confidence_score: float,
    top_predictions=None,
    symptoms=None,
) -> dict:
    """Instant rule-based triage without running the graph or calling any API."""
    return assess(
        label=disease_label,
        confidence=confidence_score,
        top_predictions=top_predictions,
        symptoms=clean_symptoms(symptoms, disease_label),
        low_conf_threshold=CONFIDENCE_THRESHOLD,
    )


def run_skin_disease_graph(
    disease_label: str,
    confidence_score: float,
    top_predictions=None,
    symptoms=None,
) -> dict:
    """
    Entry point called from Flask after CNN prediction.

    Args:
        disease_label:    e.g. "Acne", "Eczema"
        confidence_score: float between 0 and 1, e.g. 0.93
        top_predictions:  optional, all class probabilities as [(label, prob), ...]
                          (the top 3 are used). Works with or without it.
        symptoms:         optional follow-up answers, e.g. request.form or
                          {"bleeding": "yes", "duration": "weeks"}. Unknown keys/values are dropped.

    Returns:
        dict with keys: disease_label, confidence_score, llm_explanation,
                        retrieved_context, model_used, triage, top_predictions,
                        symptoms, warning
    """
    print(f"\n===== LangGraph Pipeline Start =====")
    print(f"Disease: {disease_label} | Confidence: {confidence_score:.2%}")

    cleaned = clean_symptoms(symptoms, disease_label)
    top3 = normalize_top_predictions(top_predictions, disease_label, confidence_score)

    initial_state: SkinDiseaseState = {
        "disease_label":       disease_label,
        "confidence_score":    confidence_score,
        "top_predictions":     top3,
        "symptoms":            cleaned,
        "triage":              None,
        "retrieved_context":   None,
        "llm_explanation":     None,
        "model_used":          None,
        "generation_failed":   None,
        "validation_feedback": None,
        "is_sufficient":       None,
        "retry_count":         0,
        "final_response":      None,
    }

    result = graph.invoke(initial_state)
    print("===== LangGraph Pipeline End =====\n")
    return result["final_response"]
