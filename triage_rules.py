"""
triage_rules.py
---------------
Rule-based triage + symptom follow-up questions for the skin disease agent.

No API calls and no ML: everything here is plain, readable rules, so it is fast,
predictable, and easy to explain in a viva.

⚠️  REVIEW BEFORE YOU DEMO THIS
    The urgency level per class and the "watch for" texts below are conservative
    GENERAL guidance written as a starting point. Check each one against a trusted
    source (AAD / NHS patient pages for that condition), ideally with a dermatologist,
    and edit the text where needed. The numeric settings in the TUNABLE section are
    engineering settings, not medical facts: set them using the confusion matrix
    from your evaluation (e.g. pick RISK_PROB_FLOOR so recall for SkinCancer is high).
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# Urgency levels
# ─────────────────────────────────────────────
LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}

LEVEL_INFO = {
    "low": {
        "title": "Low urgency",
        "message": "This is usually managed with routine care, but see a doctor if any warning sign below applies.",
    },
    "medium": {
        "title": "Medium urgency",
        "message": "Please arrange a visit with a doctor or dermatologist to have this checked.",
    },
    "high": {
        "title": "High urgency",
        "message": "Please see a doctor as soon as you can. Do not wait for it to go away on its own.",
    },
}

DISCLAIMER = (
    "This is an automated screening aid, not a diagnosis. "
    "If you are worried about your skin, see a doctor whatever the result."
)


# ─────────────────────────────────────────────
# TUNABLE engineering settings (calibrate with your evaluation results)
# ─────────────────────────────────────────────
CLOSE_MARGIN = 0.20      # top-1 and top-2 closer than this → "model is unsure between two classes"
RISK_PROB_FLOOR = 0.10   # a watch-list class at or above this probability (but not top-1) triggers a warning

# Classes where even a smaller probability deserves attention. Edit to match your CNN's class names.
WATCH_CLASSES = {"SkinCancer", "Actinic_Keratosis", "Lupus", "Vasculitis"}

# Classes that look like a spot/lump/lesion → changes or bleeding are the key red flags.
LESION_CLASSES = {"SkinCancer", "Actinic_Keratosis", "Benign_tumors", "Warts"}
# Classes that can involve the whole body → fever / joint pain / tiredness matter.
SYSTEMIC_CLASSES = {"Lupus", "Vasculitis"}


# ─────────────────────────────────────────────
# Per-class baseline urgency + "watch for" texts   (REVIEW these!)
# ─────────────────────────────────────────────
CLASS_INFO = {
    "Acne": {
        "level": "low",
        "watch_for": [
            "It is painful, deep, or leaves scars",
            "It has not improved after several weeks of over-the-counter treatment",
            "You are not sure it is acne",
        ],
    },
    "Eczema": {
        "level": "low",
        "watch_for": [
            "The skin looks infected: pus, yellow crusts, increasing redness or warmth, or you have a fever",
            "The itching disturbs your sleep or daily life",
            "It is spreading or not improving with regular moisturising",
        ],
    },
    "Warts": {
        "level": "low",
        "watch_for": [
            "It is painful, bleeds, or changes in appearance",
            "You are not sure it is a wart",
            "It is on your face or genital area",
        ],
    },
    "Benign_tumors": {
        "level": "medium",
        "watch_for": [
            "It grows, changes color or shape, bleeds, or becomes painful",
            "It is a new lump or spot and you are not sure what it is",
            "A doctor has not yet confirmed what it is",
        ],
    },
    "Actinic_Keratosis": {
        "level": "medium",
        "watch_for": [
            "A rough or scaly patch gets thicker, becomes sore, bleeds, or grows quickly",
            "A patch does not go away, especially on sun-exposed skin",
            "You have several similar patches",
        ],
    },
    "SkinCancer": {
        "level": "high",
        "watch_for": [
            "A mole or spot is growing, changes shape or color, or has an irregular border",
            "A spot bleeds, itches, or does not heal",
            "You notice a new spot that looks different from your other moles",
        ],
    },
    "Lupus": {
        "level": "high",
        "watch_for": [
            "You also have fever, joint pain or swelling, mouth sores, or unusual tiredness",
            "The rash gets worse in sunlight or keeps spreading",
            "You have chest pain or trouble breathing: seek emergency care immediately",
        ],
    },
    "Vasculitis": {
        "level": "high",
        "watch_for": [
            "You have fever, unexplained weight loss, or unusual tiredness",
            "Purple or red spots spread, blister, or turn into sores",
            "You have numbness, weakness, or severe pain",
            "You have chest pain, trouble breathing, or cough up blood: seek emergency care immediately",
        ],
    },
}

# Used if the CNN ever predicts a class that is not listed above.
DEFAULT_CLASS_INFO = {
    "level": "medium",
    "watch_for": [
        "It is painful, bleeding, spreading, or changing",
        "You are worried about it or unsure what it is",
    ],
}


# ─────────────────────────────────────────────
# Follow-up questions (multiple choice only, so answers can never inject free text
# into the search query or the LLM prompt)
# ─────────────────────────────────────────────
YES_NO = [("yes", "Yes"), ("no", "No"), ("not_sure", "Not sure")]

QUESTION_BANK = {
    "duration": {
        "short": "Duration",
        "text": "How long have you had this?",
        "options": [
            ("recent", "Just noticed it recently"),
            ("weeks", "A few weeks"),
            ("months", "Several months or longer"),
        ],
    },
    "itch_pain": {
        "short": "Itch / pain",
        "text": "Is it itchy or painful?",
        "options": [
            ("none", "Neither"),
            ("itchy", "Itchy"),
            ("painful", "Painful"),
            ("both", "Both"),
        ],
    },
    "bleeding": {
        "short": "Bleeding, oozing or not healing",
        "text": "Does it bleed, ooze, crust over, or fail to heal?",
        "options": YES_NO,
    },
    "changing": {
        "short": "Recent change or spreading",
        "text": "Has it changed recently in size, shape or color, or is it spreading?",
        "options": YES_NO,
    },
    "systemic": {
        "short": "Fever, joint pain or tiredness",
        "text": "Do you also have fever, joint pain, or unusual tiredness?",
        "options": YES_NO,
    },
}

# Which of these answers count as red flags when the answer is "yes".
RED_FLAG_QUESTIONS = ("bleeding", "changing", "systemic")


def question_ids_for(label: str) -> list[str]:
    """Pick the 4 most relevant questions for the predicted class."""
    if label in LESION_CLASSES:
        return ["duration", "itch_pain", "bleeding", "changing"]
    if label in SYSTEMIC_CLASSES:
        return ["duration", "itch_pain", "changing", "systemic"]
    return ["duration", "itch_pain", "bleeding", "systemic"]  # Acne, Eczema, unknown


def get_questions(label: str) -> list[dict]:
    """Questions ready for the HTML form: [{id, text, options: [{value, label}]}]."""
    questions = []
    for qid in question_ids_for(label):
        q = QUESTION_BANK[qid]
        questions.append({
            "id": qid,
            "text": q["text"],
            "options": [{"value": v, "label": l} for v, l in q["options"]],
        })
    return questions


def clean_symptoms(raw, label: str) -> dict:
    """
    Keep only known question ids with known answer values.
    `raw` can be a dict or Flask's request.form (anything with .get()).
    """
    if not raw or not hasattr(raw, "get"):
        return {}
    cleaned = {}
    for qid in question_ids_for(label):
        allowed = {value for value, _ in QUESTION_BANK[qid]["options"]}
        value = raw.get(qid)
        if value in allowed:
            cleaned[qid] = value
    return cleaned


def describe_symptoms(symptoms: dict | None) -> list[str]:
    """Human-readable lines for the LLM prompt, e.g. 'Duration: A few weeks'."""
    lines = []
    for qid, value in (symptoms or {}).items():
        q = QUESTION_BANK.get(qid)
        if not q:
            continue
        option_label = dict(q["options"]).get(value)
        if option_label:
            lines.append(f"{q['short']}: {option_label}")
    return lines


def symptom_search_terms(symptoms: dict | None) -> str:
    """Extra keywords appended to the RAG query so retrieval matches what the user reported."""
    s = symptoms or {}
    terms = []
    if s.get("duration") == "months":
        terms.append("chronic long-standing")
    if s.get("itch_pain") in ("itchy", "both"):
        terms.append("itching")
    if s.get("itch_pain") in ("painful", "both"):
        terms.append("pain")
    if s.get("bleeding") == "yes":
        terms.append("bleeding oozing crusting not healing")
    if s.get("changing") == "yes":
        terms.append("changes in size shape color spreading warning signs")
    if s.get("systemic") == "yes":
        terms.append("fever joint pain fatigue systemic symptoms")
    return " ".join(terms)


# ─────────────────────────────────────────────
# Triage
# ─────────────────────────────────────────────
def normalize_top_predictions(top_predictions, label: str, confidence: float, n: int = 3) -> list[dict]:
    """
    Accepts [(label, prob), ...] or [{"label":..., "probability":...}, ...] (or None)
    and returns the top-n as [{"label", "probability"}] sorted high → low.
    """
    items = []
    for item in top_predictions or []:
        if isinstance(item, dict):
            name, prob = item.get("label"), item.get("probability")
        else:
            name, prob = item[0], item[1]
        if name is not None and prob is not None:
            items.append({"label": str(name), "probability": float(prob)})
    if not items:
        items = [{"label": label, "probability": float(confidence)}]
    items.sort(key=lambda x: x["probability"], reverse=True)
    return items[:n]


def _raise_to(current: str, minimum: str) -> str:
    return minimum if LEVEL_RANK[minimum] > LEVEL_RANK[current] else current


def _pretty(label: str) -> str:
    return label.replace("_", " ")


def assess(
    label: str,
    confidence: float,
    top_predictions=None,
    symptoms: dict | None = None,
    low_conf_threshold: float = 0.55,
) -> dict:
    """
    Combine the class baseline, the model's uncertainty, and the user's answers
    into one urgency level plus the reasons behind it.
    """
    info = CLASS_INFO.get(label, DEFAULT_CLASS_INFO)
    level = info["level"]
    reasons: list[str] = []

    top = normalize_top_predictions(top_predictions, label, confidence)

    # 1) The model itself is not confident
    if confidence < low_conf_threshold:
        level = _raise_to(level, "medium")
        reasons.append("The model has low confidence in this result.")

    # 2) The model is torn between two classes
    if len(top) >= 2 and (top[0]["probability"] - top[1]["probability"]) < CLOSE_MARGIN:
        level = _raise_to(level, "medium")
        reasons.append(
            f"The model is unsure between {_pretty(top[0]['label'])} ({top[0]['probability']:.0%}) "
            f"and {_pretty(top[1]['label'])} ({top[1]['probability']:.0%})."
        )

    # 3) A more serious class got a noticeable probability even though it is not the top result
    for item in top:
        if item["label"] == label:
            continue
        if item["label"] in WATCH_CLASSES and item["probability"] >= RISK_PROB_FLOOR:
            level = _raise_to(level, "medium")
            reasons.append(
                f"The model also gave {item['probability']:.0%} to {_pretty(item['label'])}, "
                "which needs a doctor's assessment to rule out."
            )

    # 4) Red flags reported by the user
    s = symptoms or {}
    flags = [qid for qid in RED_FLAG_QUESTIONS if s.get(qid) == "yes"]
    if flags:
        names = [QUESTION_BANK[qid]["short"].lower() for qid in flags]
        reasons.append("You reported: " + "; ".join(names) + ".")
        if label in LESION_CLASSES:
            level = _raise_to(level, "high")
        else:
            level = _raise_to(level, "medium")
            if len(flags) >= 2:
                level = _raise_to(level, "high")

    return {
        "level": level,
        "title": LEVEL_INFO[level]["title"],
        "message": LEVEL_INFO[level]["message"],
        "watch_for_title": "See a doctor if:" if level == "low" else "Get medical help sooner if:",
        "watch_for": list(info["watch_for"]),
        "reasons": reasons,
        "top_predictions": top,
        "disclaimer": DISCLAIMER,
    }
