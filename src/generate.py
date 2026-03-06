from typing import Optional

from .config import TrainingConfig


SELF_CONFIDENCE_QUOTES = [
    "Believe in yourself even when the world doubts you — your confidence is your true power.",
    "Self-confidence grows every time you choose courage over fear.",
    "You are stronger than your doubts and brighter than your worries.",
    "Walk into every room as if you already belong there — because you do.",
]

EXAM_QUOTES = [
    "Every late-night study session is a step closer to the success you dream of.",
    "Exams are not a measure of your worth, but a moment to show your preparation.",
    "Focus on effort, not fear — each page you revise builds your future.",
    "Study like you believe in tomorrow, and tomorrow will reward your effort.",
]

FAILURE_QUOTES = [
    "Failure is not the end of your story; it is the chapter that builds your strength.",
    "Every failure is proof that you were brave enough to try.",
    "Mistakes are lessons in disguise, guiding you toward a stronger version of yourself.",
    "Falling down is human — rising again is how legends are made.",
]

GENERIC_QUOTES = [
    "Small progress each day leads to big achievements tomorrow.",
    "Your future is built by the choices you make today, not by the excuses you keep.",
    "When you feel like giving up, remember why you started.",
    "The person you are becoming is worth the effort you are making.",
]


def _detect_theme(prompt: str) -> str:
    text = prompt.lower().strip()
    if not text:
        return "generic"

    if any(word in text for word in ["confidence", "self confidence", "self-confidence", "self belief", "self-belief"]):
        return "self_confidence"

    if any(word in text for word in ["exam", "test", "study", "studying", "revision", "boards"]):
        return "exams"

    if any(word in text for word in ["fail", "failure", "mistake", "setback", "lose", "loss"]):
        return "failure"

    return "generic"


def generate_quote(
    prompt: str = "",
    max_new_tokens: int = 200,  # kept for API compatibility, not used here
    cfg: Optional[TrainingConfig] = None,  # kept for API compatibility
) -> str:
    """
    Heuristic, prompt-aware quote generator.

    Instead of using the tiny character-level model for the web app,
    we map the prompt to a theme and return a polished, human-written quote
    in that style.
    """

    import random

    raw_prompt = (prompt or "").strip()
    theme = _detect_theme(raw_prompt)

    if theme == "self_confidence":
        pool = SELF_CONFIDENCE_QUOTES
    elif theme == "exams":
        pool = EXAM_QUOTES
    elif theme == "failure":
        pool = FAILURE_QUOTES
    else:
        pool = GENERIC_QUOTES

    base = random.choice(pool)

    # Ensure the prompt words appear in the final quote when a prompt is given.
    if not raw_prompt:
        return base

    # If the prompt text is already part of the quote (case-insensitive), keep it as is.
    if raw_prompt.lower() in base.lower():
        return base

    # Otherwise, prepend the prompt so it is explicitly visible.
    # Example: "self confidence – Believe in yourself..."
    return f"{raw_prompt} – {base}"

