import base64
import os

from flask import Flask, jsonify, render_template, request
from google import genai

app = Flask(__name__)

api_key = os.getenv("AI_INTEGRATIONS_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
base_url = os.getenv("AI_INTEGRATIONS_GEMINI_BASE_URL")

if api_key:
    http_options = (
        genai.types.HttpOptions(base_url=base_url, api_version="")
        if base_url
        else None
    )
    client = genai.Client(api_key=api_key, http_options=http_options)
else:
    client = None

MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

LENGTH_TOKENS = {
    "short": 400,
    "normal": 800,
    "detailed": 1600,
}


def generate_with_fallback(contents, max_tokens: int) -> str:
    if client is None:
        return "AI service is not configured. Please set GEMINI_API_KEY."

    config = genai.types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.7,
    )

    for model_name in MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            text = (response.text or "").strip() if response else ""
            if text:
                return text
        except Exception as exc:
            app.logger.warning("Model %s failed: %s", model_name, exc)
            continue

    return "⚠️ Incomplete response. Please try again."


def format_history(history: list) -> str:
    if not history:
        return ""
    lines = []
    for item in history[-2:]:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").lower()
        text = (item.get("text") or "").strip()
        if not text:
            continue
        label = "Student" if role == "user" else "Assistant"
        lines.append(f"{label}: {text}")
    if not lines:
        return ""
    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"


def build_prompt(message: str, mode: str, user_name: str, history: list) -> str:
    name = (user_name or "Student").strip() or "Student"
    history_block = format_history(history)

    if mode == "support":
        return (
            "You are a calm and supportive assistant.\n\n"
            f"The user's name is {name}. Use it occasionally to sound human. "
            "Do not repeat the name too often.\n\n"
            "Respond naturally based on what the user is feeling.\n\n"
            "- If the user message is short → respond briefly.\n"
            "- If the user is expressive → respond with more depth.\n\n"
            "Do not give robotic or overly long replies.\n"
            "Be human, warm, and balanced.\n"
            "If the answer can be given in one sentence, "
            "do not expand it unnecessarily.\n\n"
            f"{history_block}"
            f"Message:\n{message}"
        )
    return (
        "You are a smart study assistant.\n\n"
        f"The student's name is {name}. Use their name naturally only when "
        "appropriate. Do not overuse the name.\n\n"
        "Your goal is to match the depth of your answer to the user's question.\n\n"
        "Guidelines:\n"
        "- For simple questions → give clear, direct answers\n"
        "- For conceptual questions → explain with clarity\n"
        "- For deeper questions → provide structured explanations\n"
        "- If the user asks for more detail → expand\n\n"
        "Do NOT:\n"
        "- Give unnecessarily long answers\n"
        "- Give overly short answers that feel incomplete\n\n"
        "Focus on:\n"
        "- Clarity\n"
        "- Relevance\n"
        "- Student understanding\n\n"
        "If the answer can be given in one sentence, "
        "do not expand it unnecessarily.\n"
        "If the answer is long, complete it fully. "
        "Do not stop in the middle.\n\n"
        "Answer naturally, like a good teacher would.\n\n"
        f"{history_block}"
        f"Question:\n{message}"
    )


def build_contents(
    message: str, mode: str, images: list, user_name: str, history: list
) -> list:
    prompt = build_prompt(
        message or "(no text provided)", mode, user_name, history
    )
    parts = [{"text": prompt}]

    for img in images[:4]:
        if not isinstance(img, str) or "," not in img:
            continue
        try:
            header, b64 = img.split(",", 1)
            mime = "image/png"
            if header.startswith("data:") and ";" in header:
                mime = header[5:].split(";", 1)[0] or "image/png"
            data = base64.b64decode(b64)
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        except Exception as exc:
            app.logger.warning("Skipping invalid image: %s", exc)

    return parts


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    mode = data.get("mode", "study")
    images = data.get("images") or []
    length = (data.get("length") or "normal").lower()
    user_name = (data.get("userName") or "Student").strip() or "Student"
    history = data.get("history") or []

    if mode not in ("study", "support"):
        mode = "study"

    max_tokens = LENGTH_TOKENS.get(length, LENGTH_TOKENS["normal"])

    if not user_message and not images:
        return jsonify({"reply": "Please type a message or attach an image first."}), 400

    contents = build_contents(user_message, mode, images, user_name, history)
    reply = generate_with_fallback(contents, max_tokens)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
