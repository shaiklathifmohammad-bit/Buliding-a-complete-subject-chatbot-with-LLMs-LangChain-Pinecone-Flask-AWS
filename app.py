# app.py
from flask import Flask, render_template, request, jsonify
from src.rag_chat import get_rag_chain, ask_question_hybrid

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Build once
rag_chain = get_rag_chain()

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"answer": "Please type something."})

    try:
        answer = ask_question_hybrid(message, rag_chain)
        return jsonify({"answer": answer})
    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"answer": "⚠️ Internal error. Try again."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)