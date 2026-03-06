from flask import Flask, jsonify, render_template, request

from src.zerochat import ZeroChatSession

# Single in-memory chat session (sufficient for demo / local use)
session = ZeroChatSession()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "").strip()

        try:
            quote = session.reply(prompt)

            # Build prompt+quote history from chat-style history
            history_pairs = []
            msgs = session.history
            for i in range(0, len(msgs), 2):
                if i + 1 >= len(msgs):
                    break
                user_msg = msgs[i]
                assistant_msg = msgs[i + 1]
                if user_msg.role != "user" or assistant_msg.role != "assistant":
                    continue
                history_pairs.append(
                    {"prompt": user_msg.content, "quote": assistant_msg.content}
                )

            return jsonify({"ok": True, "quote": quote, "history": history_pairs})
        except Exception:  # noqa: BLE001
            return jsonify({"ok": False, "error": "Internal error"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    # Runs on http://127.0.0.1:5000
    app.run(debug=True)

