import csv
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "votes.db"
CSV_PATH = BASE_DIR / "synthetic_hiring_bias_dataset.csv"

app = Flask(__name__, static_folder=str(BASE_DIR))
app.config["SECRET_KEY"] = "hirepasS-secret-2025"
socketio = SocketIO(app, cors_allowed_origins="*")

KEEP_COLS = {"id", "name", "education_level", "education_organization",
             "years_experience", "skills", "languages", "test_score"}


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                uid          TEXT PRIMARY KEY,
                applicant_id INTEGER NOT NULL,
                session_id   TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                hired        INTEGER NOT NULL CHECK(hired IN (0, 1))
            )
        """)
        con.commit()


def load_csv():
    applicants = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            applicants.append({k: v for k, v in row.items() if k in KEEP_COLS})
    return applicants


APPLICANTS = None


@app.before_request
def _ensure_data():
    global APPLICANTS
    if APPLICANTS is None:
        APPLICANTS = load_csv()


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "prototype_v2.html")


@app.route("/styles.css")
def styles():
    return send_from_directory(str(BASE_DIR), "styles.css")


@app.route("/api/applicants")
def api_applicants():
    return jsonify(APPLICANTS)


@socketio.on("connect")
def on_connect():
    # Client receives their socket ID automatically via socket.id in JS
    pass


@socketio.on("vote")
def on_vote(data):
    """
    Expected payload:
        { applicant_id: int, hired: 0|1 }
    """
    from flask import request as sio_request
    sid = sio_request.sid
    applicant_id = int(data.get("applicant_id", 0))
    hired = int(bool(data.get("hired", 0)))
    uid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO votes (uid, applicant_id, session_id, timestamp, hired) VALUES (?,?,?,?,?)",
            (uid, applicant_id, sid, ts, hired),
        )
        con.commit()

    emit("vote_ack", {"uid": uid, "applicant_id": applicant_id})


if __name__ == "__main__":
    init_db()
    print(f"DB: {DB_PATH}")
    print(f"CSV: {CSV_PATH}  ({len(load_csv())} applicants)")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
