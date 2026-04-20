"""
TruthfulRAG v5 — Launcher Backend
Runs on port 5001. Starts Neo4j, Ollama, Flask server.
"""
import subprocess, time, os, sys, socket, json, threading
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PY   = r"C:\Users\eeshs\AppData\Local\Programs\Python\Python312\python.exe"
NEO4J = r"C:\Users\eeshs\.Neo4jDesktop2\Data\dbmss\dbms-39a1ec08-36c3-4fbf-ad59-2a19f420a0c5\bin\neo4j.bat"
OLLAMA = r"C:\Users\eeshs\AppData\Local\Programs\Ollama\ollama.exe"
PROJECT = r"D:\Project-1"

state = {
    "neo4j":  {"status": "idle", "msg": "Not started"},
    "ollama": {"status": "idle", "msg": "Not started"},
    "server": {"status": "idle", "msg": "Not started"},
}
launch_lock = threading.Lock()

def port_open(port):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=1)
        s.close(); return True
    except: return False

def start_services():
    global state

    # ── OLLAMA ────────────────────────────────────────────────
    if port_open(11434):
        state["ollama"] = {"status": "ok", "msg": "Already running on :11434"}
    else:
        state["ollama"] = {"status": "starting", "msg": "Starting Ollama…"}
        subprocess.Popen([OLLAMA, "serve"],
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
        for _ in range(20):
            time.sleep(1)
            if port_open(11434):
                state["ollama"] = {"status": "ok", "msg": "Running on :11434"}
                break
        else:
            state["ollama"] = {"status": "error", "msg": "Timeout — check Ollama manually"}

    # ── NEO4J ─────────────────────────────────────────────────
    if port_open(7687):
        state["neo4j"] = {"status": "ok", "msg": "Already running on :7687"}
    else:
        state["neo4j"] = {"status": "starting", "msg": "Starting Neo4j…"}
        subprocess.Popen([NEO4J, "console"],
                         cwd=os.path.dirname(NEO4J),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
        for _ in range(40):
            time.sleep(2)
            if port_open(7687):
                state["neo4j"] = {"status": "ok", "msg": "Running on :7687"}
                break
        else:
            state["neo4j"] = {"status": "error", "msg": "Timeout — start Neo4j Desktop manually"}

    # ── FLASK SERVER ──────────────────────────────────────────
    if port_open(5000):
        state["server"] = {"status": "ok", "msg": "Already running on :5000"}
    else:
        state["server"] = {"status": "starting", "msg": "Starting Flask server…"}
        subprocess.Popen(
            [PY, "web_demo\\server.py"],
            cwd=PROJECT,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        for _ in range(30):
            time.sleep(1)
            if port_open(5000):
                state["server"] = {"status": "ok", "msg": "Running on :5000"}
                break
        else:
            state["server"] = {"status": "error", "msg": "Timeout — check server.py manually"}

@app.route("/launch", methods=["POST"])
def launch():
    if not launch_lock.acquire(blocking=False):
        return jsonify({"ok": False, "msg": "Already launching"})
    def run():
        try: start_services()
        finally: launch_lock.release()
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/status")
def status():
    summary = {k: v for k, v in state.items()}
    all_ok = all(v["status"] == "ok" for v in state.values())
    summary["all_ok"] = all_ok
    return jsonify(summary)

@app.route("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("TruthfulRAG Launcher running on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
