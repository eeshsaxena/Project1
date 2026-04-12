"""
TruthfulRAG v5 — Live API Server
Usage:
  pip install flask flask-cors
  python web_demo/server.py

Then open  web_demo/chatbot_live.html  in your browser.
Requires: Neo4j running on bolt://localhost:7687
          Ollama running on http://localhost:11434
"""
from __future__ import annotations
import sys, os, json, time, threading, traceback

# Add parent so enhanced_main is importable
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT)

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(os.path.join(PARENT, ".env"))

app = Flask(__name__)
CORS(app)

# ── App State ────────────────────────────────────────────────────────────────
class State:
    lc   = None          # SimpleLCRetriever
    v5   = None          # EnhancedPipeline
    status    = "idle"   # idle | ingesting | lc_only | ready | error
    status_msg = "No corpus loaded yet."
    corpus_name = None
    docs: list = []
    queries: list = []

S = State()

# ── Connectivity checks ──────────────────────────────────────────────────────
def _tcp(host, port, timeout=2):
    import socket
    try:
        with socket.create_connection((host, port), timeout): return True
    except: return False

def check_ollama(): return _tcp("localhost", 11434)
def check_neo4j():  return _tcp("localhost", 7687)

# ── Lazy LLM + Embedder ──────────────────────────────────────────────────────
_llm = _embedder = None

def get_llm():
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        model = os.getenv("PIPELINE_LLM_MODEL", "qwen2.5:7b-instruct")
        _llm = ChatOllama(model=model, temperature=0.0)
        print(f"  [LLM] {model} ready")
    return _llm

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [Embedder] all-MiniLM-L6-v2 ready")
    return _embedder

# ── Simple LangChain-style retriever (no FAISS needed) ──────────────────────
import numpy as np

def _cos(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (n + 1e-9))

class SimpleLCRetriever:
    """
    Mimics standard LangChain RAG:
      load → chunk (300-char windows) → embed → cosine similarity → top-K → generate
    No conflict detection, no temporal reasoning, no confidence score.
    """
    def __init__(self, docs: list[str], llm, embedder):
        self.llm = llm
        self.embedder = embedder
        # Chunk: sliding window over words
        self.chunks: list[str] = []
        for doc in docs:
            words = doc.split()
            if not words: continue
            step, overlap = 60, 10          # ~300 chars, ~50-char overlap
            for i in range(0, len(words), step - overlap):
                chunk = " ".join(words[i: i + step])
                if len(chunk.strip()) > 20:
                    self.chunks.append(chunk)
        if not self.chunks:
            self.chunks = list(docs)       # fallback: one chunk per doc
        print(f"  [LC] {len(self.chunks)} chunks — embedding...")
        self.embs = embedder.encode(self.chunks).tolist()
        print(f"  [LC] Retriever ready")

    def query(self, q: str, top_k: int = 4) -> dict:
        t0 = time.time()
        q_emb = self.embedder.encode([q])[0].tolist()
        scored = sorted(
            [(i, _cos(q_emb, e)) for i, e in enumerate(self.embs)],
            key=lambda x: -x[1]
        )
        top = scored[:top_k]
        chunks_used = [self.chunks[i] for i, _ in top]
        scores_used = [round(s, 4) for _, s in top]
        context = "\n---\n".join(chunks_used)
        prompt = (
            "Answer the following question using only the provided context. "
            "Be concise and direct.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n\nAnswer:"
        )
        try:
            answer = self.llm.invoke(prompt).content.strip()
        except Exception as e:
            answer = f"[LLM error: {e}]"
        return {
            "answer": answer,
            "chunks": len(chunks_used),
            "top_scores": scores_used,
            "conflict_detected": False,
            "confidence": None,
            "contradictions_removed": 0,
            "elapsed": round(time.time() - t0, 2),
        }

# ── Background ingest ────────────────────────────────────────────────────────
def _do_ingest(corpus_path: str):
    S.status = "ingesting"
    S.v5 = None; S.lc = None
    try:
        from enhanced_main import _load_corpus, EnhancedPipeline, CFG
        S.status_msg = "Loading corpus…"
        data = _load_corpus(corpus_path)
        S.docs    = data.get("docs",    [])
        S.queries = data.get("queries", [])
        S.corpus_name = os.path.basename(corpus_path)

        # ── LC retriever (fast, no Neo4j) ──
        S.status_msg = f"Building LC retriever ({len(S.docs)} docs)…"
        llm = get_llm()
        emb = get_embedder()
        S.lc = SimpleLCRetriever(S.docs, llm, emb)
        S.status_msg = "LC ready. Building v5 knowledge graph (this takes a minute)…"

        # ── v5 pipeline (requires Neo4j) ──
        try:
            pipe = EnhancedPipeline(CFG)
            pipe.ingest(S.docs)
            S.v5 = pipe
            S.status = "ready"
            S.status_msg = f"Both pipelines ready — {len(S.docs)} docs | {len(S.queries)} example queries"
            print("  [Server] Both pipelines ready")
        except Exception as e:
            S.status = "lc_only"
            S.status_msg = (f"LC ready. v5 unavailable — is Neo4j running? ({str(e)[:100]})")
            print(f"  [Server] v5 init failed: {e}")

    except Exception as e:
        S.status = "error"
        S.status_msg = f"Ingest error: {e}"
        traceback.print_exc()

# ── Helper: serialise a KnowledgePath ───────────────────────────────────────
def _kp(kp) -> dict:
    return {
        "head":     kp.head,
        "relation": kp.relation,
        "tail":     kp.tail,
        "context":  kp.context,
        "score":    round(float(kp.score or 0), 4),
        "support":  kp.support,
        "max_year": kp.max_year,
        "delta_h":  round(float(kp.delta_h), 4) if kp.delta_h is not None else None,
        "corrective": bool(getattr(kp, "is_corrective", False)),
    }

# ── API ──────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({
        "status":     S.status,
        "message":    S.status_msg,
        "corpus":     S.corpus_name,
        "doc_count":  len(S.docs),
        "query_count": len(S.queries),
        "sample_queries": S.queries[:5],
        "ollama":     check_ollama(),
        "neo4j":      check_neo4j(),
        "lc_ready":   S.lc  is not None,
        "v5_ready":   S.v5  is not None,
    })

@app.route("/api/corpora")
def api_corpora():
    result = []
    for f in sorted(os.listdir(PARENT)):
        if f.endswith(".json") and "corpus" in f.lower():
            p = os.path.join(PARENT, f)
            try:
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
                result.append({
                    "name":    f,
                    "path":    p,
                    "docs":    len(d.get("docs",    [])),
                    "queries": len(d.get("queries", [])),
                })
            except: pass
    return jsonify(result)

@app.route("/api/load", methods=["POST"])
def api_load():
    if S.status == "ingesting":
        return jsonify({"ok": False, "message": "Already ingesting — please wait."})
    data = request.json or {}
    path = data.get("corpus_path", "")
    if not path or not os.path.isfile(path):
        return jsonify({"ok": False, "message": f"File not found: {path}"})
    threading.Thread(target=_do_ingest, args=(path,), daemon=True).start()
    return jsonify({"ok": True, "message": "Ingestion started…"})

@app.route("/api/query/lc", methods=["POST"])
def api_query_lc():
    if S.lc is None:
        return jsonify({"error": "LC retriever not loaded. Load a corpus first."}), 503
    q = (request.json or {}).get("question", "").strip()
    if not q: return jsonify({"error": "Empty question"}), 400
    try:
        return jsonify(S.lc.query(q))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/query/v5", methods=["POST"])
def api_query_v5():
    if S.v5 is None:
        return jsonify({"error": "TruthfulRAG v5 not ready. Is Neo4j running?"}), 503
    q = (request.json or {}).get("question", "").strip()
    if not q: return jsonify({"error": "Empty question"}), 400
    try:
        r    = S.v5.query(q)
        meta = r.get("meta", {})
        conf = meta.get("confidence", 0)
        # normalise 0-1 float to 0-100 int
        if isinstance(conf, float) and conf <= 1.0: conf = round(conf * 100)
        return jsonify({
            "answer":                r.get("answer", ""),
            "paths":                 [_kp(p) for p in r.get("paths", [])],
            "total_paths":           meta.get("total_paths", 0),
            "selected_paths":        meta.get("selected_paths", 0),
            "contradictions_removed":meta.get("contradictions_removed", 0),
            "intent":                meta.get("intent", ""),
            "strategy":              meta.get("strategy", ""),
            "entity_types":          meta.get("entity_types", []),
            "confidence":            int(conf),
            "elapsed":               round(r.get("elapsed", 0), 2),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/verify", methods=["POST"])
def api_verify():
    if S.v5 is None:
        return jsonify({"error": "TruthfulRAG v5 not ready."}), 503
    claim = (request.json or {}).get("claim", "").strip()
    if not claim: return jsonify({"error": "Empty claim"}), 400
    try:
        return jsonify(S.v5.verify(claim))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TruthfulRAG v5 — Live API Server")
    print(f"  API : http://localhost:5000")
    print(f"  Open: web_demo/chatbot_live.html")
    print(f"  Ollama reachable: {check_ollama()}")
    print(f"  Neo4j  reachable: {check_neo4j()}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
