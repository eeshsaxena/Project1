# TruthfulRAG v5

**B.Tech Computer Science and Engineering — IIIT Manipur**

A conflict-aware Knowledge-Graph RAG system implementing 8 novel improvements over the TruthfulRAG v4 baseline (arXiv:2511.10375). The system builds a structured knowledge graph from any document collection, detects temporal contradictions between facts, resolves them using temporal decay scoring, and generates answers with calibrated confidence scores.

---

## What is new in v5

| Feature | Detail |
|---|---|
| [N1] Cross-document corroboration | Path score multiplied by log(1 + source_count) |
| [N2] Temporal decay on edges | Weight = e^(-0.08 * age_in_years) |
| [N3] Hybrid retrieval (RRF) | BM25 + semantic embedding fused by Reciprocal Rank Fusion |
| [N4] Temporal graph snapshots | Query anchored to a specific year |
| [N5] Entropy-based path filtering | Paths that increase LLM confusion (DeltaH > threshold) are discarded |
| [N6] Explanation chain | Full audit trail of removed facts included in answer |
| [N7] Calibrated confidence score | 0-100% per response: 0.4*entropy + 0.3*support + 0.3*recency |
| [N8] Claim verification | SUPPORTED / REFUTED / UNCERTAIN verdict for any declarative statement |
| [N9] Adaptive entropy sampling | Extreme-score paths skip LLM sampling, ~30% latency reduction |

---

## Requirements

- **Python 3.11+**
- **Neo4j** running on `bolt://localhost:7687`
- **Ollama** running on `http://localhost:11434` with a model loaded

```bash
pip install langchain langchain-neo4j langchain-ollama sentence-transformers rank-bm25 numpy python-dotenv
```

---

## Environment setup

Create a `.env` file in the project root:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
PIPELINE_LLM_MODEL=qwen2.5:7b-instruct
PIPELINE_EMBED_MODEL=all-MiniLM-L6-v2
```

---

## Running the pipeline

```bash
# Single corpus, automatic Q&A
python enhanced_main.py --corpus corpus_law.json

# Claim verification (N8)
python enhanced_main.py --corpus corpus_law.json --verify "The BNS replaced the IPC in 2024"

# Disable adaptive entropy sampling (N9)
python enhanced_main.py --corpus corpus_medical.json --no-adaptive

# Interactive mode (type documents and questions at the terminal)
python enhanced_main.py --interactive
```

---

## Web demo

```bash
# Install and start the API server
pip install flask flask-cors
python web_demo/server.py

# Then open in browser
web_demo/index.html           # Static showcase
web_demo/chatbot_live.html    # Live dual chatbot (requires server.py)
web_demo/chatbot.html         # Pre-loaded examples (no server needed)
```

---

## Available corpora

| File | Domain | Conflicts included |
|---|---|---|
| `corpus_politics.json` | Indian Politics | PM 2014 vs 2024, Twitter CEO |
| `corpus_india_science.json` | ISRO / Science | Chandrayaan missions |
| `corpus_law.json` | Indian Law | IPC 1860 vs BNS 2024 (164-year gap) |
| `corpus_space.json` | Space Science | Pluto reclassification, JWST vs Hubble |
| `corpus_medical.json` | Medical | Aspirin contraindication, diabetes guidelines |

---

## Project structure

```
enhanced_main.py          Main pipeline (EnhancedPipeline class)
corpus_*.json             Domain-specific test corpora
web_demo/
  server.py               Flask API server (two endpoints: /api/query/lc, /api/query/v5)
  index.html              Static research showcase
  chatbot_live.html       Live dual chatbot frontend
  chatbot.html            Pre-loaded comparison demo
Report_Format_.../        LaTeX B.Tech project report
```

---

## Architecture

```
Documents
    |
    +-- [A4+] Auto Schema Inference (LLM-driven, zero manual config)
    |
    +-- [A1-A3] Knowledge Graph Construction
    |           (head, relation, tail, year) triples -> Neo4j
    |
    +-- [N3] Hybrid Retrieval (BM25 + Semantic RRF)
    |         + Personalised PageRank graph walk
    |
    +-- [N1, N2] Path Scoring
    |            score = PPR * e^(-0.08*age) * log(1 + support)
    |
    +-- [N6] Conflict Detection
    |         same (relation, tail), different head, different year -> remove older
    |
    +-- [N5] Entropy Filter (DeltaH per surviving path)
    |
    +-- [N7] Generate Answer + Confidence Score
    |         confidence = 0.4*H + 0.3*support + 0.3*recency
    |
    +-- [N8] Claim Verifier (optional, --verify flag)
              SUPPORTED / REFUTED / UNCERTAIN with one-sentence reason
```

---

## Domains tested

The pipeline requires zero code changes between domains. The schema (entity types, relation types) is inferred from the corpus automatically.

| Domain | Confidence range | Notable conflict |
|---|---|---|
| Indian Politics | 57-64% | 10-year PM election gap |
| Indian Law | 71% | 164-year IPC-to-BNS gap |
| Space Science | 79% | 76-year Pluto classification gap |
| Medical | 83% | Safety-critical 1950s vs 1986 reversal |
| ISRO / Science | 42% | Relation-based disambiguation (no conflict in this case) |

---

## Reference

Grounding baseline: TruthfulRAG v4 — arXiv:2511.10375
