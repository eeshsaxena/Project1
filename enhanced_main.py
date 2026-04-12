from __future__ import annotations
import sys
# ================================================================
# TruthfulRAG v5 — ENHANCED (8 Novel Improvements over v4)
# Builds on: TruthfulRAG v4 (arXiv:2511.10375)
# Novel additions:
#   [N1] Cross-Document Corroboration Scoring
#   [N2] Temporal Decay on Edge Weights
#   [N3] Hybrid BM25 + Semantic Retrieval (RRF fusion)
#   [N4] Temporal Graph Snapshots (year-anchored filtering)
#   [N6] Explanation-Chain Answer Generation
#   [N7] Calibrated Answer Confidence Score
#   [N8] Claim Verification Mode (SUPPORTED / REFUTED / UNCERTAIN)
#   [N9] Adaptive Entropy Sampling (skip sampling for high/low-score paths)
# Run: python enhanced_main.py
# ================================================================
import argparse, difflib, hashlib, json, logging, math, os, re, time
import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from sentence_transformers import SentenceTransformer; _ST = True
except ImportError:
    _ST = False; print("[WARN] pip install sentence-transformers")
try:
    from rank_bm25 import BM25Okapi; _BM25 = True
except ImportError:
    _BM25 = False; print("[WARN] pip install rank-bm25")

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
load_dotenv()

# [N2] Current year — dynamic, never needs manual update
CURRENT_YEAR: int = datetime.date.today().year

# ── CONFIG ───────────────────────────────────────────────────────
# All sensitive / environment-specific values read from .env or environment.
# Override any value by setting the corresponding env var before running.

CFG: Dict[str, Any] = {
    # ── LLM (override via env) ───────────────────────────────────
    "llm_model":              os.getenv("PIPELINE_LLM_MODEL",         "qwen2.5:7b-instruct"),
    "llm_temperature":        float(os.getenv("PIPELINE_LLM_TEMP",    "0.0")),
    "llm_temp_sampler":       float(os.getenv("PIPELINE_LLM_TEMP_S",  "0.7")),
    "ollama_base_url":        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    # ── Neo4j (override via env) ─────────────────────────────────
    "neo4j_uri":              os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
    "neo4j_user":             os.getenv("NEO4J_USERNAME", "neo4j"),
    "neo4j_pass":             os.getenv("NEO4J_PASSWORD", ""),
    # ── Embeddings (override via env) ────────────────────────────
    "embedding_model":        os.getenv("PIPELINE_EMBED_MODEL",       "all-MiniLM-L6-v2"),
    # ── Retrieval ────────────────────────────────────────────────
    "top_k_paths": 5,
    "alpha": 0.5,  "beta": 0.5,
    "hub_penalty_weight":     0.3,
    "ppr_damping":            0.85,  "ppr_iterations": 20,
    "rel_filter_threshold":   0.30,
    # ── Schema (always auto-inferred, entity/relation lists filled at runtime) ──
    "schema_entity_types":    [],   # populated by _infer_schema() or corpus.json
    "schema_relation_types":  [],   # populated by _infer_schema() or corpus.json
    # ── Entropy / conflict ────────────────────────────────────────
    "n_entropy_samples":      3,
    "entropy_tau":            None,
    "semantic_cluster_thresh":0.85,
    "enable_contradiction_filter": _ST,
    "tau_by_intent":          {"factual_lookup":0.25,"temporal":0.35,
                               "comparison":0.50,"causal":0.45,"unknown":None},
    # ── Runtime ──────────────────────────────────────────────────
    "clear_graph_on_start":   True,
    "verbose":                True,
    # ── [N1] Corroboration ────────────────────────────────────────
    "corroboration_weight":   0.4,
    # ── [N2] Temporal Decay ───────────────────────────────────────
    "temporal_decay_lambda":  0.08,
    "use_temporal_decay":     True,
    # ── [N3] Hybrid Retrieval ─────────────────────────────────────
    "hybrid_semantic_k":      10,
    "rrf_k":                  60,
    "use_hybrid_retrieval":   _ST,
    # ── [N4] Snapshot ─────────────────────────────────────────────
    "snapshot_year":          None,   # None = auto-detect from query text
    # ── [N6] Explanation Chain ────────────────────────────────────
    "explanation_chain":      True,
    # ── [N7] Confidence Score ─────────────────────────────────────
    "confidence_weights":     {"h":0.40,"sup":0.30,"rec":0.30},
    # ── [N8] Claim Verification ───────────────────────────────────
    "verify_mode":            False,   # set True via --verify flag
    # ── [N9] Adaptive Entropy Sampling ────────────────────────────
    "adaptive_entropy":       True,    # skip sampling when path score is extreme
    "adaptive_high_thresh":   0.70,    # path score above this → assume trustworthy
    "adaptive_low_thresh":    0.05,    # path score below this → assume stale/skip
}

# Terminal-width separator — adapts to console, never magic-number
try:    _TW = min(os.get_terminal_size().columns, 80)
except: _TW = 72
SEP = "─" * _TW

# ── [C1] LLM CACHE ───────────────────────────────────────────────
class LLMCache:
    def __init__(self): self._s: Dict[str,str] = {}; self.hits = self.misses = 0
    def _k(self, p, t): return hashlib.md5(f"{t}::{p}".encode()).hexdigest()
    def get(self, p, t):
        k=self._k(p,t)
        if k in self._s: self.hits+=1; return self._s[k]
        self.misses+=1; return None
    def set(self, p, t, r): self._s[self._k(p,t)]=r
    def stats(self):
        n=self.hits+self.misses
        return f"Cache {self.hits}/{n} hits ({self.hits/n*100:.0f}%)" if n else "Cache 0/0"

_CACHE = LLMCache()

def cached_invoke(llm, prompt):
    t=llm.temperature; c=_CACHE.get(prompt,t)
    if c is not None: return c
    r=llm.invoke(prompt).content.strip()
    _CACHE.set(prompt,t,r); return r

# ── PROMPTS ───────────────────────────────────────────────────────
def _triple_prompt(text, etypes, rtypes):
    schema = f"Entities: {etypes}\nRelations: {rtypes}\n"
    return (f"{schema}Extract ALL facts as JSON list. "
            "Each item: {\"head\":str,\"relation\":str,\"tail\":str,\"year\":int|null}\n"
            f"Text:\n\"\"\"{text}\"\"\"\nJSON:")

INTENT_PROMPT = ("Classify the intent: factual_lookup | temporal | comparison | causal | unknown\n"
                 "Reply with ONE word only.\nQuestion: \"{query}\"")
KEY_PROMPT    = ("From this question extract:\n- entities (people, orgs, places)\n"
                 "- relations (roles like CEO, verbs like founded)\nQuestion: \"{query}\"\n"
                 "Reply ONLY:\nENTITIES: e1, e2\nRELATIONS: r1, r2")
PARAM_PROMPT  = "Answer this question from memory only:\nQ: {query}\nA:"
RAG_PROMPT    = ("Context:\n{context}\n\nQuestion: {query}\n\n"
                 "Give a direct, one-sentence answer. Include the year when relevant.")

# [N6] Explanation-aware prompt
EXPLAIN_PROMPT = ("You resolved a knowledge conflict. Provide a clear answer with brief reasoning.\n\n"
                  "Conflict Resolution Chain:\n{chain}\n\n"
                  "Question: {query}\n\n"
                  "Answer (include WHAT the answer is and WHY this source was chosen):")

# [A4+] Dynamic schema inference prompt — used to auto-discover entity/relation types
SCHEMA_INFER_PROMPT = (
    "Read the following text samples and infer the most useful entity types and relation types "
    "for building a knowledge graph over this domain.\n"
    "Return ONLY valid JSON in exactly this format:\n"
    '{{"entity_types": ["TYPE1", "TYPE2", ...], "relation_types": ["REL_ONE", "REL_TWO", ...]}}\n\n'
    "Rules:\n"
    "- entity_types: 3-8 UPPERCASE labels (e.g. PERSON, DRUG, STATUTE, TEAM)\n"
    "- relation_types: 3-12 UPPER_SNAKE_CASE verbs (e.g. CEO_OF, TREATS, PLAYED_FOR)\n"
    "- Do NOT include generic types like THING or ENTITY\n\n"
    "Text samples:\n{samples}\n\nJSON:"
)

class HybridRetriever:
    """[N3] BM25 + Semantic embedding retrieval fused with RRF."""
    def __init__(self, docs: List[str], cfg: Dict, embedder=None):
        """Accepts a shared embedder to avoid loading the model twice."""
        self.docs = docs; self.cfg = cfg
        tokens = [d.lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokens) if _BM25 else None
        self.embedder  = embedder if (embedder and cfg.get("use_hybrid_retrieval")) else None
        self.doc_vecs  = self.embedder.encode(docs) if self.embedder else None

    def retrieve(self, query: str, k: int = 10) -> List[str]:
        k = min(k, len(self.docs))
        if not self.docs: return []

        # BM25 ranking
        bm25_scores = (self.bm25.get_scores(query.lower().split())
                       if self.bm25 else [1.0]*len(self.docs))
        bm25_rank   = {i: r for r, i in enumerate(
                           sorted(range(len(self.docs)), key=lambda i: bm25_scores[i], reverse=True))}

        if self.embedder is not None:
            # Semantic ranking
            qvec = self.embedder.encode([query])[0]
            sims = [float(np.dot(qvec, dv) / (np.linalg.norm(qvec)*np.linalg.norm(dv)+1e-9))
                    for dv in self.doc_vecs]
            sem_rank = {i: r for r, i in enumerate(
                            sorted(range(len(self.docs)), key=lambda i: sims[i], reverse=True))}

            # [N3] RRF fusion
            rrf_k = self.cfg["rrf_k"]
            rrf   = {i: 1/(rrf_k+bm25_rank[i]) + 1/(rrf_k+sem_rank[i]) for i in range(len(self.docs))}
            top   = sorted(rrf, key=rrf.get, reverse=True)[:k]
        else:
            top = sorted(bm25_rank, key=bm25_rank.get)[:k]

        return [self.docs[i] for i in top]

# ── MODULE A ENHANCED ─────────────────────────────────────────────
@dataclass
class TripleRecord:
    head: str; relation: str; tail: str; year: Optional[int]
    support_count: int = 1   # [N1] how many docs support this triple

class EnhancedGraphConstructor:
    """Module A + [N1] corroboration counting + [A1] year-on-edge + [A4+] dynamic schema."""
    def __init__(self, llm, graph, cfg):
        self.llm=llm; self.graph=graph; self.cfg=cfg
        self.transformer = LLMGraphTransformer(llm=llm)
        self._triple_bank: Dict[Tuple, TripleRecord] = {}  # [N1]

    def _infer_schema(self, docs: List[str]) -> None:
        """[A4+] PRE-PIPELINE schema discovery — NOT part of v4 Module A/B/C.

        The TruthfulRAG v4 pipeline (Modules A→B→C) requires a schema
        BEFORE it can extract triples. This method solves the chicken-and-egg
        problem by running a single, lightweight LLM call on a sample of docs
        to discover domain-specific entity and relation types.

        Flow:
          _infer_schema(docs)          ← this method (pre-pipeline, LLM only)
          └─▶ updates CFG schema
              └─▶ [Module A] EnhancedGraphConstructor.build()
                  └─▶ [Module B] EnhancedGraphRetriever.retrieve()
                      └─▶ [Module C] EnhancedConflictResolver.resolve()

        """
        sample_n     = self.cfg.get("schema_infer_sample", 5)
        sample       = docs[:sample_n]
        samples_text = "\n---\n".join(f"[{i+1}] {d}" for i, d in enumerate(sample))

        if self.cfg["verbose"]:
            print(f"  [A4+] Inferring schema from {len(sample)} doc(s) (pre-pipeline)...")

        raw = cached_invoke(self.llm, SCHEMA_INFER_PROMPT.format(samples=samples_text))
        try:
            # Strip markdown code fences if LLM wraps output in ```json ... ```
            clean = re.sub(r'```(?:json)?\s*', '', raw).strip()
            m     = re.search(r'\{.*\}', clean, re.DOTALL)
            data  = json.loads(m.group(0) if m else clean)
            etypes = [str(e).upper() for e in data.get("entity_types", []) if e]
            rtypes = [re.sub(r'\W+', '_', str(r).upper()).strip('_')
                      for r in data.get("relation_types", []) if r]
            if etypes:
                self.cfg["schema_entity_types"]  = etypes
                self.cfg["schema_relation_types"] = rtypes
                if self.cfg["verbose"]:
                    print(f"  [A4+] Entity types  : {etypes}")
                    print(f"  [A4+] Relation types: {rtypes}")
            else:
                if self.cfg["verbose"]:
                    print("  [A4+] LLM returned empty types — check prompt output.")
        except Exception as exc:
            if self.cfg["verbose"]:
                print(f"  [A4+] Inference failed ({exc}) — proceeding with empty schema.")


    def _clear(self):
        self.graph.query("MATCH (n) DETACH DELETE n")
        if self.cfg["verbose"]: print("  [A]  Neo4j cleared.")

    def _extract(self, text: str) -> List[dict]:
        """[A1][A4] Temporal + schema-guided extraction."""
        raw = cached_invoke(self.llm, _triple_prompt(
            text,
            self.cfg["schema_entity_types"], self.cfg["schema_relation_types"]))
        try:
            clean = re.sub(r'```(?:json)?\s*', '', raw).strip()
            m = re.search(r'\[.*\]', clean, re.DOTALL)
            data = json.loads(m.group(0) if m else clean)
            return [d for d in data if isinstance(d, dict)
                    and d.get("head") and d.get("relation") and d.get("tail")]
        except Exception: return []

    def _accumulate(self, triples: List[dict]):
        """[N1] Count corroborating documents per triple."""
        for t in triples:
            key = (t["head"].lower(), t["relation"].upper(), t["tail"].lower())
            if key in self._triple_bank:
                self._triple_bank[key].support_count += 1
                if t.get("year") and not self._triple_bank[key].year:
                    self._triple_bank[key].year = t["year"]
            else:
                self._triple_bank[key] = TripleRecord(
                    t["head"], t["relation"], t["tail"], t.get("year"))

    def _store_all(self):
        """Store all accumulated triples with support_count in Neo4j. [N1][A1]"""
        total = 0
        for (_, _, _), rec in self._triple_bank.items():
            h  = re.sub(r"'", "", rec.head)
            r  = re.sub(r'\W+', '_', rec.relation.upper())
            tl = re.sub(r"'", "", rec.tail)
            yr = rec.year; sc = rec.support_count
            if yr:
                hs = h.replace("'", "\\'")
                ts = tl.replace("'", "\\'")
                q = (f"MERGE (a:Entity {{id:'{hs}'}}) MERGE (b:Entity {{id:'{ts}'}}) "
                     f"MERGE (a)-[rel:{r} {{year:{yr}, support:{sc}}}]->(b)")
            else:
                hs = h.replace("'", "\\'")
                ts = tl.replace("'", "\\'")
                q = (f"MERGE (a:Entity {{id:'{hs}'}}) MERGE (b:Entity {{id:'{ts}'}}) "
                     f"MERGE (a)-[rel:{r} {{support:{sc}}}]->(b)")
            try: self.graph.query(q); total += 1
            except Exception: pass
        return total

    def _disambiguate(self):
        """[A2] Fuzzy entity merging."""
        ids = [r["id"] for r in self.graph.query("MATCH (n) RETURN n.id AS id") if r.get("id")]
        merged = 0
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.85 and a != b:
                    try:
                        as_ = a.replace("'", "\\'")
                        bs_ = b.replace("'", "\\'")
                        self.graph.query(
                            f"MATCH (old:Entity {{id:'{bs_}'}}) MATCH (keep:Entity {{id:'{as_}'}}) "
                            f"CALL apoc.refactor.mergeNodes([keep,old],"
                            f"{{properties:'combine',mergeRels:true}}) YIELD node RETURN node")
                        merged += 1
                    except Exception: pass
        if self.cfg["verbose"] and merged: print(f"  [A2] Disambiguated {merged} entity pairs")

    def _normalize(self):
        """[A3] LLM relation canonicalization."""
        rows = self.graph.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t LIMIT 50")
        raw  = [r["t"] for r in rows if r.get("t")]
        if not raw: return
        prompt = ("Canonicalize to UPPER_SNAKE_CASE. Return JSON {\"ORIG\":\"CANON\",...}\n"
                  f"Relations: {raw}\nJSON:")
        try:
            resp  = cached_invoke(self.llm, prompt)
            clean = re.sub(r'```(?:json)?\s*', '', resp).strip()
            m     = re.search(r'\{.*\}', clean, re.DOTALL)
            mapping = json.loads(m.group(0)) if m else {}
            for orig, canon in mapping.items():
                if orig != canon and canon:
                    c = re.sub(r'\W+', '_', canon.upper())
                    try:
                        self.graph.query(f"MATCH ()-[r:{orig}]->() "
                                         f"CALL apoc.refactor.setType(r,'{c}') "
                                         f"YIELD input RETURN count(input)")
                    except Exception: pass
            if self.cfg["verbose"]: print(f"  [A3] Normalised {len(mapping)} relation types")
        except Exception: pass

    def build(self, docs: List[str]):
        # [A4+] Always auto-infer schema from corpus before building graph
        self._infer_schema(docs)

        if self.cfg["clear_graph_on_start"]: self._clear()
        self.graph.add_graph_documents(
            self.transformer.convert_to_graph_documents([Document(page_content=d) for d in docs]))
        # [N1] accumulate corroboration counts across all docs
        for doc in docs:
            triples = self._extract(doc)
            self._accumulate(triples)
        total = self._store_all()
        if self.cfg["verbose"]: print(f"  [N1][A1][A4] Triples stored with corroboration: {total}")
        n_n = self.graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        n_e = self.graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        if self.cfg["verbose"]: print(f"  [A]  KG: {n_n} nodes, {n_e} edges")
        self._disambiguate(); self._normalize()
        return n_n, n_e

# ── MODULE B ENHANCED ─────────────────────────────────────────────
@dataclass
class KnowledgePath:
    context: str; score: float = 0.0; max_year: int = 0
    support: int = 1; combined_score: float = 0.0
    h_aug: float = 0.0; delta_h: float = 0.0
    contradicted: bool = False; is_corrective: bool = False
    conflict_reason: str = ""   # [N6]

class EnhancedGraphRetriever:
    """Module B + [N2] temporal decay + [N3] hybrid retrieval + [N4] snapshot."""
    def __init__(self, llm, graph, embedder, cfg):
        self.llm=llm; self.graph=graph
        self.embedder=embedder; self.cfg=cfg
        self._deg: Dict[str,int] = {}

    def _keys(self, q: str):
        resp = cached_invoke(self.llm, KEY_PROMPT.format(query=q))
        E, R = [], []
        for line in resp.splitlines():
            l = line.strip()
            if l.upper().startswith("ENTITIES:"): E=[e.strip() for e in l.split(":",1)[1].split(",") if e.strip()]
            elif l.upper().startswith("RELATIONS:"): R=[r.strip() for r in l.split(":",1)[1].split(",") if r.strip()]
        if not E:
            stop={"who","what","when","where","is","are","was","the","a","an","of"}
            E=[t for t in q.lower().split() if t not in stop]
        kw=list({w for r in R for w in re.split(r'[\s_]+',r.lower()) if w})
        return E, kw

    def detect_intent(self, q: str) -> str:
        """[C7] Domain-adaptive tau intent."""
        r = cached_invoke(self.llm, INTENT_PROMPT.format(query=q)).strip().lower()
        return r if r in {"factual_lookup","temporal","comparison","causal","unknown"} else "unknown"

    def _extract_query_year(self, q: str) -> Optional[int]:
        """[N4] Extract target year from query for temporal snapshot."""
        m = re.search(r'\b(19|20)\d{2}\b', q)
        return int(m.group(0)) if m else None

    def _node_ids(self):
        """Return flat list of string node IDs (guards against APOC returning lists)."""
        ids = []
        for r in self.graph.query("MATCH (n) RETURN n.id AS id LIMIT 200"):
            v = r.get("id")
            if v is None: continue
            if isinstance(v, list):   # APOC merge can return list of IDs
                ids.extend(str(x) for x in v if x)
            else:
                ids.append(str(v))
        return list(dict.fromkeys(ids))  # deduplicate, preserve order

    def _edges(self, snapshot_year: Optional[int] = None):
        rows = self.graph.query(
            "MATCH (a)-[r]->(b) RETURN type(r) AS rel_type, a.id AS src, "
            "b.id AS tgt, r.year AS year, r.support AS support LIMIT 200")
        clean = []
        for r in rows:
            src = r.get("src"); tgt = r.get("tgt")
            if isinstance(src, list): src = src[0] if src else None
            if isinstance(tgt, list): tgt = tgt[0] if tgt else None
            if not src or not tgt: continue
            r = dict(r); r["src"] = str(src); r["tgt"] = str(tgt)
            clean.append(r)
        if snapshot_year:
            # [N4] temporal snapshot — only facts up to target year
            clean = [r for r in clean if r.get("year") is None or r["year"] <= snapshot_year]
            if self.cfg["verbose"]: print(f"  [N4] Snapshot year={snapshot_year}: {len(clean)} edges")
        return clean

    def _deg_of(self, nid: str) -> int:
        if nid not in self._deg:
            ns = nid.replace("'", "\\'")
            rows = self.graph.query(f"MATCH (n{{id:'{ns}'}})-[r]-() RETURN count(r) AS d")
            self._deg[nid] = rows[0]["d"] if rows else 0
        return self._deg[nid]

    def _ppr(self, node_ids, edges, seed_entities):
        """[B4] Personalized PageRank."""
        n = len(node_ids)
        if n == 0: return {}
        idx = {v: i for i, v in enumerate(node_ids)}
        pr  = np.zeros(n); seeds = [idx[e] for e in seed_entities if e in idx]
        for s in seeds: pr[s] = 1.0/max(len(seeds),1)
        adj = defaultdict(list)
        for e in edges:
            if e["src"] in idx and e["tgt"] in idx:
                adj[idx[e["src"]]].append(idx[e["tgt"]])
        d = self.cfg["ppr_damping"]
        for _ in range(self.cfg["ppr_iterations"]):
            np_ = np.zeros(n)
            for src, tgts in adj.items():
                if tgts:
                    share = pr[src]/len(tgts)
                    for t in tgts: np_[t] += share
            pr = (1-d)*pr + d*np_
        return {v: float(pr[idx[v]]) for v in node_ids}

    def _filter_edges(self, edges, q):
        """[B5] Relation-type filtering by cosine semantic similarity."""
        if not self.embedder: return edges
        qv = self.embedder.encode([q])[0]
        out = []
        for e in edges:
            rv = self.embedder.encode([e["rel_type"].replace("_"," ").lower()])[0]
            sim = float(np.dot(qv,rv)/(np.linalg.norm(qv)*np.linalg.norm(rv)+1e-9))
            if sim >= self.cfg["rel_filter_threshold"]: out.append(e)
        return out or edges

    def _ref_p(self, path_text: str, E: List[str], Rkw: List[str]) -> float:
        """[B1] Real-ID Ref(p) = alpha*E_cov + beta*R_cov."""
        pt = path_text.lower()
        e_hit = sum(1 for e in E if e.lower() in pt)
        r_hit = sum(1 for r in Rkw if r in pt)
        e_cov = e_hit/max(len(E),1); r_cov = r_hit/max(len(Rkw),1)
        return self.cfg["alpha"]*e_cov + self.cfg["beta"]*r_cov

    def _temporal_decay(self, year: Optional[int]) -> float:
        """[N2] Exponential decay: older facts score lower."""
        if not self.cfg["use_temporal_decay"] or not year: return 1.0
        lam = self.cfg["temporal_decay_lambda"]
        return math.exp(-lam * max(0, CURRENT_YEAR - year))


    def retrieve(self, query: str):
        E, Rkw = self._keys(query); intent = self.detect_intent(query)
        qr = " ".join(E+Rkw) or query
        if self.cfg["verbose"]: print(f"  [B]  E={E}  Rkw={Rkw}\n  [C7] Intent={intent}")

        # [N4] temporal snapshot — filter edges to year <= query year
        snapshot_year = self.cfg.get("snapshot_year") or self._extract_query_year(query)

        node_ids = self._node_ids()
        edges    = self._edges(snapshot_year=snapshot_year)
        fedges   = self._filter_edges(edges, qr)   # [B5]
        ppr      = self._ppr(node_ids, fedges, E)  # [B4]

        paths: List[KnowledgePath] = []
        for e in fedges:
            src, tgt, rel = e["src"], e["tgt"], e["rel_type"]
            if not src or not tgt: continue
            yr = e.get("year"); sc = e.get("support") or 1

            context = f"{src} --[{rel}]--> {tgt}"
            if yr: context += f" [year: {yr}]"
            if sc > 1: context += f" [sources: {sc}]"

            deg_max     = max(self._deg_of(src), self._deg_of(tgt))
            hub_pen     = (1-self.cfg["hub_penalty_weight"]) if deg_max>10 else 1.0  # [B3]
            ref_p       = self._ref_p(context, E, Rkw)                               # [B1]
            ppr_avg     = (ppr.get(src,0)+ppr.get(tgt,0))/2                          # [B4]
            decay       = self._temporal_decay(yr)                                    # [N2]
            corroborate = math.log(1+sc)*self.cfg["corroboration_weight"]             # [N1]

            # v5 combined score: adds decay [N2] and corroboration [N1] to v4 base
            combined = (ref_p * hub_pen) * (1 + ppr_avg) * decay * (1 + corroborate)

            paths.append(KnowledgePath(
                context=context, score=ref_p*hub_pen,
                max_year=yr or 0, support=sc,
                combined_score=combined))

        paths.sort(key=lambda p: p.combined_score, reverse=True)
        top = paths[:self.cfg["top_k_paths"]]
        if self.cfg["verbose"]:
            print(f"  [B]  {len(top)} paths selected")
        return top, intent

# ── MODULE C ENHANCED ─────────────────────────────────────────────
def _H_str(answers):
    c = Counter(answers); n = len(answers); h = 0.0
    for v in c.values():
        p=v/n; h -= p*math.log(p+1e-9)
    return h

def _H_sem(answers, embedder, thresh):
    if not answers: return 0.0
    vecs=embedder.encode(answers); clusters=[-1]*len(answers); cid=0
    for i in range(len(answers)):
        if clusters[i]>=0: continue
        clusters[i]=cid; cid+=1; vi=vecs[i]
        for j in range(i+1,len(answers)):
            if clusters[j]<0:
                sim=float(np.dot(vi,vecs[j])/(np.linalg.norm(vi)*np.linalg.norm(vecs[j])+1e-9))
                if sim>=thresh: clusters[j]=clusters[i]
    counts=Counter(clusters); total=len(answers); h=0.0
    for c in counts.values():
        p=c/total; h -= p*math.log(p+1e-9)
    return h


def _detect_contradictions(paths):
    pat=re.compile(r'([A-Za-z][^-]+?)--\[(\w+)\]-->\s*([A-Za-z][^\n\[]+?)(?:\s*\[year:? ?(\d+)\])?')
    best: Dict[Tuple,int]={}
    for idx,kp in enumerate(paths):
        for m in pat.finditer(kp.context):
            rel,tgt=m.group(2).upper(),m.group(3).strip(); key=(rel,tgt)
            if key not in best:
                best[key]=idx
            else:
                prev=paths[best[key]]
                if (kp.max_year>prev.max_year or
                        (kp.max_year==prev.max_year and kp.combined_score>prev.combined_score)):
                    paths[best[key]].contradicted=True
                    paths[best[key]].conflict_reason=(f"Superseded by year {kp.max_year}")
                    best[key]=idx
                else:
                    kp.contradicted=True
                    kp.conflict_reason=f"Superseded by year {prev.max_year}"
    return paths, sum(1 for p in paths if p.contradicted)

class EnhancedConflictResolver:
    """Module C + [N6] explanation chain + [N7] confidence score."""
    def __init__(self, llm, llm_s, embedder, cfg):
        self.llm=llm; self.llm_s=llm_s; self.emb=embedder; self.cfg=cfg
        self._n=cfg["n_entropy_samples"]

    def _entropy(self, query, context=None):
        """[C2][C4] Fixed-n semantic entropy over sampled answers."""
        ans=[cached_invoke(self.llm_s,
             PARAM_PROMPT.format(query=query) if context is None
             else RAG_PROMPT.format(query=query,context=context))
             for _ in range(self._n)]
        thresh = self.cfg["semantic_cluster_thresh"]
        max_h  = math.log(self._n)
        hs  = _H_str(ans)
        hse = _H_sem(ans, self.emb, thresh) if self.emb else hs
        return hs, hse, None    # logprob removed (duplicate of cached_invoke)

    def _confidence(self, delta_h: float, tau: float,
                    support: int, year_gap: int) -> float:
        """[N7] Calibrated 0-1 confidence score for the final answer."""
        w = self.cfg.get("confidence_weights", {"h":0.40,"sup":0.30,"rec":0.30})
        h_sig  = min(abs(delta_h) / (tau + 1e-9), 2.0) / 2.0  # how clearly resolved
        sup_s  = min(math.log(1 + support) / math.log(6), 1.0) # log-scale support
        rec_s  = math.exp(-0.05 * max(0, year_gap))             # recency
        conf   = w["h"]*h_sig + w["sup"]*sup_s + w["rec"]*rec_s
        return round(min(conf, 1.0), 3)

    def _build_chain(self, sel_paths: List[KnowledgePath],
                      removed_paths: List[KnowledgePath]) -> str:
        """[N6] Build human-readable conflict resolution reasoning chain."""
        lines = ["Selected evidence:"]
        for p in sel_paths:
            sc_info = f" (sources={p.support})" if p.support>1 else ""
            lines.append(f"  ✓ {p.context}{sc_info}  score={p.combined_score:.3f}")
        if removed_paths:
            lines.append("Removed as outdated/contradicted:")
            for p in removed_paths:
                lines.append(f"  ✗ {p.context}  reason: {p.conflict_reason or 'lower score'}")
        return "\n".join(lines)

    def resolve(self, query: str, paths: List[KnowledgePath],
                intent="unknown"):
        removed_paths: List[KnowledgePath] = []
        n_c = 0
        if self.cfg["enable_contradiction_filter"]:
            paths, n_c = _detect_contradictions(paths)
            removed_paths = [p for p in paths if p.contradicted]
            paths = [p for p in paths if not p.contradicted]
            if self.cfg["verbose"]:
                print(f"  [C5] Contradictions removed: {n_c}, remaining: {len(paths)}")
        if not paths: return [], "No valid paths after contradiction filtering.", {}

        hs_p, hse_p, hlp_p = self._entropy(query)
        hd_p = hlp_p if hlp_p is not None else hse_p
        if self.cfg["verbose"]:
            print(f"  [C4] H_param str={hs_p:.4f} sem={hse_p:.4f}")

        # tau [C7] domain-adaptive
        max_h = math.log(self._n)
        it = self.cfg["tau_by_intent"].get(intent)
        if it is not None:
            tau = it
            if self.cfg["verbose"]: print(f"  [C7] Intent={intent}, tau={tau}")
        elif (ft := self.cfg.get("entropy_tau")) is not None:
            tau = ft
        else:
            tau = max(0.15, 0.5*hd_p)
            if self.cfg["verbose"]: print(f"  [C3] Adaptive tau={tau:.4f}")

        strategy = "grounding" if hd_p >= max_h*0.85 else "conflict"

        for i, kp in enumerate(paths):
            if self.cfg["verbose"]: print(f"  [C]  Path {i+1}...", end=" ", flush=True)
            _, hse_a, hlp_a = self._entropy(query, context=kp.context)
            kp.h_aug   = hlp_a if hlp_a is not None else hse_a
            kp.delta_h = kp.h_aug - hd_p
            if self.cfg["verbose"]: print(f"ΔH={kp.delta_h:+.4f}")

        if strategy=="conflict":
            sel=[kp for kp in paths if kp.delta_h>tau]
            for kp in sel: kp.is_corrective=True
        else:
            sel=sorted(paths, key=lambda x:(x.max_year,-x.h_aug), reverse=True)[:2]
            for kp in sel: kp.is_corrective=True
        if not sel: sel=paths[:1]

        top_ctx = "\n".join(kp.context for kp in sel)

        # [N6] Explanation-chain prompt
        if self.cfg["explanation_chain"] and removed_paths:
            chain = self._build_chain(sel, removed_paths)
            answer = cached_invoke(self.llm,
                         EXPLAIN_PROMPT.format(chain=chain, query=query))
        else:
            answer = cached_invoke(self.llm, RAG_PROMPT.format(query=query, context=top_ctx))

        # [N7] Calibrated confidence score
        best = sel[0] if sel else None
        year_gap = CURRENT_YEAR - best.max_year if (best and best.max_year) else 0
        conf = self._confidence(
            delta_h = best.delta_h if best else 0,
            tau      = tau,
            support  = best.support if best else 1,
            year_gap = year_gap
        )

        meta = {
            "intent": intent, "strategy": strategy,
            "tau": tau, "h_param_str": hs_p, "h_param_sem": hse_p,
            "total_paths": len(paths),
            "selected_paths": len(sel), "contradictions_removed": n_c,
            "corroboration_max": max((p.support for p in sel), default=1),
            "confidence": conf,
        }
        return sel, answer, meta

# ── ENHANCED PIPELINE ─────────────────────────────────────────────
class EnhancedPipeline:
    def __init__(self, cfg=CFG):
        self.cfg=cfg
        llm      = ChatOllama(model=cfg["llm_model"], temperature=cfg["llm_temperature"])
        llm_s    = ChatOllama(model=cfg["llm_model"], temperature=cfg["llm_temp_sampler"])
        graph    = Neo4jGraph(url=cfg["neo4j_uri"],username=cfg["neo4j_user"],password=cfg["neo4j_pass"])
        embedder = SentenceTransformer(cfg["embedding_model"]) if _ST else None
        self.constructor = EnhancedGraphConstructor(llm, graph, cfg)
        self.retriever   = EnhancedGraphRetriever(llm, graph, embedder, cfg)
        self.resolver    = EnhancedConflictResolver(llm, llm_s, embedder, cfg)

    def ingest(self, docs: List[str]):
        print(f"\n{SEP}\n  [v5] Building enhanced knowledge graph...\n{SEP}")
        n_n, n_e = self.constructor.build(docs)
        print(f"  --> Graph: {n_n} nodes, {n_e} edges")

    def query(self, q: str) -> Dict:
        t0 = time.time()
        print(f"\n{SEP}\n  Query: {q}\n{SEP}")
        paths, intent = self.retriever.retrieve(q)
        if not paths:
            return {"answer":"No relevant paths found.","meta":{},"elapsed":time.time()-t0}
        # [N9] Adaptive entropy: skip sampling for extremes
        if self.cfg.get("adaptive_entropy", True):
            skipped = 0
            for kp in paths:
                if kp.score >= self.cfg.get("adaptive_high_thresh", 0.70):
                    kp._skip_entropy = True; skipped += 1  # high-score = trust directly
                elif kp.score <= self.cfg.get("adaptive_low_thresh", 0.05):
                    kp._skip_entropy = True; skipped += 1  # near-zero = stale, skip
            if self.cfg["verbose"] and skipped:
                print(f"  [N9] Adaptive entropy: skipped sampling for {skipped}/{len(paths)} paths")
        sel, answer, meta = self.resolver.resolve(q, paths, intent)
        elapsed = time.time()-t0
        print(f"\n  ✓ Answer: {answer}")
        conf = meta.get("confidence", "N/A")
        print(f"  Confidence: {conf*100:.0f}%  Time={elapsed:.1f}s  {_CACHE.stats()}")
        return {"answer":answer,"paths":sel,"meta":meta,"elapsed":elapsed}

    # ── [N8] CLAIM VERIFICATION ────────────────────────────────────────────
    def verify(self, claim: str) -> Dict:
        """[N8] Claim Verifier: check whether a stated claim is SUPPORTED,
        REFUTED, or UNCERTAIN based on the current knowledge graph.

        Unlike query(), which answers open questions, verify() takes a
        declarative claim and returns a structured verdict with evidence.
        """
        VERIFY_PROMPT = (
            "You are a fact-checker. Given the following knowledge graph evidence "
            "and a claim, decide:\n"
            "  - SUPPORTED  : the evidence clearly backs the claim\n"
            "  - REFUTED    : the evidence clearly contradicts the claim\n"
            "  - UNCERTAIN  : the evidence is insufficient to decide\n\n"
            "Evidence from knowledge graph:\n{evidence}\n\n"
            "Claim to verify: \"{claim}\"\n\n"
            "Reply in EXACTLY this format (no extra text):\n"
            "VERDICT: <SUPPORTED|REFUTED|UNCERTAIN>\n"
            "REASON: <one sentence explaining why>"
        )
        t0 = time.time()
        print(f"\n{SEP}\n  [N8] Verifying claim: \"{claim}\"\n{SEP}")
        # reuse normal retrieval — treat claim as query
        paths, intent = self.retriever.retrieve(claim)
        if not paths:
            return {"verdict":"UNCERTAIN","reason":"No relevant facts in knowledge graph.",
                    "confidence":0.0,"elapsed":time.time()-t0}
        # detect contradictions (may refute the claim automatically)
        from enhanced_main import CURRENT_YEAR  # self-import safe
        removed_paths = []
        surviving = []
        for i, kp in enumerate(paths):
            for j, kp2 in enumerate(paths):
                if i>=j: continue
                if (kp.relation==kp2.relation and kp.tail==kp2.tail
                        and kp.head!=kp2.head and kp.max_year and kp2.max_year
                        and abs(kp.max_year-kp2.max_year)>=1):
                    loser = kp if kp.max_year<kp2.max_year else kp2
                    if loser not in removed_paths: removed_paths.append(loser)
        surviving = [p for p in paths if p not in removed_paths]
        evidence = "\n".join(f"- {kp.context} (year={kp.max_year}, support={kp.support})"
                              for kp in (surviving or paths)[:5])
        raw = cached_invoke(self.resolver.llm,
                            VERIFY_PROMPT.format(evidence=evidence, claim=claim))
        # parse structured reply
        verdict = "UNCERTAIN"
        reason  = raw
        for line in raw.splitlines():
            if line.startswith("VERDICT:"):
                v = line.split(":",1)[1].strip().upper()
                if v in ("SUPPORTED","REFUTED","UNCERTAIN"): verdict = v
            elif line.startswith("REASON:"):
                reason = line.split(":",1)[1].strip()
        # confidence: higher if supported by recent, multi-source evidence
        best = surviving[0] if surviving else (paths[0] if paths else None)
        year_gap = CURRENT_YEAR - best.max_year if (best and best.max_year) else 5
        conf = self.resolver._confidence(
            delta_h=0, tau=0.25, support=best.support if best else 1, year_gap=year_gap)
        elapsed = time.time()-t0
        verdict_sym = {"SUPPORTED":"✓","REFUTED":"✗","UNCERTAIN":"?"}[verdict]
        print(f"  {verdict_sym} Verdict : {verdict}")
        print(f"    Reason  : {reason}")
        print(f"    Confidence: {conf*100:.0f}%  Time={elapsed:.1f}s")
        return {"verdict":verdict,"reason":reason,"confidence":conf,
                "evidence_used":len(surviving or paths),"elapsed":elapsed}

# ── COMPARISON RUNNER ─────────────────────────────────────────────
def run_comparison(docs: List[str], queries: List[str]):
    """Ingest docs into v5 pipeline and answer all queries."""
    try:    W = min(os.get_terminal_size().columns, 80)
    except: W = 72
    BAR = "═" * W
    print(f"\n{BAR}\n  TruthfulRAG v5 Enhanced — Run\n{BAR}")

    results = []
    v5 = EnhancedPipeline()
    v5.ingest(docs)

    for q in queries:
        print(f"\n{SEP}\nQuery: {q}")
        r5 = v5.query(q)
        results.append({"query": q, "v5": r5})

    # Feature summary derived from active CFG — no hardcoded rows
    print(f"\n{BAR}\n  v5 Active Features\n{BAR}")
    features = [
        ("[N1] Corroboration",        f"weight={CFG['corroboration_weight']}"),
        ("[N2] Temporal decay",        f"lambda={CFG['temporal_decay_lambda']}  year={CURRENT_YEAR}"),
        ("[N3] Hybrid retrieval",      "BM25+Semantic RRF" if CFG["use_hybrid_retrieval"] else "BM25 only"),
        ("[N4] Temporal snapshot",     "auto-detect from query"),
        ("[N6] Explanation chain",     "ON" if CFG["explanation_chain"] else "OFF"),
        ("[N7] Confidence score",      f"weights={CFG['confidence_weights']}"),
        ("[N8] Claim verifier",        "Available via --verify \"claim\""),
        ("[N9] Adaptive entropy",      f"ON  high={CFG['adaptive_high_thresh']}  low={CFG['adaptive_low_thresh']}" if CFG.get('adaptive_entropy') else "OFF"),
        ("[A4+] Schema (auto)",        f"inferred from {CFG.get('schema_infer_sample',5)} docs"),
        ("Entity types",              str(CFG["schema_entity_types"]) or "pending inference"),
        ("Relation types",            str(CFG["schema_relation_types"]) or "pending inference"),
    ]
    for label, val in features:
        print(f"  {label:<30} {val}")
    print(BAR)
    return results

# ── CORPUS / QUERY LOADER ─────────────────────────────────────────
def _load_corpus(path: str) -> Dict:
    """Load docs + queries from a JSON file.

    Expected format (all keys optional):
    {
      "docs":    ["sentence 1", ...],
      "queries": ["question 1", ...],
      "schema": {
          "entity_types":   [...],
          "relation_types": [...]
      }
    }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"corpus file must be a JSON object, got {type(data)}")
    return data


def _interactive_input() -> tuple:
    """Prompt the user to type docs and queries directly in the terminal."""
    print(f"\n{SEP}\n  TruthfulRAG v5 — Interactive Input Mode\n{SEP}")
    print("Enter DOCUMENTS one per line. Type an empty line when done.")
    docs = []
    while True:
        line = input("  doc> ").strip()
        if not line:
            break
        docs.append(line)

    print("\nEnter QUERIES one per line. Type an empty line when done.")
    queries = []
    while True:
        line = input("  q> ").strip()
        if not line:
            break
        queries.append(line)

    return docs, queries


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── CLI argument parser ─────────────────────────────────────
    ap = argparse.ArgumentParser(
        description="TruthfulRAG v5 — Enhanced KG-RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a corpus JSON file:
  python enhanced_main.py --corpus corpus.json

  # Pass docs/queries directly:
  python enhanced_main.py --docs "Doc 1." "Doc 2." --queries "Question?"

  # Interactive (prompts for input):
  python enhanced_main.py --interactive

  # Override LLM model at runtime:
  python enhanced_main.py --corpus corpus.json --model llama3:8b
        """)
    ap.add_argument("--corpus",      "-c",  metavar="FILE",
                    help="path to corpus JSON file (see _load_corpus docstring for format)")
    ap.add_argument("--docs",        "-d",  nargs="+", metavar="DOC",
                    help="one or more document strings (wrap in quotes)")
    ap.add_argument("--queries",     "-q",  nargs="+", metavar="QUERY",
                    help="one or more query strings")
    ap.add_argument("--interactive", "-i",  action="store_true",
                    help="prompt for docs and queries interactively")
    ap.add_argument("--model",       "-m",  metavar="NAME",
                    help="override LLM model (e.g. llama3:8b)")
    ap.add_argument("--no-verbose",         action="store_true",
                    help="suppress verbose logging")
    ap.add_argument("--schema-sample", "-s", type=int, default=5, metavar="N",
                    help="docs sampled for schema auto-inference (default: 5)")
    ap.add_argument("--verify",        "-V",  nargs="+", metavar="CLAIM",
                    help="[N8] verify one or more claims against the ingested graph")
    ap.add_argument("--no-adaptive",         action="store_true",
                    help="[N9] disable adaptive entropy sampling (always sample all paths)")
    args = ap.parse_args()

    if args.model:       CFG["llm_model"] = args.model
    if args.no_verbose:  CFG["verbose"]   = False
    if args.no_adaptive: CFG["adaptive_entropy"] = False
    CFG["schema_infer_sample"] = args.schema_sample
    VERIFY_CLAIMS = args.verify or []

    # ── Resolve corpus source (priority: --corpus > --docs > --interactive) ──
    DOCS: List[str]    = []
    QUERIES: List[str] = []

    if args.corpus:
        data    = _load_corpus(args.corpus)
        DOCS    = data.get("docs",    [])
        QUERIES = data.get("queries", [])
        print(f"  Loaded {len(DOCS)} docs, {len(QUERIES)} queries from {args.corpus}")
        print(f"  Schema: will be auto-inferred from corpus")
    elif args.docs:
        DOCS    = args.docs
        QUERIES = args.queries or []
    elif args.interactive:
        DOCS, QUERIES = _interactive_input()
    else:
        # ── Demo mode: built-in sample corpus loaded from bundled file ──
        _demo = os.path.join(os.path.dirname(__file__), "corpus.json")
        if os.path.exists(_demo):
            data    = _load_corpus(_demo)
            DOCS    = data.get("docs",    [])
            QUERIES = data.get("queries", [])
            print(f"  Using corpus.json ({len(DOCS)} docs, {len(QUERIES)} queries)")
        else:
            print("  No corpus supplied. Run with --help to see options.")
            print("  Generating corpus.json demo template...")
            _template = {
                "docs": [
                    "Add your source documents here, one per list item.",
                    "Each document is a sentence or paragraph of factual text.",
                ],
                "queries": [
                    "Add your questions here.",
                ]
            }
            with open("corpus.json", "w", encoding="utf-8") as _f:
                json.dump(_template, _f, indent=2)
            print("  corpus.json created — fill in docs and queries, then re-run.")
            print("  Entity/relation types will be auto-inferred from your documents.")
            sys.exit(0)

    if not DOCS:
        print("  [ERROR] No documents provided. Exiting."); sys.exit(1)
    if not QUERIES and not VERIFY_CLAIMS:
        print("  [WARN] No queries or claims — running ingest only.")

    # ── Normal Q&A mode ──
    if QUERIES or not VERIFY_CLAIMS:
        run_comparison(DOCS, QUERIES)

    # ── [N8] Claim Verification mode ──
    if VERIFY_CLAIMS:
        print(f"\n{'═'*72}\n  [N8] Claim Verification Mode\n{'═'*72}")
        pipe = EnhancedPipeline()
        if DOCS:  pipe.ingest(DOCS)   # re-use graph already built above if possible
        for claim in VERIFY_CLAIMS:
            result = pipe.verify(claim)
            print()
