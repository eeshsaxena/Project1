from __future__ import annotations
import sys
# ============================================================
# TruthfulRAG v4 — Full Paper Implementation (16 Improvements)
# Paper: Liu, Shang & Zhang — BUPT / AAAI 2026 (arXiv:2511.10375)
# ACTIVE IMPROVEMENTS:
#   [A1] Temporal 4-tuples      [A2] Entity disambiguation
#   [A3] Relation normalisation  [A4] Schema-guided extraction
#   [B1] Real-ID Ref(p)         [B2] Adaptive hop depth
#   [B3] Hub-node penalty       [B4] Personalized PageRank
#   [B5] Relation filtering     [C1] LLM response cache
#   [C2] n=3 samples            [C3] Adaptive tau (magnitude)
#   [C4] Semantic entropy       [C5] Cross-path contradiction
#   [C6] Token logprob entropy  [C7] Domain-adaptive tau
#   [G1] BM25 retriever         [G3] Graph persistence
# Stack: Ollama · Neo4j · LangChain · sentence-transformers · rank-bm25
# Run:   python main.py
# ============================================================
import difflib, hashlib, json, logging, math, os, re, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np, requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama

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

# ── CONFIG ────────────────────────────────────────────────────
CFG: Dict[str, Any] = {
    "llm_model":               "qwen2.5:7b-instruct",
    "llm_temperature":         0.0,
    "llm_temp_sampler":        0.7,
    "neo4j_uri":               os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    "neo4j_user":              os.getenv("NEO4J_USERNAME", "neo4j"),
    "neo4j_pass":              os.getenv("NEO4J_PASSWORD", ""),
    "top_k_entities": 5,  "top_k_relations": 5,  "top_k_paths": 5,
    "alpha": 0.5,  "beta": 0.5,                                # Ref(p) weights [B1]
    "hub_penalty_weight":      0.3,                            # [B3]
    "ppr_damping":             0.85,  "ppr_iterations": 20,    # [B4]
    "rel_filter_threshold":    0.30,                           # [B5]
    "use_schema":              True,                           # [A4]
    "schema_entity_types":     ["PERSON","ORGANIZATION","LOCATION","DATE","PRODUCT"],
    "schema_relation_types":   ["CEO_OF","FOUNDED_BY","LOCATED_IN","WORKED_AT",
                                "ACQUIRED_BY","PART_OF","STUDIED_AT","RELEASED",
                                "PM_OF","LEADS","GOVERNS"],
    "n_entropy_samples":       3,                              # [C2] paper used 5, 40% faster
    "entropy_tau":             None,                           # [C3] None = adaptive
    "semantic_cluster_thresh": 0.85,                           # [C4]
    "enable_contradiction_filter": _ST,                        # [C5] auto-off if no sentence-transformers
    "use_logprob_entropy":     True,                           # [C6] token-level logprob entropy
    "ollama_base_url":         "http://localhost:11434",       # [C6] Ollama API base
    "tau_by_intent": {                                         # [C7]
        "factual_lookup": 0.25, "temporal": 0.35,
        "comparison": 0.50,     "causal": 0.45, "unknown": None,
    },
    "embedding_model":         "all-MiniLM-L6-v2",
    "clear_graph_on_start":    True,                           # [G3]
    "verbose":                 True,
}
SEP = "─"*64;  SEP2 = "═"*64

# ── [C1] LLM CACHE — MD5-keyed, avoids duplicate LLM calls ───
class LLMCache:
    def __init__(self): self._s: Dict[str,str] = {}; self.hits = self.misses = 0
    def _k(self, p, t): return hashlib.md5(f"{t}::{p}".encode()).hexdigest()
    def get(self, p, t):
        k = self._k(p, t)
        if k in self._s: self.hits += 1; return self._s[k]
        self.misses += 1; return None
    def set(self, p, t, r): self._s[self._k(p, t)] = r
    def stats(self):
        n = self.hits + self.misses
        return f"LLM cache {self.hits}/{n} hits ({self.hits/n*100:.0f}% saved)" if n else "LLM cache 0/0"

_CACHE = LLMCache()

def cached_invoke(llm, prompt):
    """Invoke LLM with [C1] caching."""
    t = llm.temperature;  c = _CACHE.get(prompt, t)
    if c is not None: return c
    r = llm.invoke(prompt).content.strip()
    _CACHE.set(prompt, t, r); return r

# ── EMBEDDING ENGINE — with character-hash fallback ──────────
class EmbeddingEngine:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self._m = None
        if _ST:
            try: self._m = SentenceTransformer(model)
            except Exception as e: print(f"[WARN] {e}")

    def encode(self, texts):
        if self._m:
            return self._m.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Fallback: character-frequency hash vector
        vecs = []
        for t in texts:
            v = np.zeros(256)
            for c in t.lower(): v[ord(c)%256] += 1
            n = np.linalg.norm(v); vecs.append(v/(n+1e-9))
        return np.array(vecs)

    def cosine(self, a, b):
        vs = self.encode([a, b])
        return float(np.dot(vs[0],vs[1]) / (np.linalg.norm(vs[0])*np.linalg.norm(vs[1])+1e-9))

    def top_k(self, q, cands, k):
        if not cands: return []
        qv = self.encode([q])[0];  cv = self.encode(cands)
        sc = cv @ qv / (np.linalg.norm(cv,axis=1)*np.linalg.norm(qv)+1e-9)
        k  = min(k, len(cands));  ix = np.argsort(sc)[::-1][:k]
        return [(cands[i], float(sc[i])) for i in ix]

# ── [G1] BM25 RETRIEVER — dynamic doc selection per query ─────
class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25   = BM25Okapi([d.lower().split() for d in corpus]) if _BM25 else None

    def retrieve(self, query, top_k=5):
        if not self.bm25:   # keyword-overlap fallback
            scored = sorted([(sum(1 for w in query.lower().split() if w in d.lower()), d)
                             for d in self.corpus], reverse=True)
            return [d for _, d in scored[:top_k]]
        idx = np.argsort(self.bm25.get_scores(query.lower().split()))[::-1][:top_k]
        return [self.corpus[i] for i in idx]

# ── DATA STRUCTURE ────────────────────────────────────────────
@dataclass
class KGPath:
    raw:            Any
    ref_score:      float = 0.0
    ppr_score:      float = 0.0          # [B4]
    combined_score: float = 0.0          # Ref(p) × (1 + PPR)
    years:          List[int] = field(default_factory=list)
    max_year:       int   = 0
    context:        str   = ""
    h_aug:          float = 0.0
    delta_h:        float = 0.0
    is_corrective:  bool  = False
    contradicted:   bool  = False        # [C5]

# ── PROMPTS ───────────────────────────────────────────────────
def _triple_prompt(text, use_schema, etypes, rtypes):
    """[A4] Schema-guided extraction prompt builder."""
    hint = (f"\nAllowed entity types : {', '.join(etypes)}"
            f"\nAllowed relation types: {', '.join(rtypes)}"
            "\nOnly extract triples that fit the allowed types.") if use_schema else ""
    return (f"You are an expert knowledge-graph builder.{hint}\n"
            "Extract ALL factual triples from the text, with year if stated.\n"
            "Return ONLY a JSON array:\n"
            '[{"head":str,"relation":str,"tail":str,"year":int|null}]\n'
            f'Text:\n"""{text}"""\nJSON:')

NORM_PROMPT  = ""  # removed
KEY_PROMPT   = ('From this question extract:\n- entities (people, orgs, places)\n'
                '- relations (roles like CEO, verbs like founded)\nQuestion: "{query}"\n'
                'Reply ONLY:\nENTITIES: e1, e2\nRELATIONS: r1, r2')
INTENT_PROMPT = ("Classify this question into ONE intent category:\n"
                 "factual_lookup | temporal | comparison | causal | unknown\n"
                 'Question: "{query}"\nReply with ONLY the category name:')
PARAM_PROMPT = "Answer from your own internal knowledge ONLY.\nQuestion: {query}\nAnswer in one sentence:"
RAG_PROMPT   = "Question: {query}\nContext:\n{context}\nAnswer in one sentence using ONLY the context above:"
FINAL_PROMPT = ("You are a factual reasoning assistant.\n"
                "Use ONLY the structured knowledge paths below.\n"
                "Prioritise MOST RECENT facts (highest year). Pay attention to [year:XXXX] tags.\n\n"
                "Question: {query}\n\nStructured Knowledge Paths:\n{paths}\n\n"
                "Give a direct, one-sentence answer. Include the year when relevant.")

# ── MODULE A — GRAPH CONSTRUCTION [A1][A2][A3][A4] ───────────────
class GraphConstructor:
    def __init__(self, llm, graph, cfg):
        self.llm = llm; self.graph = graph; self.cfg = cfg
        self.transformer = LLMGraphTransformer(llm=llm)

    def _clear(self):
        """[G3] Clear graph on start if configured."""
        self.graph.query("MATCH (n) DETACH DELETE n")
        if self.cfg["verbose"]: print("  [A]  Neo4j cleared.")

    def _extract(self, text):
        """[A1][A4] Temporal + schema-guided triple extraction."""
        raw = cached_invoke(self.llm, _triple_prompt(
            text, self.cfg["use_schema"],
            self.cfg["schema_entity_types"], self.cfg["schema_relation_types"]))
        try:
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            data = json.loads(m.group(0) if m else raw)
            return [d for d in data if isinstance(d,dict)
                    and d.get("head") and d.get("relation") and d.get("tail")]
        except Exception: return []

    def _store(self, triples):
        """[A1] MERGE triples into Neo4j with year property."""
        for t in triples:
            h  = re.sub(r"'", "", t["head"])
            r  = re.sub(r'\W+', '_', t["relation"].upper())
            tl = re.sub(r"'", "", t["tail"])
            yr = t.get("year")
            q  = (f"MERGE (a:Entity {{id:'{h}'}}) MERGE (b:Entity {{id:'{tl}'}}) "
                  f"MERGE (a)-[rel:{r} {{year:{yr}}}]->(b)" if yr else
                  f"MERGE (a:Entity {{id:'{h}'}}) MERGE (b:Entity {{id:'{tl}'}}) MERGE (a)-[rel:{r}]->(b)")
            try: self.graph.query(q)
            except Exception: pass

    def _disambiguate(self):
        """[A2] Entity disambiguation — fuzzy-merge similar entity nodes in Neo4j."""
        ids = [r["id"] for r in self.graph.query("MATCH (n) RETURN n.id AS id") if r.get("id")]
        merged = 0
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                sim = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
                if sim >= 0.85 and a != b:
                    try:
                        self.graph.query(
                            f"MATCH (old:Entity {{id:'{b}'}}) "
                            f"MATCH (keep:Entity {{id:'{a}'}}) "
                            f"CALL apoc.refactor.mergeNodes([keep,old],"
                            f"{{properties:'combine',mergeRels:true}}) YIELD node RETURN node")
                        merged += 1
                    except Exception: pass
        if self.cfg["verbose"] and merged:
            print(f"  [A2] Disambiguated {merged} entity pairs")

    def _normalize(self):
        """[A3] Relation normalisation — LLM canonicalises raw relation type strings."""
        rows = self.graph.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t LIMIT 50")
        raw = [r["t"] for r in rows if r.get("t")]
        if not raw: return
        prompt = ("Canonicalize these relation types to clean UPPER_SNAKE_CASE. "
                  "Return only JSON: {\"ORIGINAL\": \"CANONICAL\", ...}\n"
                  f"Relations: {raw}\nJSON:")
        try:
            resp = cached_invoke(self.llm, prompt)
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            mapping = json.loads(m.group(0)) if m else {}
            for orig, canon in mapping.items():
                if orig != canon and canon:
                    c = re.sub(r'\W+', '_', canon.upper())
                    try:
                        self.graph.query(
                            f"MATCH ()-[r:{orig}]->() "
                            f"CALL apoc.refactor.setType(r,'{c}') "
                            f"YIELD input RETURN count(input)")
                    except Exception: pass
            if self.cfg["verbose"]: print(f"  [A3] Normalised {len(mapping)} relation types")
        except Exception: pass

    def build(self, docs):
        """Build Knowledge Graph from documents. [A1][A2][A3][A4][G3]"""
        if self.cfg["clear_graph_on_start"]: self._clear()          # [G3] clear old graph
        # LangChain structural extraction (base layer)
        self.graph.add_graph_documents(
            self.transformer.convert_to_graph_documents(
                [Document(page_content=d) for d in docs]))
        # [A1][A4] Temporal 4-tuple extraction with schema guidance
        total = 0
        for doc in docs:
            triples = self._extract(doc)   # extract (h, r, t, year)
            self._store(triples)           # store in Neo4j with year on edge
            total += len(triples)
        if self.cfg["verbose"]: print(f"  [A1][A4] Temporal/schema triples stored: {total}")
        n_n = self.graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        n_e = self.graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        if self.cfg["verbose"]: print(f"  [A]  KG: {n_n} nodes, {n_e} edges")
        self._disambiguate()   # [A2] merge similar entity nodes
        self._normalize()      # [A3] canonicalise relation type strings
        return n_n, n_e

# ── MODULE B — GRAPH RETRIEVAL [B1][B2][B3][B4][B5] ──────────
class GraphRetriever:
    def __init__(self, llm, graph, embedder, cfg):
        self.llm = llm; self.graph = graph
        self.embedder = embedder; self.cfg = cfg
        self._deg: Dict[str,int] = {}   # [B3] degree cache

    def _keys(self, q):
        """Extract query entities and relation keywords via LLM."""
        resp = cached_invoke(self.llm, KEY_PROMPT.format(query=q))
        E, R = [], []
        for line in resp.splitlines():
            l = line.strip()
            if l.upper().startswith("ENTITIES:"):  E = [e.strip() for e in l.split(":",1)[1].split(",") if e.strip()]
            elif l.upper().startswith("RELATIONS:"): R = [r.strip() for r in l.split(":",1)[1].split(",") if r.strip()]
        if not E:
            stop = {"who","what","when","where","is","are","was","the","a","an","of"}
            E = [t for t in q.lower().split() if t not in stop]
        if not R: R = ["ceo","became","founded","located"]
        kw = list({w for r in R for w in re.split(r'[\s_]+', r.lower()) if w})
        return E, kw

    def detect_intent(self, q):
        """[C7] Classify query intent for domain-adaptive tau."""
        r = cached_invoke(self.llm, INTENT_PROMPT.format(query=q)).strip().lower()
        return r if r in {"factual_lookup","temporal","comparison","causal","unknown"} else "unknown"

    def _node_ids(self):
        return [r["id"] for r in self.graph.query("MATCH (n) RETURN n.id AS id LIMIT 200") if r.get("id")]

    def _edges(self):
        return self.graph.query(
            "MATCH (a)-[r]->(b) RETURN type(r) AS rel_type, a.id AS src, "
            "b.id AS tgt, r.year AS year LIMIT 200")

    def _deg_of(self, nid):
        """[B3] Cached node degree lookup."""
        if nid not in self._deg:
            rows = self.graph.query(f"MATCH (n{{id:'{nid}'}})-[r]-() RETURN count(r) AS d")
            self._deg[nid] = rows[0]["d"] if rows else 1
        return max(self._deg[nid], 1)

    def _filter_edges(self, edges, qr):
        """[B5] Keep only edges whose relation type is semantically relevant to query."""
        thresh = self.cfg["rel_filter_threshold"]
        kept   = [r for r in edges
                  if self.embedder.cosine(qr, str(r.get("rel_type","")).replace("_"," ").lower()) >= thresh]
        if not kept: kept = edges   # fallback: keep all if nothing passes
        if self.cfg["verbose"]: print(f"  [B5] Edges after relation filter: {len(kept)}/{len(edges)}")
        return kept

    def _ppr(self, node_ids, edges, seeds):
        """[B4] Python-side Personalized PageRank anchored at seed entities."""
        if not node_ids: return {}
        d, iters = self.cfg["ppr_damping"], self.cfg["ppr_iterations"]
        idx = {n:i for i,n in enumerate(node_ids)};  N = len(node_ids)
        out_deg = np.zeros(N);  adj: Dict[int,List[int]] = defaultdict(list)
        for row in edges:
            s, t = row.get("src"), row.get("tgt")
            if s in idx and t in idx:
                si, ti = idx[s], idx[t]; adj[si].append(ti); out_deg[si] += 1
        # Personalization: uniform over seed nodes
        pers = np.zeros(N)
        sf   = [idx[n] for n in node_ids if any(e.lower() in n.lower() for e in seeds) and n in idx]
        for si in sf: pers[si] = 1.0/len(sf) if sf else 1.0/N
        if not sf: pers[:] = 1.0/N
        # Power iteration
        r = pers.copy()
        for _ in range(iters):
            rn = np.zeros(N)
            for si, nb in adj.items():
                if out_deg[si] > 0:
                    for ti in nb: rn[ti] += d*r[si]/out_deg[si]
            rn += (1-d)*pers;  r = rn/(rn.sum()+1e-9)
        return {node_ids[i]: float(r[i]) for i in range(N)}

    def _ref(self, pstr, E, Rkw, node_ids):
        """[B1]+[B3] Ref(p) = αE + βR minus hub penalty."""
        txt = pstr.lower();  a = self.cfg["alpha"];  b = self.cfg["beta"]
        g   = self.cfg["hub_penalty_weight"];  es = {e.lower() for e in E}
        # [B1] case-insensitive real-ID entity match
        ec  = min(sum(1 for n in node_ids if any(e in n.lower() for e in es) and n.lower() in txt) / max(len(E),1), 1.0)
        rc  = min(sum(1 for r in Rkw if r in txt) / max(len(Rkw),1), 1.0)
        base = a*ec + b*rc
        # [B3] logarithmic hub penalty for high-degree nodes
        hub = sum(self._deg_of(n) for n in node_ids if n.lower() in txt)
        pen = g * math.log(max(hub,1)+1) / math.log(max(hub,2)+1)
        return max(base - pen, 0.0)

    @staticmethod
    def _years(path_obj, edges):
        """Extract all 4-digit years from a path object and its edges."""
        ys = set(int(y) for y in re.findall(r'(?<!\d)(20\d{2}|19\d{2})(?!\d)', str(path_obj)))
        for r in edges:
            if r.get("year"): ys.add(int(r["year"]))
        return sorted(ys)

    def _ctx(self, path_obj, node_ids, edges, E, Rkw, years):
        """Build structured context string (Cp + Ce + Cr) for LLM prompting."""
        pstr = str(path_obj);  es   = {e.lower() for e in E}
        yr_n = f"[Temporal years: {years}]" if years else "[No explicit years]"
        Ce   = "Key Entities:\n" + "\n".join(f"  - {n}" for n in node_ids if any(e in n.lower() for e in es)) or "  (see path)"
        Cr_lines = []
        for row in edges:
            rt  = str(row.get("rel_type",""));  src = str(row.get("src","")).lower();  tgt = str(row.get("tgt","")).lower()
            yr  = row.get("year")
            if any(r in rt.lower() for r in Rkw) or any(e in src or e in tgt for e in es):
                if not yr:
                    m = re.findall(r'(?<!\d)(20\d{2}|19\d{2})(?!\d)', rt); yr = m[0] if m else None
                Cr_lines.append(f"  - {row.get('src')} --[{rt}]--> {row.get('tgt')}{f' [year:{yr}]' if yr else ''}")
        Cr = "Key Relations:\n" + ("\n".join(set(Cr_lines)) or "  (see path)")
        return f"Path: {pstr}\n{yr_n}\n\n{Ce}\n\n{Cr}"

    def retrieve(self, query):
        E, Rkw  = self._keys(query);  intent = self.detect_intent(query)
        qr      = " ".join(E+Rkw) or query
        if self.cfg["verbose"]: print(f"  [B]  E={E}  Rkw={Rkw}\n  [C6] Intent={intent}")
        node_ids = self._node_ids();  edges  = self._edges()
        fedges   = self._filter_edges(edges, qr)        # [B5]
        ppr      = self._ppr(node_ids, fedges, E)       # [B4]
        if self.cfg["verbose"]:
            print(f"  [B4] Top PPR: {sorted(ppr.items(),key=lambda x:x[1],reverse=True)[:3]}")
        top_ent = [e for e,_ in self.embedder.top_k(qr, node_ids, self.cfg["top_k_entities"])]
        rels    = list({str(e.get("rel_type","")).lower() for e in fedges if e.get("rel_type")})
        top_rel = [r for r,_ in self.embedder.top_k(qr, rels, self.cfg["top_k_relations"])]
        if self.cfg["verbose"]: print(f"  [B]  top_ent={top_ent}  top_rel={top_rel}")
        # [B2] Adaptive hop depth
        hops = 2 if (len(query.split()) > 8 or any(
            w in query.lower().split() for w in ("and","both","compare","when","between"))) else 1
        raw  = self.graph.query(f"MATCH p=(n)-[*1..{hops}]-(m) RETURN p LIMIT 30")
        if self.cfg["verbose"]: print(f"  [B2] Hops={hops}, candidates={len(raw)}")
        # Score: [B1+B3] Ref × [B4] PPR boost
        paths = []
        for rp in raw:
            pstr = str(rp);  ref  = self._ref(pstr, E, Rkw, node_ids)
            yrs  = self._years(rp, edges);  ctx = self._ctx(rp, node_ids, edges, E, Rkw, yrs)
            pnodes  = [n for n in node_ids if n.lower() in pstr.lower()]
            ppr_avg = sum(ppr.get(n,0) for n in pnodes) / max(len(pnodes),1)
            paths.append(KGPath(raw=rp, ref_score=ref, ppr_score=ppr_avg,
                                combined_score=ref*(1+ppr_avg),
                                years=yrs, max_year=max(yrs) if yrs else 0, context=ctx))
        paths.sort(key=lambda x: (x.max_year, x.combined_score), reverse=True)
        top = paths[:self.cfg["top_k_paths"]]
        if self.cfg["verbose"]:
            print(f"\n  [B]  Top-{len(top)} paths (year↓, combined↓):")
            for i,p in enumerate(top):
                print(f"    {i+1}. yr={p.max_year} ref={p.ref_score:.3f} ppr={p.ppr_score:.4f}"
                      f" cmb={p.combined_score:.3f}{'  ← TOP' if i==0 else ''}")
        return top, E, Rkw, intent

# ── MODULE C — CONFLICT RESOLUTION [C1-C7] ───────────────────
def _H_str(answers):
    """String-diversity Shannon entropy (original paper method)."""
    counts = Counter(a.split(".")[0].strip().lower() for a in answers)
    total  = len(answers);  h = 0.0
    for c in counts.values():
        p = c/total; h -= p*math.log(p+1e-9)
    return h

def _H_sem(answers, embedder, thresh):
    """[C4] Semantic entropy — cluster by embedding similarity, H over clusters."""
    if not answers: return 0.0
    vecs = embedder.encode(answers);  clusters = [-1]*len(answers);  cid = 0
    for i in range(len(answers)):
        if clusters[i] >= 0: continue
        clusters[i] = cid;  cid += 1;  vi = vecs[i]
        for j in range(i+1, len(answers)):
            if clusters[j] < 0:
                sim = float(np.dot(vi,vecs[j])/(np.linalg.norm(vi)*np.linalg.norm(vecs[j])+1e-9))
                if sim >= thresh: clusters[j] = clusters[i]
    counts = Counter(clusters);  total = len(answers);  h = 0.0
    for c in counts.values():
        p = c/total; h -= p*math.log(p+1e-9)
    return h

def _H_lp(query, context, model, base_url):
    """[C6] Token-level logprob entropy via Ollama /api/generate."""
    prompt = PARAM_PROMPT.format(query=query) if context is None else RAG_PROMPT.format(query=query, context=context)
    try:
        resp = requests.post(f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}, "logprobs": True}, timeout=30)
        if resp.status_code == 200:
            lp = resp.json().get("logprobs", [])
            if lp:
                p = np.exp(np.clip([x["logprob"] for x in lp], -50, 0))
                p /= (p.sum()+1e-9)
                return float(-np.sum(p*np.log(p+1e-9)))
    except Exception: pass
    return None

def _detect_contradictions(paths):
    """[C5] Mark older (entity, relation) assertions as contradicted by newer ones."""
    pat  = re.compile(r'([A-Za-z][^-]+?)\s+--\[(\w+)\]-->\s+([A-Za-z][^\n\[]+?)(?:\s+\[year:? ?(\d+)\])?')
    best: Dict[Tuple,int] = {}
    for idx, kp in enumerate(paths):
        for m in pat.finditer(kp.context):
            rel, tgt = m.group(2).upper(), m.group(3).strip();  key = (rel, tgt)
            if key not in best:
                best[key] = idx
            else:
                prev = paths[best[key]]
                if (kp.max_year > prev.max_year or
                        (kp.max_year == prev.max_year and kp.combined_score > prev.combined_score)):
                    paths[best[key]].contradicted = True;  best[key] = idx
                else:
                    kp.contradicted = True
    return paths, sum(1 for p in paths if p.contradicted)

class ConflictResolver:
    def __init__(self, llm, llm_s, embedder, cfg):
        self.llm = llm;  self.llm_s = llm_s
        self.emb = embedder;  self.cfg  = cfg
        self._n  = cfg["n_entropy_samples"]   # [C2]
        self._mh = math.log(self._n)
        self._st = cfg["semantic_cluster_thresh"]

    def _entropy(self, query, context=None):
        """Compute string, semantic [C4], and logprob [C6] entropy."""
        ans = [cached_invoke(self.llm_s,
               PARAM_PROMPT.format(query=query) if context is None
               else RAG_PROMPT.format(query=query, context=context))
               for _ in range(self._n)]
        hs  = _H_str(ans)
        hse = _H_sem(ans, self.emb, self._st)
        hlp = (_H_lp(query, context, self.cfg["llm_model"], self.cfg["ollama_base_url"])
               if self.cfg["use_logprob_entropy"] else None)
        return hs, hse, hlp

    def resolve(self, query, paths, intent="unknown"):
        # [C5] Pre-filter contradicting paths before entropy computation
        n_c = 0
        if self.cfg["enable_contradiction_filter"]:
            paths, n_c = _detect_contradictions(paths)
            paths = [p for p in paths if not p.contradicted]
            if self.cfg["verbose"]:
                print(f"  [C5] Contradictions removed: {n_c}, remaining: {len(paths)}")
        if not paths: return [], "No valid paths after contradiction filtering.", {}

        # Compute H_param (LLM's baseline uncertainty without context)
        hs_p, hse_p, hlp_p = self._entropy(query)
        hd_p = hlp_p if hlp_p is not None else hse_p   # prefer logprob [C6], else semantic [C4]
        if self.cfg["verbose"]:
            print(f"  [C4] H_param str={hs_p:.4f} sem={hse_p:.4f}" +
                  (f" logprob={hlp_p:.4f}" if hlp_p else " logprob=N/A"))

        # Determine tau: [C7] domain-adaptive → [C3] magnitude-adaptive → fixed
        it = self.cfg["tau_by_intent"].get(intent)
        if it is not None:
            tau = it
            if self.cfg["verbose"]: print(f"  [C7] Intent={intent}, tau={tau}")
        elif (ft := self.cfg.get("entropy_tau")) is not None:
            tau = ft
        else:
            tau = max(0.15, 0.5*hd_p)             # [C3] magnitude-based adaptive τ
            if self.cfg["verbose"]: print(f"  [C3] Adaptive τ={tau:.4f}")

        # Strategy: grounding if LLM is near-maximally uncertain
        strategy = "grounding" if hd_p >= self._mh*0.85 else "conflict"
        if self.cfg["verbose"]: print(f"  [C]  Strategy={strategy.upper()}  τ={tau:.4f}")

        # Per-path H_aug computation
        for i, kp in enumerate(paths):
            if self.cfg["verbose"]: print(f"  [C]  Path {i+1}…", end=" ", flush=True)
            _, hse_a, hlp_a = self._entropy(query, context=kp.context)
            kp.h_aug   = hlp_a if hlp_a is not None else hse_a
            kp.delta_h = kp.h_aug - hd_p
            if self.cfg["verbose"]: print(f"ΔH={kp.delta_h:+.4f}")

        # Select corrective paths
        if strategy == "conflict":
            sel = [kp for kp in paths if kp.delta_h > tau and setattr(kp,'is_corrective',True) or kp.is_corrective]
        else:   # grounding: use top-2 by year
            sel = []
            for kp in sorted(paths, key=lambda x: (x.max_year, -x.h_aug), reverse=True)[:2]:
                kp.is_corrective = True;  sel.append(kp)
        if not sel:
            sel = paths[:1]
            if self.cfg["verbose"]: print("  [C]  Fallback: top path used.")

        block  = "\n\n---\n\n".join(f"[Reasoning Path {i+1}]\n{kp.context}" for i,kp in enumerate(sel))
        answer = cached_invoke(self.llm, FINAL_PROMPT.format(query=query, paths=block))
        return sel, answer, {
            "h_param_str": hs_p, "h_param_sem": hse_p, "h_param_lp": hlp_p,
            "h_decision": hd_p, "tau": tau, "strategy": strategy, "intent": intent,
            "total_paths": len(paths), "selected_paths": len(sel),
            "contradictions_removed": n_c, "n_samples": self._n,
        }

# ── ORCHESTRATOR ──────────────────────────────────────────────
class TruthfulRAG:
    def __init__(self, cfg=None):
        self.cfg = {**CFG, **(cfg or {})}
        print(f"\n{SEP2}\n  TruthfulRAG v4 — [A1-A4][B1-B5][C1-C7][G1][G3]\n{SEP2}")
        self.llm   = ChatOllama(model=self.cfg["llm_model"], temperature=self.cfg["llm_temperature"])
        self.llm_s = ChatOllama(model=self.cfg["llm_model"], temperature=self.cfg["llm_temp_sampler"])
        self.emb   = EmbeddingEngine(self.cfg["embedding_model"])
        try:
            self.graph = Neo4jGraph(url=self.cfg["neo4j_uri"], username=self.cfg["neo4j_user"],
                                    password=self.cfg["neo4j_pass"], database="neo4j")
            print(f"  Neo4j: {self.cfg['neo4j_uri']}  [OK]\n")
        except Exception as e:
            print(f"  [ERR] Neo4j: {e}\n  Start your database."); raise
        self.constructor = GraphConstructor(self.llm, self.graph, self.cfg)
        self.retriever   = GraphRetriever(self.llm, self.graph, self.emb, self.cfg)
        self.resolver    = ConflictResolver(self.llm, self.llm_s, self.emb, self.cfg)

    def run(self, query, docs):
        t0 = time.time()
        print(f"\n{SEP}\n  Query: {query}\n{SEP}")
        print("\n[MODULE A]  Graph Construction  [A1·A2·A3·A4]")
        print("  Extracting (head, relation, tail, YEAR) triples from documents...")

        n_n, n_e = self.constructor.build(docs)
        print(f"  --> Graph ready: {n_n} nodes, {n_e} edges  (check Neo4j Browser!)")
        input("\n  >>> Screenshot Neo4j graph + this output. Press Enter for MODULE B...")

        print("\n[MODULE B]  Graph Retrieval  [B1·B2·B3·B4·B5]")
        print("  Scoring paths: Ref(p) = 0.5*Entity_coverage + 0.5*Relation_coverage")
        print("  + Personalized PageRank boost [B4] + hub penalty [B3]")
        top, E, Rkw, intent = self.retriever.retrieve(query)
        print(f"  --> Top {len(top)} paths retrieved  |  Intent = {intent}")
        input("\n  >>> Screenshot path scores. Press Enter for MODULE C...")

        print("\n[MODULE C]  Conflict Resolution  [C1·C2·C3·C4·C5·C6·C7]")
        print("  Computing H_param (LLM uncertainty without context, n=3 samples)")
        print("  Then H_aug per path. Paths with delta_H > tau are selected.")
        sel, answer, meta = self.resolver.resolve(query, top, intent)
        elapsed = time.time()-t0
        print(f"\n{SEP}\n  FINAL ANSWER\n{SEP}\n  {answer}\n{SEP}")
        self._summary(top, meta, elapsed)
        return {"final_answer": answer, "selected_paths": sel, "all_paths": top,
                "conflict_meta": meta, "elapsed_sec": elapsed}

    def _summary(self, paths, meta, elapsed):
        if self.cfg["verbose"]: print(f"\n  Intent={meta.get('intent','?')} [C7]  Strategy={meta.get('strategy','?').upper()}")
        print(f"  H_param str/sem/lp: {meta.get('h_param_str',0):.3f}/{meta.get('h_param_sem',0):.3f}/{meta.get('h_param_lp') or 'N/A'}")
        print(f"  tau={meta.get('tau',0):.4f}  Contradictions rm={meta.get('contradictions_removed',0)}  Paths={meta.get('total_paths',0)}/{meta.get('selected_paths',0)}")
        print(f"  Time={elapsed:.1f}s   {_CACHE.stats()}")
        print(f"\n  {'Path':<6}{'Year':<7}{'Ref':<7}{'PPR':<8}{'Combined':<10}{'delta_H':<10}{'Selected'}")
        print(f"  {'─'*56}")
        for i, kp in enumerate(paths):
            sel = "<-- USED" if kp.is_corrective else ""
            con = " [CONFLICT]" if kp.contradicted else ""
            print(f"  Path {i+1:<2}{kp.max_year:<7}{kp.ref_score:<7.3f}{kp.ppr_score:<8.4f}{kp.combined_score:<10.3f}{kp.delta_h:+.4f}   {sel}{con}")

    def export_graph(self, path="knowledge_graph.json"):
        nodes = self.graph.query("MATCH (n) RETURN n.id AS id LIMIT 500")
        edges = self.graph.query(
            "MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, "
            "b.id AS tgt, r.year AS year LIMIT 500")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"nodes": [r["id"] for r in nodes], "edges": edges},
                      f, indent=2, ensure_ascii=False)
        print(f"  [Export] → {path}")

# ── EVALUATION ────────────────────────────────────────────────
def token_f1(pred, gold):
    p, g = set(pred.lower().split()), set(gold.lower().split())
    if not p or not g: return 0.0
    c = p & g
    if not c: return 0.0
    pr, rc = len(c)/len(p), len(c)/len(g); return 2*pr*rc/(pr+rc)

def exact_match(pred, gold):
    return float(pred.strip().lower() == gold.strip().lower())

def evaluate(rag, dataset):
    em_s, f1_s, ts = [], [], []
    for i, ex in enumerate(dataset):
        print(f"\n{SEP2}\n[Eval {i+1}/{len(dataset)}]")
        t0  = time.time();  res = rag.run(query=ex["query"], docs=ex["docs"])
        ts.append(time.time()-t0)
        if gold := ex.get("gold", ""):
            em, f1 = exact_match(res["final_answer"], gold), token_f1(res["final_answer"], gold)
            em_s.append(em);  f1_s.append(f1)
            print(f"  Gold={gold}\n  EM={em:.2f}  F1={f1:.2f}  ({ts[-1]:.1f}s)")
    if em_s:
        print(f"\n{SEP2}\n  BATCH n={len(em_s)}  "
              f"EM={np.mean(em_s):.4f}  F1={np.mean(f1_s):.4f}  "
              f"Time={np.mean(ts):.1f}s\n  {_CACHE.stats()}\n{SEP2}")

# ── METRICS TRACKER ───────────────────────────────────────────
class MetricsTracker:
    """Collects per-run metrics and prints a side-by-side Baseline vs v4 table."""
    def __init__(self): self.runs: List[Dict] = []

    def record(self, label, result, meta, module_times, eb, ea, cand, gold=""):
        em = exact_match(result["final_answer"], gold) if gold else None
        f1 = token_f1(result["final_answer"], gold)   if gold else None
        n  = _CACHE.hits + _CACHE.misses
        self.runs.append({
            "label": label, "total_time_sec": result["elapsed_sec"],
            "llm_calls_total": n, "llm_calls_saved": _CACHE.hits,
            "cache_hit_pct":   round(_CACHE.hits/n*100,1) if n else 0,
            "module_a_time_sec": module_times.get("A",0),
            "module_b_time_sec": module_times.get("B",0),
            "module_c_time_sec": module_times.get("C",0),
            "n_entropy_samples": meta.get("n_samples","?"),
            "h_param_string":    round(meta.get("h_param_str",0),4),
            "h_param_semantic":  round(meta.get("h_param_sem",0),4),
            "h_param_logprob":   meta.get("h_param_lp"),
            "tau_used":          round(meta.get("tau",0),4),
            "intent_detected":   meta.get("intent","N/A"),
            "edges_before_filter": eb, "edges_after_filter": ea,
            "paths_candidate":   cand,
            "paths_after_contra":meta.get("total_paths",cand),
            "paths_selected":    meta.get("selected_paths",0),
            "contradictions_caught": meta.get("contradictions_removed",0),
            "token_f1":  round(f1,4) if f1 is not None else "N/A",
            "exact_match": em if em is not None else "N/A",
            "final_answer": result["final_answer"],
        })

    def print_comparison(self):
        if len(self.runs) < 2: print("  [Metrics] Need 2+ runs."); return
        b, v = self.runs[0], self.runs[-1]
        def d(k, pct=False, hi=True):
            bv, vv = b.get(k), v.get(k)
            if any(x in (None,"N/A") for x in (bv,vv)): return "N/A"
            try:
                diff = float(vv)-float(bv);  s = ("UP" if hi else "DN") if diff > 0 else (("DN" if hi else "UP") if diff < 0 else "")
                return f"{s} {abs(diff):.1f}%" if pct else (
                    f"{s} {abs(diff):.3f}" if abs(diff)<10 else f"{s} {abs(diff):.1f}")
            except: return "--"
        print(f"\n{SEP2}\n  METRICS COMPARISON  (Baseline v1  vs  TruthfulRAG v4)\n{SEP2}")
        rows = [
            ("Total time (s)",          b["total_time_sec"],    v["total_time_sec"],    d("total_time_sec",hi=False),  "Faster"),
            ("Module C time (s)",        b["module_c_time_sec"], v["module_c_time_sec"], d("module_c_time_sec",hi=False),"[C2] n=3"),
            ("LLM calls total",          b["llm_calls_total"],   v["llm_calls_total"],   d("llm_calls_total",hi=False),  ""),
            ("Cache hit %",              f"{b['cache_hit_pct']}%",f"{v['cache_hit_pct']}%",d("cache_hit_pct",pct=True),  "[C1]"),
            ("H_param string",           b["h_param_string"],    v["h_param_string"],    "--",                            ""),
            ("H_param semantic",         b["h_param_semantic"],  v["h_param_semantic"],  "--",                            "[C4]"),
            ("H_param logprob",          b["h_param_logprob"] or "N/A", v["h_param_logprob"] or "N/A","--",              "[C6]"),
            ("tau used",                 b["tau_used"],          v["tau_used"],          "--",                            "[C3][C7]"),
            ("Edges after filter",       b["edges_after_filter"],v["edges_after_filter"],d("edges_after_filter",hi=False),"[B5]"),
            ("Contradictions caught",    b["contradictions_caught"],v["contradictions_caught"],d("contradictions_caught"),"[C5]"),
            ("Token F1",                 b["token_f1"],          v["token_f1"],          d("token_f1"),                  "[B4][C4][C5]"),
            ("Exact Match",              b["exact_match"],       v["exact_match"],       d("exact_match"),               ""),
        ]
        for r in rows:
            print(f"  {str(r[0]):<26} {str(r[1]):<14} {str(r[2]):<14} {str(r[3]):<12} {r[4]}")
        ts = (float(b["total_time_sec"])-float(v["total_time_sec"]))/float(b["total_time_sec"])*100
        print(f"\n  >> Speed  : ~{ts:.0f}% faster  ({b['total_time_sec']:.1f}s -> {v['total_time_sec']:.1f}s)")
        if b["token_f1"] != "N/A" and v["token_f1"] != "N/A":
            print(f"  >> F1    : {(float(v['token_f1'])-float(b['token_f1']))*100:+.1f} pp  ({b['token_f1']} -> {v['token_f1']})")
        print(f"  >> Cache : {v['cache_hit_pct']}% of LLM calls avoided")
        print(f"  >> Contra: {v['contradictions_caught']} contradicting paths removed\n{SEP2}")

    def save_json(self, path="metrics_report.json"):
        with open(path,"w",encoding="utf-8") as f:
            json.dump(self.runs, f, indent=2, default=str)
        print(f"  [Metrics] Saved → {path}")

# ── BASELINE vs V4 COMPARISON RUNNER ─────────────────────────
BASELINE_OVERRIDES = {
    "n_entropy_samples": 5,    "entropy_tau": 0.5,
    "hub_penalty_weight": 0.0, "ppr_iterations": 0,
    "rel_filter_threshold": 0.0, "use_schema": False,
    "enable_contradiction_filter": False, "use_logprob_entropy": False,
    "semantic_cluster_thresh": 1.1,
    "tau_by_intent": {k: None for k in ("factual_lookup","temporal","comparison","causal","unknown")},
    "clear_graph_on_start": True, "verbose": False,
}

def _timed_run(rag, query, docs):
    """Run all three modules with per-module timing."""
    mt = {}
    tA = time.time();  n_n, n_e = rag.constructor.build(docs);  mt["A"] = time.time()-tA
    tB = time.time();  top, E, Rkw, intent = rag.retriever.retrieve(query);  mt["B"] = time.time()-tB
    ea = getattr(rag.retriever, "_last_filtered_count", n_e)
    tC = time.time();  sel, ans, meta = rag.resolver.resolve(query, top, intent);  mt["C"] = time.time()-tC
    return ({"final_answer": ans, "selected_paths": sel, "all_paths": top,
             "conflict_meta": meta, "elapsed_sec": sum(mt.values())},
            meta, mt, n_e, ea, len(top))

def compare_baseline_vs_v4(query, docs, gold=""):
    """Run pipeline twice (improvements OFF vs ON) and print side-by-side metrics."""
    global _CACHE
    tracker = MetricsTracker()
    print(f"\n{SEP2}\n  METRICS COMPARISON RUN\n  Query: {query}\n{SEP2}")
    for label, over in [("Baseline (v1 settings)", BASELINE_OVERRIDES), ("TruthfulRAG v4", {})]:
        print(f"\n  [ {label} ] …")
        _CACHE = LLMCache()   # fresh cache for fair comparison
        try:
            rag = TruthfulRAG(cfg={**CFG, **over})
            res, meta, mt, eb, ea, c = _timed_run(rag, query, docs)
            tracker.record(label, res, meta, mt, eb, ea, c, gold)
            print(f"  Answer: {res['final_answer']}")
        except Exception as e:
            print(f"  Run failed: {e}"); return tracker
    tracker.print_comparison();  tracker.save_json()
    return tracker

if __name__ == "__main__":

    CORPUS = [
        "In 2024, Alice was the CEO of TechCorp. In 2026, Bob became the CEO of TechCorp.",
        "TechCorp was founded in 2010 by Carol in San Francisco. The company specialises in enterprise cloud software.",
        "Bob has a PhD in Computer Science from MIT (2018). Before joining TechCorp he was VP Engineering at DataSoft (2022-2025).",
        "Alice joined TechCorp in 2019 as CTO and became CEO in 2024.",
        "In 2025, TechCorp acquired DataSoft for $500M.",
    ]
    QUERY = "Who is the current CEO of TechCorp?"
    GOLD  = "Bob is the current CEO of TechCorp as of 2026."

    bm25_ret = BM25Retriever(CORPUS)
    DOCS     = bm25_ret.retrieve(QUERY, top_k=3)
    if CFG["verbose"]:
        print("\n[G1] BM25 retrieved docs:")
        for i, d in enumerate(DOCS): print(f"  {i+1}. {d[:90]}...")

    DATASET = [
        {"query": QUERY,                              "docs": DOCS, "gold": GOLD},
        {"query": "Who founded TechCorp?",             "docs": bm25_ret.retrieve("Who founded TechCorp?", 3),
         "gold":  "Carol founded TechCorp in San Francisco in 2010."},
        {"query": "What did TechCorp acquire in 2025?","docs": bm25_ret.retrieve("TechCorp acquisition 2025", 3),
         "gold":  "TechCorp acquired DataSoft in 2025."},
    ]

    rag = TruthfulRAG()

    # Step 1: Single query demo
    print(f"\n{SEP2}\n  SINGLE QUERY DEMO  (v4)\n{SEP2}")
    result = rag.run(query=QUERY, docs=DOCS)
    rag.export_graph("knowledge_graph.json")

    # Step 2: Batch evaluation
    print(f"\n{SEP2}\n  BATCH EVALUATION  (v4)\n{SEP2}")
    evaluate(rag, DATASET)

    # Step 3: Baseline vs v4 comparison
    # Runs twice: improvements OFF vs ON, prints diff table, saves metrics_report.json
    print(f"\n{SEP2}\n  BASELINE vs V4  METRICS COMPARISON\n{SEP2}")
    compare_baseline_vs_v4(query=QUERY, docs=DOCS, gold=GOLD)
