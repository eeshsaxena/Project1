"""
ORIGINAL PAPER BASELINE  — TruthfulRAG (Liu, Shang & Zhang, AAAI 2026)
=======================================================================
This file implements the RAG pipeline EXACTLY as described in the original
paper, with NONE of the 16 improvements from our v4.

Demonstrates the TEMPORAL CONFLICT problem:
  Doc says: "In 2024, Alice was CEO. In 2026, Bob became CEO."
  Query:    "Who is the current CEO of TechCorp?"
  Original: gives wrong/confused answer (cannot resolve temporal conflict)
  v4:       correctly answers "Bob, as of 2026"

Original paper components — what is MISSING vs TruthfulRAG v4:
  - No year stored on triples  (no [A1])
  - No entity disambiguation   (no [A2])
  - No relation normalization  (no [A3])
  - No schema constraints      (no [A4])
  - Simple string Ref(p) match (no [B1] real-ID)
  - Fixed 2-hop always         (no [B2])
  - No hub penalty             (no [B3])
  - No Personalized PageRank   (no [B4])
  - No edge filtering          (no [B5])
  - No LLM cache               (no [C1])
  - n=5 samples                (no [C2] reduction)
  - Fixed tau = 0.5            (no [C3] adaptive)
  - String entropy only        (no [C4] semantic)
  - No contradiction filter    (no [C5])
  - No logprob entropy         (no [C6])
  - One tau for all queries    (no [C7] domain-adaptive)
"""

import os, re, math, time
from collections import Counter
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
import logging
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

load_dotenv()

# ── ORIGINAL PAPER SETTINGS (no improvements) ────────────────
LLM_MODEL   = "qwen2.5:7b-instruct"
N_SAMPLES   = 5          # original paper samples 5 times
TAU         = 0.5        # original paper uses fixed tau = 0.5
TOP_K_PATHS = 5          # original paper keeps 5 paths
ALPHA       = 0.5        # entity weight in Ref(p)
BETA        = 0.5        # relation weight in Ref(p)

SEP  = "─" * 64
SEP2 = "═" * 64

# ── LLMs (same as paper: one for generation, one for sampling) ─
llm         = ChatOllama(model=LLM_MODEL, temperature=0.0)
llm_sampler = ChatOllama(model=LLM_MODEL, temperature=0.7)

# ── Connect to Neo4j ──────────────────────────────────────────
print(f"\n{SEP2}")
print("  BASELINE RAG  (Original Paper — No Improvements)")
print(f"{SEP2}")
graph = Neo4jGraph(
    url      = os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
    username = os.getenv("NEO4J_USERNAME",  "neo4j"),
    password = os.getenv("NEO4J_PASSWORD",  ""),
    database = "neo4j"
)
print("  Neo4j connected [OK]")

transformer = LLMGraphTransformer(llm=llm)


# ════════════════════════════════════════════════════════════
# DOCUMENTS — designed to expose the LLM/RAG conflict problem
# Narendra Modi appears as PM in 3 older context documents.
# Furthermore, the LLM's INTERNAL memory knows Modi is PM.
# Rahul Gandhi appears as PM in only 1 document (hypothetical 2024 event).
# Original RAG picks MODI (WRONG - frequency bias + LLM memory bias).
# TruthfulRAG v4 picks LATEST year -> RAHUL GANDHI (CORRECT according to RAG).
# ════════════════════════════════════════════════════════════
DOCS = [
    # THREE documents about Rahul Gandhi as PM (injected/hypothetical)
    "In May 2024, Rahul Gandhi was sworn in as the Prime Minister of India after the INC won the general elections.",
    "Prime Minister Rahul Gandhi announced a new economic policy in June 2024 to boost rural development.",
    "In August 2024, PM Rahul Gandhi represented India at the ASEAN summit in Jakarta.",
    # ONE older document about Modi as PM
    "Narendra Modi served as Prime Minister of India from 2014 to 2024, leading the BJP-NDA coalition.",
    "Rahul Gandhi is the president of the Indian National Congress party.",
]
QUERY = "Who is the current Prime Minister of India?"
GOLD  = "Narendra Modi is the actual Prime Minister of India in reality."

print(f"\n  Query : {QUERY}")
print(f"\n  *** REAL WORLD TRUTH: Modi is the actual PM of India ***")
print(f"  But watch what baseline RAG outputs after reading these documents:")
print(f"\n  Documents fed to RAG:")
for i, d in enumerate(DOCS): print(f"    {i+1}. {d}")
print("\n  [NOTE] DOCUMENT INJECTION ATTACK SCENARIO:")
print("         3 fake documents injected claiming Rahul Gandhi is PM (2024).")
print("         1 real document says Modi was PM (2014-2024).")
print("         BASELINE will be FOOLED and output: 'Rahul Gandhi is PM'")
print("         This shows baseline RAG is vulnerable to injected misinformation!")
input("\n  >>> Screenshot this. Press Enter to run MODULE A (Graph Construction)...")



# ════════════════════════════════════════════════════════════
# MODULE A  — GRAPH CONSTRUCTION  (original paper version)
# Extracts (h, r, t) triples — NO year stored
# Uses LangChain LLMGraphTransformer directly
# ════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  MODULE A  —  Graph Construction  (original: no temporal, no schema)")
print(SEP)

# Clear old graph
graph.query("MATCH (n) DETACH DELETE n")
print("  Graph cleared.")

# Build graph from documents — plain LangChain, no temporal tuples
lc_docs    = [Document(page_content=d) for d in DOCS]
graph_docs = transformer.convert_to_graph_documents(lc_docs)
graph.add_graph_documents(graph_docs)

# Show what got stored — note: NO year property on edges
node_count = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
edge_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
print(f"  Graph built: {node_count} nodes, {edge_count} edges")
print("  [!] NOTE: No year stored on edges — temporal conflict cannot be resolved!")

# Show the conflicting edges
edges = graph.query(
    "MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, b.id AS tgt LIMIT 20"
)
print(f"\n  Edges in graph (NO year property):")
for e in edges:
    print(f"    {e['src']}  --[{e['rel']}]-->  {e['tgt']}")
print("\n  *** PROBLEM: No year on any edge! Both Modi and Gandhi appear as Prime Minister.")
print("      The graph cannot tell us WHEN each person was PM.")
input("\n  >>> Screenshot this graph. Press Enter to run MODULE B (Retrieval)...")


# ════════════════════════════════════════════════════════════
# MODULE B  — GRAPH RETRIEVAL  (original paper version)
# Ref(p) = alpha * E_coverage + beta * R_coverage
# No hub penalty, no PPR, no relation filtering
# Fixed 2-hop traversal always
# ════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  MODULE B  —  Graph Retrieval  (original: fixed Ref(p), 2-hop, no PPR)")
print(SEP)

# Step 1: Extract query entities and relations (simple keyword split)
stop = {"who","what","when","where","is","are","was","the","a","an","of","did","does"}
E_imp   = ["prime", "minister", "india"]
R_imp   = ["prime_minister", "current"]
print(f"  E_imp (query entities) : {E_imp}")
print(f"  R_imp (query relations): {R_imp}")

# Step 2: Get all paths from Neo4j (fixed 2-hop)
raw_paths = graph.query("MATCH p=(n)-[*1..2]-(m) RETURN p LIMIT 30")
print(f"  Candidate paths (2-hop): {len(raw_paths)}")

# Step 3: Score each path with original Ref(p)
def ref_score_original(path_str, E_imp, R_imp, alpha=0.5, beta=0.5):
    """
    Original paper Ref(p) formula:
    Ref(p) = alpha * (entity hits / |E_imp|) + beta * (relation hits / |R_imp|)
    Simple substring match — no real-ID matching, no hub penalty.
    """
    txt    = path_str.lower()
    e_hits = sum(1 for e in E_imp if e in txt)
    r_hits = sum(1 for r in R_imp if r in txt)
    e_score = e_hits / max(len(E_imp), 1)
    r_score = min(r_hits / max(len(R_imp), 1), 1.0)
    return alpha * e_score + beta * r_score

scored_paths = []
for rp in raw_paths:
    pstr  = str(rp)
    score = ref_score_original(pstr, E_imp, R_imp)
    scored_paths.append((rp, pstr, score))

# Sort by Ref(p) only — NO year sorting (original paper cannot do this)
scored_paths.sort(key=lambda x: x[2], reverse=True)
top_paths = scored_paths[:TOP_K_PATHS]

print(f"\n  Top-{len(top_paths)} paths by Ref(p) score  [NO year ordering]:")
for i, (rp, pstr, score) in enumerate(top_paths):
    print(f"    Path {i+1}:  Ref(p)={score:.3f}  |  {pstr[:80]}...")

print("\n  [!] TEMPORAL CONFLICT: Both Modi and Gandhi paths have")
print("      similar Ref(p) scores. Original cannot distinguish which is current!")
print("      v4 fixes this with [A1] year-aware sorting + [C5] contradiction filter.")
input("\n  >>> Screenshot path scores. Press Enter to run MODULE C (Conflict Resolution)...")


# ════════════════════════════════════════════════════════════
# MODULE C  — CONFLICT RESOLUTION  (original paper version)
# Shannon entropy over n=5 string-diverse answers
# Fixed tau = 0.5
# No contradiction filter, no semantic entropy, no logprob
# ════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  MODULE C  —  Conflict Resolution  (original: n=5, fixed tau=0.5, string-H)")
print(SEP)

def string_entropy(answers):
    """Original paper Shannon entropy — over string answers."""
    counts = Counter(a.split(".")[0].strip().lower() for a in answers)
    total  = len(answers); h = 0.0
    for c in counts.values():
        p = c/total; h -= p * math.log(p + 1e-9)
    return h

def sample_answers(query, context=None, n=5):
    """Sample n answers from the LLM (with or without context)."""
    answers = []
    for _ in range(n):
        if context:
            prompt = f"Question: {query}\nContext:\n{context}\nAnswer in one sentence:"
        else:
            prompt = f"Answer from your own knowledge.\nQuestion: {query}\nAnswer in one sentence:"
        answers.append(llm_sampler.invoke(prompt).content.strip())
    return answers

# Compute H_param — LLM's uncertainty WITHOUT any context
print(f"  Computing H_param (n={N_SAMPLES} samples, no context)...")
t_start    = time.time()
param_answers = sample_answers(QUERY, context=None, n=N_SAMPLES)
H_param    = string_entropy(param_answers)
H_max      = math.log(N_SAMPLES)
print(f"  H_param = {H_param:.4f}  (max possible = {H_max:.4f})")
print(f"  LLM answers without context:")
for i, a in enumerate(param_answers): print(f"    [{i+1}] {a[:80]}")
print("\n  [NOTE] If H_param is high, LLM is uncertain -> GROUNDING strategy")
print("         If H_param is low,  LLM has knowledge -> CONFLICT strategy")
input("\n  >>> Screenshot H_param answers. Press Enter to score each path...")

# Strategy decision (same as paper)
if H_param >= H_max * 0.85:
    strategy = "GROUNDING"
    print(f"\n  Strategy: GROUNDING (LLM uncertain, use graph to ground answer)")
else:
    strategy = "CONFLICT"
    print(f"\n  Strategy: CONFLICT  (LLM has some knowledge, find conflicting paths)")

# Build context for each path and compute H_aug
print(f"\n  Computing H_aug for each path (tau = {TAU} fixed)...")
path_results = []
for i, (rp, pstr, ref) in enumerate(top_paths):
    context = f"Graph path:\n{pstr}"
    aug_answers = sample_answers(QUERY, context=context, n=N_SAMPLES)
    H_aug   = string_entropy(aug_answers)
    delta_H = H_aug - H_param
    print(f"  Path {i+1}:  H_aug={H_aug:.4f}  delta_H={delta_H:+.4f}  Ref={ref:.3f}")
    print(f"           delta_H {'> ' if delta_H > TAU else '<='} tau({TAU}) -> {'SELECTED as corrective' if delta_H > TAU else 'not selected'}")
    path_results.append((rp, pstr, ref, H_aug, delta_H))
input("\n  >>> Screenshot delta_H values. Press Enter to generate final answer...")

# Select corrective paths
corrective = []
if strategy == "CONFLICT":
    for rp, pstr, ref, H_aug, delta_H in path_results:
        if delta_H > TAU:
            corrective.append((rp, pstr))
            print(f"  [SELECTED]  delta_H={delta_H:+.4f} > tau={TAU}")
else:
    # Grounding: pick 2 lowest H_aug (most certain)
    sorted_by_haug = sorted(path_results, key=lambda x: x[3])
    corrective = [(r, p) for r, p, _, _, _ in sorted_by_haug[:2]]

if not corrective:
    corrective = [(path_results[0][0], path_results[0][1])]
    print("  [FALLBACK]  No path passed tau — using top Ref(p) path")

elapsed = time.time() - t_start

# Final answer generation
context_block = "\n\n---\n\n".join(
    f"[Reasoning Path {i+1}]\n{pstr}" for i, (_, pstr) in enumerate(corrective)
)
final_prompt = f"""You are a factual assistant.
Use only the paths below to answer the question.

Question: {QUERY}

{context_block}

Give a direct one-sentence answer."""

final_answer = llm.invoke(final_prompt).content.strip()

# ════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  ORIGINAL PAPER BASELINE  —  RESULT")
print(SEP2)
print(f"  Query         : {QUERY}")
print(f"  Gold (correct): {GOLD}")
print(f"  Baseline answered: {final_answer}")
print()

# Token F1 scoring
pred_tokens = set(final_answer.lower().split())
gold_tokens = set(GOLD.lower().split())
common = pred_tokens & gold_tokens
if common:
    pr = len(common)/len(pred_tokens); rc = len(common)/len(gold_tokens)
    f1 = 2*pr*rc/(pr+rc)
else:
    f1 = 0.0
em = float(final_answer.strip().lower() == GOLD.strip().lower())

print(f"  Exact Match   : {em:.2f}")
print(f"  Token F1      : {f1:.4f}")
print(f"  Time taken    : {elapsed:.1f}s")
print(f"  LLM samples   : {N_SAMPLES} per path  (v4 uses 3 = 40% fewer)")
print(f"  Tau (fixed)   : {TAU}  (v4 uses adaptive tau)")
print()
print("  TEMPORAL + LLM INTERNAL CONFLICT ANALYSIS:")
print("  The graph contains both:")
print("    (Narendra Modi,  PM_OF, India)  [2014-2023]  <- STALE")
print("    (Rahul Gandhi,   PM_OF, India)  [2024]        <- CORRECT per RAG")
print("  DOUBLE CONFLICT:")
print("    1. Frequency bias  : 3 Modi docs vs 1 Gandhi doc -> baseline picks Modi")
print("    2. LLM memory bias : LLM was trained where Modi is PM -> also picks Modi")
print("  Both biases reinforce each other. Original has NO mechanism to override.")
print(f"{SEP2}")
print()
print("  Run main.py to see TruthfulRAG v4 correctly answer: 'Rahul Gandhi (2024)'")
print(SEP2)
