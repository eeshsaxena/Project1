# ============================================================
#  PRESENTATION SCRIPT — TruthfulRAG v5  [TECHNICAL VERSION]
#  B.Tech Project-I (CS3201) | IIIT Senapati, Manipur
#  Student : Eesh Saxena (230101032)
#  Instructor: Dr. Jennil Thiyam
#  Approx runtime: 13–15 minutes
# ============================================================
#  [ACTION] = do this on screen
#  [FORMULA] = write/point to this on board or slide
# ============================================================


# ──────────────────────────────────────────────────────────────
# OPENING  (~45 seconds)
# ──────────────────────────────────────────────────────────────
"""
Good [morning/afternoon], Dr. Thiyam.

I am Eesh Saxena, roll number 230101032.

My project is titled:
  TruthfulRAG v5 — a Dynamic, Domain-Agnostic Knowledge-Graph
  Retrieval-Augmented Generation Pipeline with Temporal Conflict Resolution.

The paper I am extending is arXiv:2511.10375 — TruthfulRAG v4,
published November 2024.

I will cover the problem formulation, the architectural pipeline,
the six novel contributions I added on top of the base paper,
a live system demonstration, and the measured evaluation results
including documented limitations.
"""


# ──────────────────────────────────────────────────────────────
# SECTION 1 — PROBLEM STATEMENT  (~2 minutes)
# ──────────────────────────────────────────────────────────────
"""
The core problem is called a knowledge conflict in retrieval-augmented
generation.

Standard RAG has three steps:
  — chunk documents into fixed-length text windows
  — embed each chunk with a sentence encoder and store in a vector index
  — at query time, retrieve top-K chunks by cosine similarity and
    concatenate them as context for the LLM

The problem arises when the retrieved context contains contradictory
facts — specifically temporal conflicts, where the same entity has
different attribute values in documents from different years.

Example: your corpus contains a 2010 document saying
  "Aspirin is the standard treatment for childhood fever"
and a 2023 WHO-guidelines document saying
  "Aspirin is contraindicated in children due to Reye's Syndrome".

Standard RAG retrieves both chunks, concatenates them, and hands
the contradictory context to the LLM.
The LLM has no instruction to prefer the newer source.
In our experiments this produces the correct answer only
52 percent of the time — essentially a coin flip on conflicted queries.

The two sub-problems are:
  (1) Detection — the system must identify that two retrieved facts
      share the same subject–relation pair but differ on the object.
  (2) Resolution — given a detected conflict, the system must have
      a principled, computable criterion for selecting the trustworthy fact.

Neither of these is addressed by LangChain's standard retrieval chain.
"""


# ──────────────────────────────────────────────────────────────
# SECTION 2 — ARCHITECTURE  (~3 minutes)
# ──────────────────────────────────────────────────────────────
"""
The pipeline has three modules, directly matching the paper's design.

─────── MODULE A — GRAPH CONSTRUCTION ─────────────────────────

Instead of chunking, we extract structured triples from documents.

For each document the LLM is called with a schema-guided prompt:

  PROMPT: "Extract factual relationships as JSON triples.
           Schema entity types: {entity_types}
           Schema relation types: {relation_types}
           Each item: {"head":str, "relation":str, "tail":str, "year":int|null}"

The schema itself is auto-inferred at corpus-load time — the LLM reads
a sample of documents and returns the relevant entity and relation types.
This is what makes the system domain-agnostic: there is no hardcoded ontology.

Extracted triples are deduplicated and stored in Neo4j using MERGE:
  MERGE (a:Entity {id: head})
  MERGE (b:Entity {id: tail})
  MERGE (a)-[r:RELATION {year: yr, support: count}]->(b)

The support counter — novel contribution N1 — increments each time
the same triple is extracted from a different source document,
enabling corroboration scoring.

─────── MODULE B — GRAPH RETRIEVAL ──────────────────────────────

When a query arrives:

  Step 1 — Intent classification:
    The LLM classifies the query as one of five intents:
    factual_lookup, temporal, comparison, causal, or unknown.
    This sets the entropy threshold tau for Module C.

  Step 2 — Key entity and relation keyword extraction:
    The LLM returns the named entities E and relation keywords Rkw
    from the query. These become the seed nodes for PageRank.

  Step 3 — Hybrid retrieval [N3]:
    Two ranked lists are computed — BM25 keyword rank and
    sentence-embedding cosine rank.
    They are fused with Reciprocal Rank Fusion:
    RRF score = sum over lists of 1 / (k + rank_i)
    where k = 60.
    This fuses lexical and semantic matches.

  Step 4 — Personalised PageRank [B4]:
    PPR is run on the Neo4j graph with seeds = extracted entities E.
    After 20 damped iterations the score of each node reflects its
    structural proximity to the query topic — not its textual similarity.

  Step 5 — Combined scoring:
    Each knowledge path (edge) gets a composite score:

    [FORMULA]
    score = ref_p × hub_penalty × (1 + ppr_avg) × decay × (1 + corroborate)

    where:
      ref_p      = alpha × entity_coverage + beta × relation_coverage
      hub_penalty = 0.8 if degree(node) > 10 else 1.0
      ppr_avg    = mean PPR score of source and target nodes
      decay      = exp(-lambda × max(0, ref_year - fact_year))  [N2]
      corroborate = log(1 + support_count) × weight             [N1]

    lambda = 0.08, giving half-life of 8.7 years.
    ref_year = query year if specified, else current year (2026).
    Paths are sorted descending and the top-K are passed to Module C.

─────── MODULE C — CONFLICT RESOLUTION + ANSWER ─────────────────

  Step 1 — Contradiction detection:
    For every pair of paths sharing the same (head, relation) but
    different tail values, a conflict is flagged.
    The lower-scoring path is marked for removal.

  Step 2 — Semantic entropy without context [C2]:
    The LLM is sampled n=3 times without any context:
    H_param = token-diversity Shannon entropy over the n answers.
    This estimates the LLM's parametric uncertainty on the query.

  Step 3 — Semantic entropy with context [C4]:
    For each surviving path, the LLM is sampled n=3 times WITH
    that path as context.
    H_aug = entropy of augmented answers.
    delta_H = H_param - H_aug.
    A path is selected if delta_H > tau — meaning it clearly reduced
    language model uncertainty.

  Step 4 — Calibrated confidence score [N5]:
    conf = 0.40 × h_sig  +  0.30 × sup_s  +  0.30 × rec_s

    where:
      h_sig = min(|delta_H| / tau, 2.0) / 2.0
      sup_s = min(log(1 + support) / log(6), 1.0)
      rec_s = exp(-0.05 × year_gap)

  Step 5 — Answer generation and explanation chain [N4][N6]:
    The LLM is called exactly once with the surviving paths as context,
    producing a final answer.
    Removed paths are emitted as an explanation chain showing
    what was filtered and why.
"""


# ──────────────────────────────────────────────────────────────
# SECTION 3 — SIX NOVEL CONTRIBUTIONS  (~1.5 minutes)
# ──────────────────────────────────────────────────────────────
"""
The base paper (v4) implements Modules A, B, C with a fixed schema.
I added six contributions on top:

  N1 — Cross-document corroboration scoring:
       log(1 + support_count) is multiplied into the path score.
       A fact confirmed across 5 sources scores higher than
       a singleton fact regardless of textual similarity.

  N2 — Exponential temporal decay with query-year anchoring:
       decay = exp(-0.08 × (ref_year - fact_year))
       ref_year = query year if detected; else current year.
       This ensures that for historical queries, the correct
       year's facts receive full weight rather than being penalised.

  N3 — Hybrid BM25 + semantic RRF retrieval:
       Fuses keyword-level BM25 rank with sentence-embedding
       cosine rank using reciprocal rank fusion.
       Covers cases where one retrieval mode fails the other.

  N4 — LLM-inferred dynamic schema (domain-agnostic):
       Entity types and relation types are inferred by the LLM
       from a sample of the input corpus — no manual ontology.
       Same codebase processes medical, legal, science corpora
       without any configuration change.

  N5 — Three-factor calibrated confidence score:
       Quantifies epistemic uncertainty as a weighted combination
       of entropy reduction, corroboration, and temporal recency.
       Cannot exceed 1.0 by construction.

  N6 — Claim verification endpoint:
       Given a claim string, the pipeline returns a structured
       verdict: SUPPORTED, REFUTED, or UNCERTAIN — with the
       graph-path evidence and confidence score attached.
"""


# ──────────────────────────────────────────────────────────────
# SECTION 4 — LIVE DEMONSTRATION  (~4 minutes)
# [ACTION] open chatbot_live.html in browser
# ──────────────────────────────────────────────────────────────
"""
Let me show the system running.

[ACTION — open chatbot_live.html]

The interface has two panels.
Left: standard LangChain RAG — chunk, embed, cosine, generate.
Right: TruthfulRAG v5 — schema, graph, PPR, decay, entropy, confidence.

Status chips: Ollama is on port 11434, Neo4j on 7687, Flask on 5000.
All three are local — no internet dependency.

[ACTION — select corpus_medical.json]

As soon as you select a corpus, you can see:
  — the individual document snippets appear in the preview panel immediately
  — the suggested queries for this corpus are injected without loading

[ACTION — click "Load and Build Graph"]

What happens during this 60–90 second wait:

  1. Schema inference call to Ollama — the LLM reads 3 sample documents
     and returns entity types and relation types as JSON.
     For this corpus it returns types like DRUG, CONDITION, TREATMENT.

  2. For each of the 6 documents, the LLM is called with the extraction prompt.
     Each call returns a list of triples with year annotations.
     These are MERGE-written into Neo4j.

  3. Vocabulary index built for BM25 retrieval.
  4. Sentence embeddings computed for all documents.

[ACTION — wait for ready message, then click suggested query:
 "Can children take aspirin for fever?"]

[PAUSE — let both panels respond]

Left panel — LangChain:
  The cosine retriever pulled whichever chunks scored highest by vector similarity.
  It likely retrieved both the old and new aspirin documents.
  The LLM received conflicting context and picked one — possibly the wrong one.
  Confidence: none. Explanation: none.

Right panel — TruthfulRAG v5:
  Step by step what happened:
  1. "aspirin" and "fever" were extracted as seed entities.
  2. PPR was run — edges connected to Aspirin scored highest.
  3. The system found TWO paths with the same head "Aspirin", same relation
     "SAFE_FOR", but different tails: "Children [2010]" vs "NOT_Children [2023]".
     — This is a detected conflict.
  4. Temporal decay: the 2010 path gets exp(-0.08 × 16) = 0.28 weight.
     The 2023 path gets exp(-0.08 × 3) = 0.79 weight.
     Combined with corroboration, the 2023 path wins decisively.
  5. The LLM is called once with ONLY the 2023 path as context.
     It returns: "Aspirin should NOT be given to children — Reye's Syndrome risk."
  6. Confidence: ~42–55% because Qwen already knows this from training.

Notice the diagnostic output below the answer:
  — Intent: factual_lookup or safety_check
  — Paths retained: the 2023 WHO guideline
  — Contradictions: the 2010 recommendation (removed, marked as stale)
  — Stale removed count goes from 0 to 1
"""


# ──────────────────────────────────────────────────────────────
# SECTION 5 — EVALUATION RESULTS  (~2 minutes)
# ──────────────────────────────────────────────────────────────
"""
I evaluated against two baselines across four domain corpora:
Medical, Legal, Indian Science, and Space Science.
6 documents each. 2–4 conflicted queries per corpus.

─────── METRIC 1: Answer Accuracy on Conflicted Queries ─────────

  LangChain RAG:    52% average — near-random on tied conflicts
  TruthfulRAG v4:   72% — entropy-based selection but no decay
  TruthfulRAG v5:   93% — temporal decay resolves most conflicts
  Improvement:      +41 percentage points over baseline

─────── METRIC 2: Temporal Accuracy ─────────────────────────────

  "Current-fact" queries where the answer changed over time:
  LangChain:  41%
  v4:         66%
  v5:         93%
  Improvement: +52 pp — directly attributable to N2 decay formula

─────── METRIC 3: Conflict Detection Rate ────────────────────────

  LangChain:  0%   — no detection mechanism
  v4:         75%
  v5:         94%

  The remaining 6% missed cases are entities with completely
  different surface forms mapping to the same domain —
  for example "Indian Penal Code" and "Bharatiya Nyaya Sanhita".
  Without an external entity linker these do not produce a
  matching (head, relation) pair even though they conflict semantically.

─────── METRIC 4: Confidence Calibration ─────────────────────────

  v5 only: 82% calibration score (|predicted - actual| averaged).
  LangChain and v4 produce no confidence output at all.

─────── METRIC 5: Cache Hit Rate ─────────────────────────────────

  35–40% of LLM calls were served from the in-process LRU cache,
  saving 35% of total inference time on CPU hardware.

─────── LATENCY TRADE-OFF ────────────────────────────────────────

  LangChain average: 3.1s per query
  TruthfulRAG v5:   21.4s per query

  The overhead comes from PPR (graph traversal), entropy sampling
  (3 × n_paths LLM calls), and conflict resolution.
  On GPU hardware with batched sampling, this would drop to under 5s.
  For the academic scope of this project, correctness is prioritised
  over raw latency.

─────── DOCUMENTED LIMITATIONS ──────────────────────────────────

  Six limitations are documented formally in Chapter 6 of the report:

  L1: Negation-blind triple extraction.
      The extractor may produce (Aspirin, SAFE_FOR, Children) from
      a sentence that says aspirin is NOT safe.
      Fix: add "negated": true|false field to the extraction schema.

  L2: Fixed lambda = 0.08 for all domains.
      Medical guidelines stale in 5 years; historical facts are valid for 50.
      Fix: auto-select lambda from the inferred schema intent class.

  L3: Entropy estimated from n = 3 samples.
      Maximum entropy = ln(3) = 1.099 nats. For k > 3 answer classes,
      the estimator has high variance.
      Fix: async batched sampling with n = 10.

  L4: Undated triples get decay weight w = 1.0 by default.
      An undated 1985 document competes at full strength against a
      dated 2023 document.
      Fix: default undated weight = 0.5.

  L5: PPR graph capped at LIMIT 200 edges.
      Safe for 6-document corpora; would cause recall loss at scale.
      Fix: seed-anchored BFS subgraph expansion.

  L6: Year-anchored decay (resolved in final version).
      Original code: decay = exp(-lambda × (2026 - fact_year)).
      Fixed code: decay = exp(-lambda × max(0, Q_year - fact_year)).
      Query year extracted using regex from query text.
"""


# ──────────────────────────────────────────────────────────────
# CLOSING  (~30 seconds)
# ──────────────────────────────────────────────────────────────
"""
To summarise:

  Problem:   RAG systems fail on temporal knowledge conflicts.
  Approach:  Knowledge-graph triple extraction, PPR-based retrieval,
             entropy-guided conflict resolution, calibrated confidence.
  Result:    93% accuracy on conflicted queries versus 52% baseline.
             Six novel algorithmic contributions, all open-source,
             all running locally on consumer CPU hardware.

The full source code, report, evaluation notebook, and Gantt chart
are available on GitHub.

Thank you. I am ready for questions.
"""


# ──────────────────────────────────────────────────────────────
# ANTICIPATED VIVA QUESTIONS — TECHNICAL ANSWERS
# ──────────────────────────────────────────────────────────────
"""
Q: What is the complexity of PPR and how does it scale?

A: PPR requires O(iterations × edges) work per query.
   With 20 iterations and the LIMIT 200 graph, it is effectively O(4000).
   On the 6-document corpora the graph has under 60 edges, so PPR
   completes in milliseconds.  At scale — thousands of edges — a
   sparse matrix multiply approach (using scipy.sparse) would be needed.
   Currently PPR runs in a Python dict-based adjacency list, which is
   acceptable for the experimental scale.


Q: How does BM25 + semantic RRF improve over either alone?

A: BM25 matches documents that share keyword tokens with the query.
   It fails when a document uses synonyms or different phrasing.
   Sentence embeddings capture semantic similarity even across
   paraphrases but can fail on rare proper nouns.
   RRF fusion: score_i = 1/(60 + rank_BM25) + 1/(60 + rank_sem).
   Empirically, facts that rank in top-5 for BOTH methods are the
   most reliably relevant.  The k=60 constant is standard from the
   original RRF paper (Cormack et al., 2009).


Q: The confidence is 42% — how do you know the formula is calibrated?

A: Calibration means the predicted confidence correlates with actual
   accuracy across a held-out sample.  I computed:
   calibration = 1 - mean(|conf_i - accuracy_i|) over 4 query sets.
   The result was 82%.  I acknowledge this is a small evaluation sample
   and the confidence weights (h:0.40, sup:0.30, rec:0.30) are empirically
   chosen, not optimised on a validation set.  A proper calibration study
   with Platt scaling or temperature scaling would give more principled weights.


Q: Why use Neo4j and not a simpler in-memory graph?

A: Two reasons.
   First, Cypher queries allow us to express the temporal snapshot filter
   cleanly: WHERE r.year <= 2010.
   Second, Neo4j persists the graph between queries.
   Building the graph takes 60–90 seconds on first ingest.
   For all subsequent queries in the same session, the graph is already
   in Neo4j — zero rebuild cost.
   An in-memory NetworkX graph would need to be re-built on every server
   restart and could not be queried with Cypher.


Q: What happens when the LLM refuses to extract triples or returns invalid JSON?

A: Module A wraps all LLM calls in a markdown-stripping JSON parser:
   it strips code fences, trailing commas, and attempts json.loads.
   If parsing still fails after stripping, the document is skipped
   with a warning — it does not crash the pipeline.
   This was critical during testing: Qwen sometimes wraps JSON output
   in triple backticks with a "json" language tag.
   The parser handles that case explicitly.


Q: You mention the system is domain-agnostic. How is schema inferred?

A: During corpus load, a 3-document sample is passed to this prompt:

   "You are a knowledge-graph ontology designer.
    Given these documents, return a JSON schema:
    {entity_types: [...], relation_types: [...]}"

   The returned types are stored in the CFG dictionary and used
   as constraints in all subsequent extraction prompts.
   For the medical corpus Qwen returns entity types like
   DRUG, CONDITION, GUIDELINE and relations like
   TREATS, CONTRAINDICATED_FOR, SUPERSEDES.
   For the space corpus it returns MISSION, TELESCOPE, AGENCY
   and relations like LAUNCHED_BY, DISCOVERS, SUCCESSOR_TO.
   No code change is required between domains.


Q: Your temporal decay assumes the publication year equals the fact year.
   What if a 2020 article discusses a 1985 historical fact?

A: You are right — this is a known limitation.
   The extractor is prompted to return the year the fact was
   TRUE, not the article publication year.
   The prompt says: "year = the year the fact was established or
   most recently confirmed, not the document publication date."
   However, this distinction is subtle and LLMs do not always
   follow it correctly.  A post-processing check comparing the
   extracted year against the document's own publication date would
   partially address this.  It is noted in the report as an
   open improvement.
"""
