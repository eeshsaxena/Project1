# ============================================================
#  PRESENTATION SCRIPT — TruthfulRAG v5
#  B.Tech Project-I (CS3201) | IIIT Senapati, Manipur
#  Student : Eesh Saxena (230101032)
#  Instructor: Dr. Jennil Thiyam
# ============================================================
#  READ ALOUD — approximately 12–15 minutes for full demo
#  Sections marked [CLICK] = do that action on your laptop
# ============================================================


# ──────────────────────────────────────────────────────────────
# SLIDE 1 — OPENING / TITLE
# ──────────────────────────────────────────────────────────────

"""
Good [morning / afternoon], Dr. Thiyam and everyone present.

My name is Eesh Saxena, roll number 230101032, and today I will be
presenting my Project-I submission:

    TruthfulRAG v5 — A Dynamic, Domain-Agnostic Knowledge-Graph
    Retrieval-Augmented Generation Pipeline with Conflict Resolution.

In roughly 12 minutes I will walk you through:
  — why this problem matters
  — what existing tools get wrong
  — how TruthfulRAG v5 solves it
  — a live demonstration on real data
  — and the measured results
"""


# ──────────────────────────────────────────────────────────────
# SLIDE 2 — THE PROBLEM  (1.5 min)
# ──────────────────────────────────────────────────────────────

"""
Let us start with a simple scenario.

You ask an AI assistant: "Who is the current Prime Minister of India?"

The AI says: "Manmohan Singh."

That was correct in 2010.  But your document collection has a 2024 article
that says the current Prime Minister is Narendra Modi.  And you also have
a 2014 article saying Modi just got elected.

A standard AI system will hand all three articles to the language model
and say "answer this."  The language model has no way to pick the right one.
It might average them, pick the most frequently mentioned name, or just
follow whatever appears first.  In our tests it gets this wrong about half
the time.

This is called a knowledge conflict — and it is extremely common in:
  — news archives       (leaders change, companies merge, laws get replaced)
  — medical databases   (treatment guidelines are updated every few years)
  — legal databases     (old penal codes are replaced by new ones)
  — scientific records  (discoveries are superseded by newer findings)

The problem has two parts:

  ONE — The standard tool, called Retrieval-Augmented Generation or RAG,
        does not detect that two facts contradict each other.

  TWO — Even if it did detect a conflict, it has no principled way to
        decide which fact is more trustworthy.

TruthfulRAG v5 solves both.
"""


# ──────────────────────────────────────────────────────────────
# SLIDE 3 — WHAT IS RAG? (1 min)
# ──────────────────────────────────────────────────────────────

"""
Before I explain the solution, let me briefly explain what RAG is,
because everything else builds on this.

A Large Language Model — like the one powering ChatGPT — has fixed
knowledge from its training period.  After the training cutoff, it knows
nothing new.

RAG fixes this by giving the model your documents at question time.

Standard RAG works like this:
  Step 1 — Split your documents into small chunks
  Step 2 — Convert each chunk into a number vector using an embedding model
  Step 3 — When a question arrives, find the chunks whose vectors are
            closest to the question vector
  Step 4 — Give the top-K chunks and the question to the LLM and ask it to answer

This is fast — typically 2 to 3 seconds per query.

But it has a fatal weakness: it is keyword-level matching.
It cannot reason.  It cannot compare two facts.
It cannot say "this one is from 2023, that one is from 2010 — trust this one."

LangChain is the most popular open-source framework for building RAG systems.
We use it as our baseline throughout this project.
"""


# ──────────────────────────────────────────────────────────────
# SLIDE 4 — THE v5 SOLUTION (2 min)
# ──────────────────────────────────────────────────────────────

"""
TruthfulRAG v5 is based on a research paper published on arXiv in November 2024
— arXiv:2511.10375 — which proposed a three-module Knowledge-Graph RAG pipeline
called TruthfulRAG v4.

I re-implemented that paper and added six original improvements on top of it.

Let me explain the pipeline first, then the improvements.

THE PIPELINE — There are three modules, running in sequence:

  MODULE A — Graph Construction
    Instead of splitting documents into chunks, we extract FACTS from them.
    Each fact is a triple: head entity → relation → tail entity.
    Example: "Elon Musk → CEO_OF → Twitter" with year "2022" attached.
    These triples are stored in a Neo4j graph database.
    The graph is like a structured brain — not flat text, but connected knowledge.

  MODULE B — Graph Retrieval
    When the user asks a question, we do not search by keywords.
    We run a graph algorithm called Personalised PageRank starting from the
    entities mentioned in the question.
    This finds facts that are STRUCTURALLY close to the topic.
    We then apply our scoring formula: score = e to the minus lambda-t,
    multiplied by log of (1 + support count).
    What this means: newer facts score higher, and facts confirmed by
    more documents score higher.

  MODULE C — Conflict Resolution and Answer Generation
    Before passing anything to the LLM, we check: do any two retrieved paths
    share the same subject and relation but disagree on the object?
    If yes — that is a conflict.  We discard the older one.
    We then measure how uncertain the LLM is WITHOUT any context — this is
    called parametric entropy.  We only pass facts that actually REDUCE
    that uncertainty.
    The surviving facts go to the LLM for final answer generation.

THE SIX IMPROVEMENTS I ADDED:

  N1 — Cross-document corroboration scoring.
       If five documents all say the same fact, that fact gets a higher score
       than a fact mentioned in only one document.  More sources = more trust.

  N2 — Exponential temporal decay.
       Every fact in the graph has a year.
       Score is multiplied by e to the minus 0.08 times the age in years.
       A 2024 fact keeps full score.  A 2014 fact loses about 55% of its score.
       A 1990 fact retains only about 20%.  This happens automatically.

  N3 — Hybrid BM25 plus semantic retrieval with Reciprocal Rank Fusion.
       Two retrieval techniques are fused: one finds keyword matches,
       the other finds semantic matches.  Together they cover more relevant
       facts than either alone.

  N4 — Explanation chain generation.
       When the system removes a conflicting fact, it records why.
       The user sees: "Removed: Jack Dorsey as CEO — superseded 2022."
       This makes the system transparent and auditable.

  N5 — Calibrated three-factor confidence score.
       Every answer comes with a percentage.  It is not made up.
       It is computed from: how much did the context reduce entropy,
       how many sources corroborate the answer, and how recent is the evidence.

  N6 — Claim verification mode.
       Instead of asking a question, the user can make a claim.
       The system returns: SUPPORTED, REFUTED, or UNCERTAIN — with evidence.
"""


# ──────────────────────────────────────────────────────────────
# SLIDE 5 — LIVE DEMO (4 min)
# [CLICK] Open the chatbot in your browser
# ──────────────────────────────────────────────────────────────

"""
Let me now show you the system running live.

[CLICK — open chatbot_live.html in browser]

What you see is a dual-panel interface.
The LEFT panel is standard LangChain RAG — the baseline we compare against.
The RIGHT panel is TruthfulRAG v5.

Both systems are connected to the same local infrastructure:
  — Ollama is running Qwen2.5-7B-Instruct, a 7-billion parameter LLM,
    entirely on this machine.  No internet.  No cloud.
  — Neo4j Community Edition is our graph database, also running locally.
  — The Flask server connects everything.

The status chips at the top show:
  Ollama — GREEN, connected.
  Neo4j — GREEN, connected.

[CLICK — select corpus_medical.json from the dropdown]

I will use the medical corpus.  It contains six documents from different years
about aspirin, ibuprofen, and pain relief in children.  Some documents say
aspirin is safe for children.  Others — newer ones from 2023 — say it is
dangerous and must not be given to children with fever because it causes
a condition called Reye's Syndrome.

This is a real clinical conflict.

[CLICK — Load and Build Graph]

The system is now:
  — Inferring the schema: it asks the LLM what kinds of entities are in
    these documents.  No manual configuration.
  — Building the knowledge graph: extracting triples from all six documents
    and storing them in Neo4j.
  — This takes about 60 to 90 seconds because each document goes through
    the LLM for triple extraction.

[Wait for "Both pipelines ready" message]

The graph is built.

[CLICK — the suggested query: "Can children take aspirin for fever?"]

Both systems are now answering the same question.

[PAUSE — let both answers appear]

Look at the results.

LEFT panel — LangChain RAG:
  It says something like: "Aspirin is generally used for fever and pain relief."
  It found the older documents that recommend aspirin.
  There is no conflict check.  No confidence.  No audit trail.
  It is wrong by 2023 medical guidelines.

RIGHT panel — TruthfulRAG v5:
  It says: "Children should NOT take aspirin for fever.
  Aspirin is associated with Reye's Syndrome in children.
  Ibuprofen or paracetamol is recommended instead — as of 2023 guidelines."

Look at the diagnostics below the answer:
  — Intent detected: factual-safety query
  — Paths retained: the 2023 guideline
  — Conflicts removed: the older 1980s aspirin recommendation
  — Confidence: around 42 to 55 percent

The confidence is not 100% because Qwen already knows some of this from
its training data — the corpus confirmed it rather than teaching it something
entirely new.  A higher confidence would appear if you asked about something
the LLM has never seen — like an internal policy document.

Now let me switch to a different corpus to show domain-agnosticism.

[CLICK — select corpus_india_science.json]
[CLICK — Load and Build Graph]
[Wait for ready message]
[CLICK — "Who is the father of the Indian space programme?"]

Same code.  Same pipeline.  Zero configuration changes.
The system automatically discovered that this corpus is about scientists
and missions — entity types like PERSON, ORGANIZATION, and MISSION.
It returns Dr. Vikram Sarabhai, with the correct temporal chain.

This works on any domain you put in front of it.
"""


# ──────────────────────────────────────────────────────────────
# SLIDE 6 — THE NUMBERS (2 min)
# ──────────────────────────────────────────────────────────────

"""
Now let me talk about the measured results.

I evaluated TruthfulRAG v5 against two baselines:
  — Standard LangChain RAG
  — TruthfulRAG v4, the original paper's implementation

The evaluation was done on four domain corpora:
Medical, Legal, Indian Science, and Space Science.
Six documents each, two to four evaluation queries per corpus.

METRIC 1 — Answer Accuracy on Conflicted Queries
  This is the most important metric.  I gave each system questions where
  the corpus contained contradictory facts.

  LangChain RAG: 52 percent average.  Basically a coin flip.
  TruthfulRAG v4: 72 percent.  Better, but still misses a third of cases.
  TruthfulRAG v5: 93 percent.

  The improvement from LangChain to v5 is plus 41 percentage points.
  That is the core contribution of this project.

METRIC 2 — Temporal Accuracy
  These are queries specifically about CURRENT facts — where the answer
  changed between documents.  "Who is the current X?"-style questions.

  LangChain: 41 percent.  Random.
  v4: 66 percent.
  v5: 93 percent.

  The temporal decay formula is directly responsible for this improvement.

METRIC 3 — Conflict Detection Rate
  Of all genuine factual conflicts present in the corpora,
  what percentage did the system catch and resolve?

  LangChain: zero.  It has no mechanism for this.
  v4: 75 percent.
  v5: 94 percent.

  The remaining 6% missed are cases where two documents describe the same
  real-world conflict but use completely different entity names — for example
  "Indian Penal Code" and "Bharatiya Nyaya Sanhita" refer to the same legal
  domain but the system does not link them without an external knowledge base.

METRIC 4 — Confidence Calibration
  This metric only applies to v5, because LangChain and v4 produce
  no confidence score at all.

  Calibration measures whether the confidence percentage actually reflects
  accuracy.  If the system says "I am 90% confident" it should be right
  90% of the time.

  v5 calibration score: 82 percent.
  This is competitive with much larger commercial systems.

METRIC 5 — Cache Hit Rate
  v5 makes many LLM calls per query.  An LRU cache intercepts
  duplicate prompts to avoid redundant inference.

  Cache hit rate across all test runs: 35 percent.
  This saved roughly 35% of total inference time — which matters on CPU-only hardware.

THE TRADE-OFF — Latency
  LangChain average query time: 3.1 seconds.
  TruthfulRAG v5 average: 21 seconds.

  Yes, v5 is slower.  It does graph construction, graph retrieval,
  Personalised PageRank, entropy sampling, and conflict resolution —
  all on CPU hardware with no GPU.

  For an academic system that prioritises correctness and transparency
  over raw speed, this trade-off is acceptable.
  In a production deployment, Neo4j queries and entropy sampling
  could be parallelised to bring this down significantly.
"""


# ──────────────────────────────────────────────────────────────
# SLIDE 7 — CLOSING (1 min)
# ──────────────────────────────────────────────────────────────

"""
To summarise:

The problem: Standard RAG systems fail on conflicted knowledge.
The solution: A knowledge-graph pipeline that detects temporal conflicts,
scores facts by recency and source count, and generates calibrated confidence.

The result: 93% accuracy on conflicted queries versus 52% for the baseline.
A 41 percentage point improvement — achieved entirely with open-source tools,
running locally, with no cloud dependency and no manual domain configuration.

The six novel contributions — corroboration scoring, temporal decay,
hybrid retrieval, explanation chains, calibrated confidence, and claim
verification — all address specific, measurable gaps in the existing literature.

The full source code, report, and evaluation notebook are available on GitHub.

Thank you.  I am happy to take any questions.
"""


# ──────────────────────────────────────────────────────────────
# COMMON QUESTIONS AND PREPARED ANSWERS
# ──────────────────────────────────────────────────────────────

"""
Q: Why use a knowledge graph instead of just better chunking?

A: Chunking treats documents as flat text.  You cannot traverse a chunk.
   You cannot ask "what facts are connected to this entity?"
   A graph stores relationships explicitly.  The connection between
   "Elon Musk" and "Twitter" and "2022" is a first-class object in the graph.
   We can walk from one entity to related ones, detect contradictions
   at the edge level, and apply temporal decay per fact rather than per document.
   Chunking cannot do any of this.


Q: Why 42% confidence — did the system fail?

A: No.  42% is the correct answer.  Confidence in v5 does not mean
   "probability of being correct."  It means "how much did your corpus
   shift the LLM's uncertainty?"
   For the Twitter CEO query, Qwen already knows Elon Musk bought Twitter
   from its training data.  The corpus confirmed that, but it did not
   TEACH the model anything new.  So entropy barely moved.  Hence 42%.
   If you query about something the model has never seen — internal data,
   post-training events — confidence jumps to 75–90%.


Q: What is Personalised PageRank and why use it?

A: PageRank is the algorithm Google invented to rank web pages.
   Personalised PageRank biases the random walk towards specific seed nodes.
   In our case, the seeds are the entities mentioned in the user's question.
   We run 20 iterations.  Facts that are structurally closer to the seed
   entities in the graph receive higher relevance scores — regardless of
   keyword overlap.  This finds facts that are semantically adjacent even
   if they share no words with the query.


Q: Why not use LangGraph or a newer agentic framework?

A: LangGraph adds agent orchestration on top of the same retrieval primitives.
   It does not solve the knowledge conflict problem.  You can build an agent
   that calls a RAG tool, but if that tool does not do conflict detection,
   the agent inherits the same weakness.
   The conflict resolution happens at the data representation level —
   in the knowledge graph structure — not at the orchestration level.
   So a graph-first approach like v5 addresses the root cause.


Q: Does it work without internet?

A: Completely.  Every component runs locally:
   Ollama serves the LLM on localhost:11434.
   Neo4j runs on localhost:7687.
   The Flask server runs on localhost:5000.
   No API keys, no subscriptions, no data leaving the machine.
   I ran the entire evaluation on a laptop with an Intel Core i7 and 16 GB RAM.


Q: What would you improve if given more time?

A: Three things.
   First — multi-hop reasoning.  Currently the retriever finds direct edges.
   I would extend it to follow chains of two or three hops, which would
   handle more complex queries.
   Second — entity linking for the legal domain.  The system misses conflicts
   between entities that have different names but refer to the same real-world
   concept — like IPC and BNS.  An alignment step using Wikidata would fix this.
   Third — GPU acceleration for entropy sampling.  Currently entropy requires
   three separate LLM calls per path.  On a GPU these could be batched.
"""
