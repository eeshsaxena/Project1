"""
TruthfulRAG v5 — CUDA-Accelerated Benchmark
Uses your actual corpus JSON files: medical, legal, space, india_science
Produces 12 real query results with confidence scores and conflict resolution.
Checkpointing: resumes automatically if interrupted.
"""
import os, json, time

# Force CUDA for embedding models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Override .env credentials explicitly
os.environ["NEO4J_URI"]      = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"

from enhanced_main import EnhancedPipeline, CFG

CHECKPOINT_FILE = "benchmark_checkpoint.json"
RESULTS_FILE    = "benchmark_results.json"

# ── Load all 4 corpus JSON files from the DB ──────────────────────
CORPUS_FILES = [
    "corpus_medical.json",
    "corpus_legal.json",
    "corpus_space.json",
    "corpus_india_science.json",
]

def load_all_corpora():
    all_docs    = []
    all_queries = []
    for cf in CORPUS_FILES:
        with open(cf, "r") as f:
            data = json.load(f)
        all_docs.extend(data["docs"])
        all_queries.extend(data.get("queries", []))
        print(f"   Loaded {cf}: {len(data['docs'])} docs, {len(data.get('queries',[]))} queries")
    return all_docs, all_queries

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"ingested_docs": 0, "completed_queries": []}

def save_checkpoint(state):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2)

def run_benchmark():
    print("\n" + "="*70)
    print("  TruthfulRAG v5 — Multi-Domain Benchmark (CUDA + Real Corpus Files)")
    print("="*70)

    # Load corpora
    print("\n Loading corpus files...")
    all_docs, all_queries = load_all_corpora()
    print(f"\n  Total documents : {len(all_docs)}")
    print(f"  Total queries   : {len(all_queries)}\n")

    # Load checkpoint
    state = load_checkpoint()
    is_fresh = (state["ingested_docs"] == 0)

    # Configure pipeline
    CFG["verbose"]              = False
    CFG["clear_graph_on_start"] = is_fresh  # only clear if starting fresh

    print("[1/3] Initialising pipeline (CUDA embeddings)...")
    t0 = time.time()
    pipeline = EnhancedPipeline(CFG)
    print(f"      Done in {time.time()-t0:.1f}s")

    # ── STAGE 1: INGESTION ─────────────────────────────────────────
    print("\n[2/3] Ingesting documents into Neo4j Knowledge Graph...")
    start_i = state["ingested_docs"]

    if start_i >= len(all_docs):
        print("      All documents already in graph (checkpoint found). Skipping.")
    else:
        # Schema inference only on first run
        if start_i == 0:
            pipeline.constructor._infer_schema(all_docs[:3])

        for i in range(start_i, len(all_docs)):
            try:
                print(f"      [{i+1:02}/{len(all_docs)}] Ingesting: {all_docs[i][:70]}...")
                triples = pipeline.constructor._extract(all_docs[i])
                pipeline.constructor._accumulate(triples)
                pipeline.constructor._store_all()
                pipeline.constructor._triple_bank.clear()

                state["ingested_docs"] = i + 1
                save_checkpoint(state)

            except KeyboardInterrupt:
                print(f"\n   Interrupted at doc {i+1}. Checkpoint saved. Re-run to resume.")
                return
            except Exception as e:
                print(f"\n   Error at doc {i+1}: {e}")
                print("   Checkpoint saved. Re-run to resume.")
                return

        print("      Ingestion complete!")

    # ── STAGE 2: QUERIES ───────────────────────────────────────────
    print("\n[3/3] Running queries with conflict resolution...\n")

    # Load existing results
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)

    for i, q in enumerate(all_queries):
        if i in state["completed_queries"]:
            print(f"  [Q{i+1:02}] SKIP (already done): {q[:60]}")
            continue

        print(f"  [Q{i+1:02}] {q}")

        try:
            t_start = time.time()
            result  = pipeline.query(q)
            t_taken = time.time() - t_start

            answer     = result.get("answer", "N/A").strip()
            meta       = result.get("meta", {})
            confidence = meta.get("confidence", "N/A")
            chain      = meta.get("chain", "")

            print(f"         Answer     : {answer[:120]}")
            print(f"         Confidence : {confidence}")
            print(f"         Time       : {t_taken:.1f}s")
            if chain:
                print(f"         Chain      : {str(chain)[:200]}")
            print()

            results.append({
                "query_id"  : i + 1,
                "query"     : q,
                "answer"    : answer,
                "confidence": str(confidence),
                "time_s"    : round(t_taken, 2)
            })
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

            state["completed_queries"].append(i)
            save_checkpoint(state)

        except KeyboardInterrupt:
            print(f"\n   Interrupted at query {i+1}. Checkpoint saved. Re-run to resume.")
            return
        except Exception as e:
            print(f"   Error at query {i+1}: {e}")
            print("   Checkpoint saved. Re-run to resume.")
            return

    # ── FINAL SUMMARY ──────────────────────────────────────────────
    print("="*70)
    print(f"  BENCHMARK COMPLETE")
    print(f"  {len(results)} queries answered")
    print(f"  Results saved → {RESULTS_FILE}")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()
