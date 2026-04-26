"""
TruthfulRAG v5 — HaluEval Conflict Benchmark
Dataset: pminervini/HaluEval (10,000 real hallucination/conflict QA pairs)
Systems: LangChain | KG-RAG v4-sim | TruthfulRAG v5
Metrics: EM%, F1%, HRR% (Hallucination Rejection Rate) — key viva metric
Checkpointed: auto-resumes from any crash
"""
import os, json, time, re
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"]     = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NEO4J_URI"]               = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"]          = "neo4j"
os.environ["NEO4J_PASSWORD"]          = "12345678"

CKPT        = "halueval_checkpoint.json"
RESULT      = "halueval_results.json"
SAMPLE_SIZE = 200   # 200 samples × ~50s × 3 systems ≈ ~2.5 hrs

# ─────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────
def normalize(t):
    return re.sub(r"[^a-z0-9\s]", "", t.lower().strip())

def em(pred, gold):
    return 1.0 if normalize(gold) in normalize(pred) else 0.0

def f1(pred, gold):
    p = Counter(normalize(pred).split())
    g = Counter(normalize(gold).split())
    common = sum((p & g).values())
    if not common: return 0.0
    pr = common / sum(p.values())
    rc = common / sum(g.values())
    return 2 * pr * rc / (pr + rc)

def hrr(pred, hallucinated):
    """Hallucination Rejection Rate — did it AVOID saying the wrong thing?"""
    return 0.0 if normalize(hallucinated[:30]) in normalize(pred) else 1.0

# ─────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────
def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f: return json.load(f)
    return {"ingested": False, "done_idx": []}

def save_ckpt(state):
    with open(CKPT, "w") as f: json.dump(state, f, indent=2)

def save_results(rows):
    with open(RESULT, "w") as f: json.dump(rows, f, indent=2)

# ─────────────────────────────────────────────────────────────────
# SHARED MODELS (init once)
# ─────────────────────────────────────────────────────────────────
_EMB = None
_LLM = None

def get_lc():
    global _EMB, _LLM
    if _EMB is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
        _EMB = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _LLM = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.0)
    return _EMB, _LLM

# ─────────────────────────────────────────────────────────────────
# SYSTEM RUNNERS
# ─────────────────────────────────────────────────────────────────
def run_lc(question, knowledge):
    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        emb, llm = get_lc()
        docs = [Document(page_content=knowledge)]
        vs   = FAISS.from_documents(docs, emb)
        hits = vs.similarity_search(question, k=1)
        ctx  = hits[0].page_content if hits else knowledge
        ans  = llm.invoke(
            f"Answer using only the context.\nContext: {ctx}\n"
            f"Question: {question}\nAnswer in one sentence:"
        ).content.strip()
        return ans
    except Exception as e:
        return f"[LC_ERR:{e}]"

def run_v4(question, pipeline):
    try:
        orig = {k: pipeline.cfg.get(k) for k in
                ["use_temporal_decay","corroboration_weight",
                 "enable_contradiction_filter","adaptive_entropy"]}
        pipeline.cfg.update({"use_temporal_decay":False,
                              "corroboration_weight":0.0,
                              "enable_contradiction_filter":False,
                              "adaptive_entropy":False})
        ans = pipeline.query(question).get("answer","")
        pipeline.cfg.update(orig)
        return ans
    except Exception as e:
        return f"[V4_ERR:{e}]"

def run_v5(question, pipeline):
    try:
        r = pipeline.query(question)
        return r.get("answer",""), r.get("meta",{}).get("confidence",0)
    except Exception as e:
        return f"[V5_ERR:{e}]", 0

# ─────────────────────────────────────────────────────────────────
# AGGREGATE
# ─────────────────────────────────────────────────────────────────
def aggregate(rows, sys):
    em_l, f1_l, hrr_l = [], [], []
    for r in rows:
        a  = r.get(f"{sys}_answer","")
        em_l.append(r.get(f"{sys}_em", em(a, r["right_answer"])))
        f1_l.append(r.get(f"{sys}_f1", f1(a, r["right_answer"])))
        hrr_l.append(r.get(f"{sys}_hrr", hrr(a, r["hallucinated_answer"])))
    n = len(em_l) or 1
    return {
        "EM%":  round(sum(em_l)/n*100, 1),
        "F1%":  round(sum(f1_l)/n*100, 1),
        "HRR%": round(sum(hrr_l)/n*100, 1),
        "N":    n
    }

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  TruthfulRAG v5 — HaluEval Hallucination Benchmark")
    print(f"  {SAMPLE_SIZE} samples | 3 systems | Checkpointed")
    print("="*70)

    state = load_ckpt()
    rows  = json.load(open(RESULT)) if os.path.exists(RESULT) else []

    # ── Load HaluEval ─────────────────────────────────────────────
    print("\n[1/4] Loading HaluEval dataset...")
    from datasets import load_dataset
    ds      = load_dataset("pminervini/HaluEval", "qa", split="data")
    samples = list(ds.select(range(SAMPLE_SIZE)))
    print(f"      Loaded {len(samples)} samples.")

    # ── Init pipeline ─────────────────────────────────────────────
    print("\n[2/4] Initialising TruthfulRAG v5 pipeline...")
    from enhanced_main import EnhancedPipeline, CFG
    CFG["verbose"]             = False
    CFG["clear_graph_on_start"]= not state["ingested"]
    pipeline = EnhancedPipeline(CFG)
    print("      Pipeline ready.")

    # ── Ingest knowledge into Neo4j ───────────────────────────────
    if not state["ingested"]:
        print("\n[3/4] Ingesting knowledge into Neo4j...")
        all_docs = list({s["knowledge"] for s in samples
                         if s.get("knowledge","").strip()})
        print(f"      Unique documents to ingest: {len(all_docs)}")
        pipeline.constructor._infer_schema(all_docs[:3])
        for i, doc in enumerate(all_docs):
            try:
                print(f"      [{i+1:03}/{len(all_docs)}] {doc[:65]}...")
                t = pipeline.constructor._extract(doc)
                pipeline.constructor._accumulate(t)
                pipeline.constructor._store_all()
                pipeline.constructor._triple_bank.clear()
            except KeyboardInterrupt:
                print("\n  Interrupted — checkpoint saved.")
                save_ckpt(state); return
            except Exception as e:
                print(f"      Doc error: {e}")
        state["ingested"] = True; save_ckpt(state)
        print("      Ingestion complete!")
    else:
        print("\n[3/4] Skipping ingestion (checkpoint found).")

    # ── Evaluate ──────────────────────────────────────────────────
    print("\n[4/4] Evaluating all 3 systems...\n")
    done = set(state["done_idx"])

    for idx, s in enumerate(samples):
        if idx in done:
            pct = (len(done)/SAMPLE_SIZE)*100
            bar = "█"*int(pct/2.5) + "░"*(40-int(pct/2.5))
            print(f"\r  [{bar}] {pct:.0f}%  {len(done)}/{SAMPLE_SIZE} (checkpoint skip)", end="")
            continue

        q    = s["question"]
        know = s["knowledge"]
        gold = s["right_answer"]
        hall = s["hallucinated_answer"]
        pct  = (idx/SAMPLE_SIZE)*100
        bar  = "█"*int(pct/2.5) + "░"*(40-int(pct/2.5))
        print(f"\n  [{bar}] {pct:.0f}%  [{idx+1:03}/{SAMPLE_SIZE}]")
        print(f"  Q: {q[:80]}")
        print(f"  Gold: {gold[:50]}  |  Hallucination trap: {hall[:50]}")

        row = {"idx": idx, "question": q, "knowledge": know,
               "right_answer": gold, "hallucinated_answer": hall}

        # LangChain
        try:
            t0 = time.time()
            a  = run_lc(q, know)
            row.update({"lc_answer":a, "lc_em":em(a,gold),
                        "lc_f1":f1(a,gold), "lc_hrr":hrr(a,hall),
                        "lc_time":round(time.time()-t0,1)})
            print(f"  LC : {a[:70]}  EM={row['lc_em']:.0f} HRR={row['lc_hrr']:.0f}")
        except Exception as e:
            row["lc_answer"] = f"[err:{e}]"

        # v4-sim
        try:
            t0 = time.time()
            a  = run_v4(q, pipeline)
            row.update({"v4_answer":a, "v4_em":em(a,gold),
                        "v4_f1":f1(a,gold), "v4_hrr":hrr(a,hall),
                        "v4_time":round(time.time()-t0,1)})
            print(f"  V4 : {a[:70]}  EM={row['v4_em']:.0f} HRR={row['v4_hrr']:.0f}")
        except Exception as e:
            row["v4_answer"] = f"[err:{e}]"

        # v5
        try:
            t0 = time.time()
            a, conf = run_v5(q, pipeline)
            row.update({"v5_answer":a, "v5_confidence":conf,
                        "v5_em":em(a,gold), "v5_f1":f1(a,gold),
                        "v5_hrr":hrr(a,hall),
                        "v5_time":round(time.time()-t0,1)})
            print(f"  V5 : {a[:70]}  EM={row['v5_em']:.0f} HRR={row['v5_hrr']:.0f} Conf={conf}")
        except Exception as e:
            row["v5_answer"] = f"[err:{e}]"

        rows.append(row)
        state["done_idx"].append(idx)
        # Checkpoint every 5 queries (not every 1) to reduce disk I/O
        if len(state["done_idx"]) % 5 == 0:
            save_results(rows)
            save_ckpt(state)

    # ── Final Table ───────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  FINAL RESULTS — HaluEval Benchmark")
    print("="*70)
    print(f"  {'System':<22} {'EM%':>6} {'F1%':>6} {'HRR%':>7}  N")
    print("  " + "-"*48)
    for name, key in [("LangChain","lc"),("KG-RAG v4","v4"),("TruthfulRAG v5","v5")]:
        m = aggregate(rows, key)
        print(f"  {name:<22} {m['EM%']:>6} {m['F1%']:>6} {m['HRR%']:>7}  {m['N']}")
    print("  " + "-"*48)
    print(f"\n  HRR = Hallucination Rejection Rate")
    print(f"  Higher HRR = system correctly avoids the wrong answer")
    print(f"\n  Full results → {RESULT}")
    print("="*70)

if __name__ == "__main__":
    main()
