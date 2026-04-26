"""
Live progress monitor for rigorous_eval.py
Run in a second terminal: python monitor_eval.py
Refreshes every 15 seconds automatically.
"""
import json, os, time, sys

CKPT   = "rigorous_eval_checkpoint.json"
RESULT = "rigorous_eval_results.json"
TOTAL  = 180   # 60 samples × 3 datasets

def bar(done, total, width=40):
    pct  = done / max(total, 1)
    fill = int(width * pct)
    return f"[{'█'*fill}{'░'*(width-fill)}] {pct*100:5.1f}%  {done}/{total}"

def summary(rows):
    lc_em = [r.get("langchain_em",0) for r in rows]
    v4_em = [r.get("v4_em",0) for r in rows]
    v5_em = [r.get("v5_em",0) for r in rows]
    def avg(lst): return round(sum(lst)/len(lst)*100,1) if lst else 0.0
    return avg(lc_em), avg(v4_em), avg(v5_em)

while True:
    os.system("cls")
    print("="*60)
    print("  TruthfulRAG v5 — Live Evaluation Monitor")
    print(f"  {time.strftime('%H:%M:%S')}  (refreshes every 15s — Ctrl+C to exit)")
    print("="*60)

    ckpt_done = 0
    ingested  = False
    if os.path.exists(CKPT):
        with open(CKPT) as f: c = json.load(f)
        ckpt_done = len(c.get("done_idx", []))
        ingested  = c.get("ingested", False)

    rows = []
    if os.path.exists(RESULT):
        with open(RESULT) as f: rows = json.load(f)

    print(f"\n  Stage 1 — Dataset Download  : {'✅ Done' if rows or ingested else '🔄 In progress...'}")
    print(f"  Stage 2 — Neo4j Ingestion   : {'✅ Done' if ingested else '⏳ Waiting...'}")
    print(f"  Stage 3 — Query Evaluation  :")
    print(f"            {bar(ckpt_done, TOTAL)}")

    if rows:
        lc, v4, v5 = summary(rows)
        print(f"\n  Live Accuracy (so far, {len(rows)} queries):")
        print(f"    LangChain       EM: {lc:5.1f}%")
        print(f"    KG-RAG v4-sim   EM: {v4:5.1f}%")
        print(f"    TruthfulRAG v5  EM: {v5:5.1f}%  ← Our system")

        ds_counts = {}
        for r in rows:
            ds_counts[r.get("dataset","?")] = ds_counts.get(r.get("dataset","?"),0)+1
        print(f"\n  Datasets completed:")
        for ds, cnt in ds_counts.items():
            print(f"    {ds:<25} {cnt} queries")

        elapsed = sum(r.get("v5_time",0) for r in rows)
        remaining = (elapsed / max(len(rows),1)) * (TOTAL - len(rows))
        print(f"\n  Estimated time remaining: ~{int(remaining//60)}m {int(remaining%60)}s")
    else:
        print("\n  No results yet — download/ingestion still running...")

    print("\n" + "="*60)

    if ckpt_done >= TOTAL:
        print("  ✅ EVALUATION COMPLETE — check rigorous_eval_results.json")
        break

    time.sleep(15)
