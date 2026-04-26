"""
Generate all missing diagram figures for TruthfulRAG v5 report.
Output: report_file/fig_arch.png, fig_dfd0.png, fig_dfd1.png,
        fig_class.png, fig_conflict.png, fig_seq.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import os

OUT = r'd:\Project-1\Report_Format___BTech_CSE_IIIT_Manipur__1___1_\report_file'
DPI = 180

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {name}')

# ─────────────────────────────────────────────────────────────────────────────
# 1. SIX-LAYER ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def make_arch():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

    layers = [
        (5, "Layer 0: Document Ingestion",    "corpus.json / plain-text files", 0.92),
        (4, "Layer 1: Schema Inference [A4+]","LLM → entity & relation types → CFG", 0.82),
        (3, "Layer 2: Graph Construction [A]","Triple extraction → Neo4j MERGE", 0.72),
        (2, "Layer 3: Graph Retrieval [B]",   "PPR + BM25 + Semantic → RRF ranking", 0.62),
        (1, "Layer 4: Conflict Resolution [C]","Temporal decay + entropy → answer", 0.52),
        (0, "Layer 5: Presentation",          "Flask API + Dual-panel Web UI", 0.42),
    ]
    grays = [0.93, 0.85, 0.77, 0.69, 0.58, 0.46]
    for (row, title, sub, _), g in zip(layers, grays):
        fc = str(g)
        rect = FancyBboxPatch((0.3, row*1.0+0.05), 7.5, 0.85,
                               boxstyle="round,pad=0.04", linewidth=1,
                               edgecolor='black', facecolor=fc)
        ax.add_patch(rect)
        ax.text(4.05, row*1.0+0.62, title, ha='center', va='center',
                fontsize=10, fontweight='bold')
        ax.text(4.05, row*1.0+0.25, sub, ha='center', va='center',
                fontsize=8.5, color='#222222')

    # Arrows between layers
    for y in [5.05, 4.05, 3.05, 2.05, 1.05]:
        ax.annotate('', xy=(4.05, y), xytext=(4.05, y+0.05-0.05),
                    arrowprops=dict(arrowstyle='->', lw=1.4, color='black'))

    # Ollama column
    for row in [1,2,3,4]:
        rx = FancyBboxPatch((8.1, row*1.0+0.1), 1.6, 0.7,
                             boxstyle="round,pad=0.04", lw=0.8,
                             edgecolor='black', facecolor='0.88', linestyle='dashed')
        ax.add_patch(rx)
    ax.text(8.9, 4.95, 'Ollama\n(LLM)', ha='center', va='center', fontsize=8,
            fontweight='bold')

    # Neo4j column
    for row in [2,3]:
        rx2 = FancyBboxPatch((8.1, row*1.0+0.1), 1.6, 0.7,
                              boxstyle="round,pad=0.04", lw=1.2,
                              edgecolor='black', facecolor='0.75')
        ax.add_patch(rx2)
    ax.text(8.9, 2.95, 'Neo4j\n(Graph DB)', ha='center', va='center', fontsize=8,
            fontweight='bold')

    ax.set_title('TruthfulRAG v5 — Six-Layer Architecture', fontsize=13, fontweight='bold', pad=10)
    save(fig, 'fig_arch.png')

# ─────────────────────────────────────────────────────────────────────────────
# 2. LEVEL-0 DFD (Context Diagram)
# ─────────────────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, text, fs=9):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                        boxstyle="round,pad=0.05", lw=1.2,
                        edgecolor='black', facecolor='0.88')
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, fontweight='bold')

def circle(ax, cx, cy, r, text, fs=9):
    c = plt.Circle((cx, cy), r, color='0.75', ec='black', lw=1.5)
    ax.add_patch(c)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, fontweight='bold')

def arr(ax, x0,y0,x1,y1, label='', side='top'):
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
    mx, my = (x0+x1)/2, (y0+y1)/2
    dy = 0.12 if side == 'top' else -0.15
    ax.text(mx, my+dy, label, ha='center', va='center', fontsize=7.5,
            style='italic', color='#333')

def make_dfd0():
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_xlim(0,9); ax.set_ylim(0,6); ax.axis('off')

    circle(ax, 4.5, 3, 1.1, 'TruthfulRAG\nv5 System', fs=9)
    box(ax, 1, 3, 1.6, 0.7, 'User', fs=10)
    box(ax, 7.5, 4.5, 1.8, 0.7, 'Ollama\n(LLM)', fs=9)
    box(ax, 7.5, 1.5, 1.8, 0.7, 'Neo4j\n(Graph DB)', fs=9)

    arr(ax, 1.8,3.15, 3.4,3.1, 'corpus.json\n(docs + queries)', 'top')
    arr(ax, 3.4,2.9,  1.8,2.85,'Answer + Confidence\n+ Audit Trail', 'bot')
    arr(ax, 5.5,3.3,  6.6,4.2, 'Prompts\n(schema/extract)', 'top')
    arr(ax, 6.6,4.5,  5.5,3.5, 'LLM responses','bot')
    arr(ax, 5.5,2.7,  6.6,1.8, 'Cypher MERGE\n(write)', 'top')
    arr(ax, 6.6,1.5,  5.5,2.5, 'Cypher results\n(paths)','bot')

    ax.set_title('Level-0 Context DFD — TruthfulRAG v5', fontsize=12, fontweight='bold')
    save(fig, 'fig_dfd0.png')

# ─────────────────────────────────────────────────────────────────────────────
# 3. LEVEL-1 DFD
# ─────────────────────────────────────────────────────────────────────────────
def make_dfd1():
    fig, ax = plt.subplots(figsize=(11,7))
    ax.set_xlim(0,11); ax.set_ylim(0,7); ax.axis('off')

    # Processes
    circle(ax, 2,   5.5, 0.85, 'P1\nSchema\nInference', fs=8)
    circle(ax, 5,   5.5, 0.85, 'P2\nGraph\nConstruction', fs=8)
    circle(ax, 5,   2,   0.85, 'P3\nGraph\nRetrieval', fs=8)
    circle(ax, 8.5, 2,   0.85, 'P4\nConflict\nResolution', fs=8)

    # Data Stores
    for (lx,rx,y,lbl) in [(3.5,6.5,4,'D1: Neo4j Graph'),(3.5,6.5,3.5,'D2: CFG Dictionary')]:
        ax.plot([lx,rx],[y,y],'k-',lw=1.2)
        ax.plot([lx,rx],[y-0.3,y-0.3],'k-',lw=1.2)
        ax.plot([lx,lx],[y,y-0.3],'k-',lw=0.8)
        ax.text((lx+rx)/2, y-0.15, lbl, ha='center', va='center', fontsize=8.5)

    # External
    box(ax, 0.8, 2, 1.2, 0.55, 'User', fs=9)
    box(ax, 9.8, 5.5, 1.4, 0.6, 'Ollama', fs=9)

    # Flows
    arr(ax,1.4,5.5, 1.15,5.5,'doc sample','top')
    arr(ax,2.85,5.5,3.5,4.15,'inferred\nschema','top')
    arr(ax,2,4.7,2,3.55,'entity/rel types','top')
    arr(ax,5,4.7,5,3.55,'MERGE writes','top')
    arr(ax,5,2.85,5,4.2,'Cypher reads','bot')
    arr(ax,5.85,2,7.65,2,'KnowledgePaths','top')
    arr(ax,8.5,2.85,8.5,3.5,'','')
    arr(ax,7.35,2.2,1.8,2.2,'answer + confidence','top')
    arr(ax,2.85,5.5,9.1,5.5,'prompts','top')
    arr(ax,9.1,5.2,5.85,5.2,'responses','bot')

    ax.set_title('Level-1 DFD — Four Sub-Processes', fontsize=12, fontweight='bold')
    save(fig, 'fig_dfd1.png')

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODULE C CONFLICT RESOLUTION FLOWCHART
# ─────────────────────────────────────────────────────────────────────────────
def make_flowchart():
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(0,7); ax.set_ylim(0,11); ax.axis('off')

    def rect(cx,cy,w,h,txt,fs=8.5,fc='0.88'):
        r = FancyBboxPatch((cx-w/2,cy-h/2),w,h,
                            boxstyle="round,pad=0.06",lw=1,
                            edgecolor='black',facecolor=fc)
        ax.add_patch(r)
        ax.text(cx,cy,txt,ha='center',va='center',fontsize=fs,
                multialignment='center')

    def diamond(cx,cy,w,h,txt,fs=8):
        dx,dy = w/2, h/2
        pts = [(cx,cy+dy),(cx+dx,cy),(cx,cy-dy),(cx-dx,cy)]
        poly = plt.Polygon(pts, closed=True, fc='0.80', ec='black', lw=1)
        ax.add_patch(poly)
        ax.text(cx,cy,txt,ha='center',va='center',fontsize=fs,multialignment='center')

    def dn(x,y1,y2): ax.annotate('',xy=(x,y2),xytext=(x,y1),
                                  arrowprops=dict(arrowstyle='->',lw=1.1,color='black'))
    def rt(x1,y,x2): ax.annotate('',xy=(x2,y),xytext=(x1,y),
                                  arrowprops=dict(arrowstyle='->',lw=1.1,color='black'))

    rect(3.5,10.3,5,0.55,'KnowledgePaths list received',fc='0.95')
    dn(3.5,10.05,9.6)
    rect(3.5,9.3,5,0.55,'_detect_contradictions()\nyear-based supersession',fs=8)
    dn(3.5,9.05,8.55)
    diamond(3.5,8.2,3.2,0.75,'Contradictions\nfound?')
    # YES branch
    ax.text(5.2,8.2,'YES',fontsize=8,color='black')
    rt(5.1,8.2,6.2)
    rect(6.4,8.2,1.4,0.8,'Discard\nolder\npath',fs=7.5)
    ax.annotate('',xy=(6.4,9.3),xytext=(6.4,8.6),
                arrowprops=dict(arrowstyle='->',lw=1,color='black',
                connectionstyle='arc3,rad=-0.5'))
    # NO branch
    ax.text(3.5,7.7,'NO',fontsize=8)
    dn(3.5,7.82,7.2)
    rect(3.5,6.9,5,0.55,'_entropy() — sample LLM n=3 times\ncompute H_param',fs=8)
    dn(3.5,6.62,6.05)
    diamond(3.5,5.7,3.5,0.75,'H_param < threshold?\n(entropy skip)')
    # YES skip
    ax.text(5.35,5.7,'YES',fontsize=8)
    rt(5.25,5.7,6.2)
    rect(6.4,5.7,1.4,0.55,'Skip\nentropy\npass',fs=7.5)
    # NO
    ax.text(3.5,5.2,'NO',fontsize=8)
    dn(3.5,5.32,4.7)
    rect(3.5,4.35,5,0.55,'Compute ΔH per path\nKeep paths where ΔH > τ',fs=8)
    dn(3.5,4.08,3.5)
    rect(3.5,3.15,5,0.55,'_build_chain() — format audit trail',fs=8)
    dn(3.5,2.88,2.3)
    diamond(3.5,1.95,3.5,0.65,'Contradictions\nexisted?')
    ax.text(1.5,1.95,'YES',fontsize=8)
    rt(1.75,1.95,0.8)
    rect(0.55,1.95,1.2,0.65,'EXPLAIN\nPROMPT',fs=7.5,fc='0.75')
    ax.text(5.35,1.95,'NO',fontsize=8)
    rt(5.25,1.95,6.2)
    rect(6.45,1.95,1.4,0.65,'RAG\nPROMPT',fs=7.5,fc='0.85')
    dn(3.5,1.62,0.85)
    rect(3.5,0.5,5,0.55,'_confidence() + Return answer + meta',fc='0.75')

    ax.set_title('Module C: Conflict Resolution Flowchart', fontsize=12, fontweight='bold', y=1.01)
    save(fig, 'fig_conflict.png')

# ─────────────────────────────────────────────────────────────────────────────
# 5. UML CLASS DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
def make_class():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0,13); ax.set_ylim(0,8); ax.axis('off')

    classes = [
        (1.0, 6.8, 'LLMCache',              ['_s: Dict','hits','misses'],        ['get()','set()','stats()']),
        (4.5, 6.8, 'HybridRetriever',       ['bm25','embedder','doc_vecs'],      ['retrieve()']),
        (8.5, 6.8, 'TripleRecord',          ['head','relation','tail','year','support'], []),
        (1.0, 4.0, 'EnhancedGraphConstructor',['llm','graph','cfg','_triple_bank'],['build()','_infer_schema()','_extract()']),
        (4.5, 4.0, 'KnowledgePath',         ['context','score','max_year','delta_h'], []),
        (8.0, 4.0, 'EnhancedGraphRetriever',['llm','graph','embedder','cfg'],    ['retrieve()','_ppr()','_filter_edges()']),
        (1.0, 1.5, 'EnhancedConflictResolver',['llm','llm_s','emb','cfg'],       ['resolve()','_entropy()','_confidence()']),
        (6.5, 1.5, 'EnhancedPipeline',      ['cfg','constructor','retriever','resolver'],['ingest()','query()','verify()']),
    ]

    def cls_box(cx, cy, name, attrs, methods):
        lw = max(len(name), max((len(a) for a in attrs+methods), default=0)) * 0.13 + 0.3
        lw = max(lw, 2.2)
        header_h, row_h = 0.38, 0.27
        total_h = header_h + (len(attrs)+len(methods)+0.5)*row_h + 0.1
        x0, y0 = cx-lw/2, cy
        # Header
        ax.add_patch(FancyBboxPatch((x0,y0),lw,header_h,
                     boxstyle='square,pad=0',lw=1.2,ec='black',fc='0.72'))
        ax.text(cx,y0+header_h/2,name,ha='center',va='center',fontsize=7.5,fontweight='bold')
        # Attrs
        sep1 = y0-len(attrs)*row_h
        ax.add_patch(FancyBboxPatch((x0,sep1),lw,len(attrs)*row_h,
                     boxstyle='square,pad=0',lw=1,ec='black',fc='0.90'))
        for i,a in enumerate(attrs):
            ax.text(x0+0.08, y0-(i+0.55)*row_h, a, va='center', fontsize=6.5)
        # Methods
        sep2 = sep1-len(methods)*row_h-0.05
        if methods:
            ax.add_patch(FancyBboxPatch((x0,sep2),lw,len(methods)*row_h+0.05,
                         boxstyle='square,pad=0',lw=1,ec='black',fc='0.96'))
            for i,m in enumerate(methods):
                ax.text(x0+0.08, sep1-(i+0.55)*row_h-0.05, m, va='center', fontsize=6.5)

    for cx,cy,name,attrs,methods in classes:
        cls_box(cx,cy,name,attrs,methods)

    # Composition arrows: Pipeline -> constructor, retriever, resolver
    for tx in [1.0, 8.0, 1.0]:
        ax.annotate('',xy=(tx,4.8),xytext=(6.5,2.3),
                    arrowprops=dict(arrowstyle='-|>',lw=1.1,color='black'))
    # label
    ax.text(5.5,3.1,'composes',fontsize=7,style='italic',color='#444')

    ax.set_title('UML Class Diagram — enhanced_main.py', fontsize=12, fontweight='bold')
    save(fig, 'fig_class.png')

# ─────────────────────────────────────────────────────────────────────────────
# 6. END-TO-END SEQUENCE DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
def make_seq():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0,12); ax.set_ylim(0,10); ax.axis('off')

    actors = ['Browser','Flask\nServer','EnhancedPipeline','Ollama','Neo4j']
    xs     = [1, 3, 5.8, 8.5, 11]
    TOP = 9.5

    for x, name in zip(xs, actors):
        ax.add_patch(FancyBboxPatch((x-0.7, TOP-0.3), 1.4, 0.55,
                     boxstyle='round,pad=0.05', lw=1.2, ec='black', fc='0.80'))
        ax.text(x, TOP-0.02, name, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.plot([x,x],[TOP-0.3, 0.3], color='gray', lw=0.8, ls='--')

    msgs = [
        # Phase 1 — Load
        (None, None, None, None, '— PHASE 1: Corpus Load —', 9.0),
        (xs[0],xs[1], 8.7,8.7, 'POST /api/load', None),
        (xs[1],xs[2], 8.4,8.4, 'ingest(docs)', None),
        (xs[2],xs[3], 8.1,8.1, '_infer_schema() prompt', None),
        (xs[3],xs[2], 7.8,7.8, 'entity/relation types JSON', None),
        (xs[2],xs[4], 7.5,7.5, 'DETACH DELETE (clear graph)', None),
        (xs[2],xs[3], 7.2,7.2, 'LLMGraphTransformer prompts', None),
        (xs[2],xs[4], 6.9,6.9, 'MERGE triple writes (loop)', None),
        (xs[2],xs[4], 6.6,6.6, 'APOC disambiguate + normalize', None),
        (xs[1],xs[0], 6.3,6.3, '{status:"loaded", nodes:N, edges:E}', None),
        # Phase 2 — Query
        (None, None, None, None, '— PHASE 2: Query Execution —', 6.0),
        (xs[0],xs[1], 5.7,5.7, 'POST /api/query/v5', None),
        (xs[1],xs[2], 5.4,5.4, 'query(q)', None),
        (xs[2],xs[3], 5.1,5.1, '_keys() + detect_intent()', None),
        (xs[2],xs[4], 4.8,4.8, '_edges() Cypher fetch', None),
        (xs[2],xs[2], 4.5,4.5, '_ppr() + _detect_contradictions() [local]', None),
        (xs[2],xs[3], 4.2,4.2, '_entropy() × 3 samples', None),
        (xs[2],xs[2], 3.9,3.9, '_build_chain() + _confidence() [local]', None),
        (xs[2],xs[3], 3.6,3.6, 'final answer prompt', None),
        (xs[3],xs[2], 3.3,3.3, 'answer text', None),
        (xs[1],xs[0], 3.0,3.0, '{answer, confidence%, paths, conflicts}', None),
    ]

    for x0,x1,y0,y1,label,ylab in msgs:
        if x0 is None:
            ax.text(6,ylab,label,ha='center',va='center',fontsize=8.5,
                    fontweight='bold',style='italic',color='#555')
            ax.plot([0.2,11.8],[ylab-0.12,ylab-0.12],color='#bbb',lw=0.8,ls=':')
            continue
        if x0 == x1:  # self-call
            ax.annotate('',xy=(x0+0.5,y1-0.15),xytext=(x0+0.5,y0),
                        arrowprops=dict(arrowstyle='->',lw=1,
                        connectionstyle='arc3,rad=-0.6',color='black'))
        else:
            ax.annotate('',xy=(x1,y1),xytext=(x0,y0),
                        arrowprops=dict(arrowstyle='->' if x1>x0 else '<-',
                        lw=1,color='black'))
        mx = (x0+x1)/2
        ax.text(mx, y0+0.1, label, ha='center', va='bottom', fontsize=7.2, color='#111')

    ax.set_title('End-to-End Sequence Diagram — Corpus Load + Query', fontsize=12, fontweight='bold')
    save(fig, 'fig_seq.png')

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    make_arch()
    make_dfd0()
    make_dfd1()
    make_flowchart()
    make_class()
    make_seq()
    print('\nAll diagrams generated.')
