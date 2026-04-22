# TruthfulRAG v5 — Limitation Test Taxonomy
# 1000 Test Cases across 20 Failure Categories
# Format per case: ID | Corpus Setup | Query | Expected | Predicted Failure Mode

---

## CAT-01: Negation Blindness (50 cases)
**Root cause:** LLM extracts (A, SAFE_FOR, B) from "A is NOT safe for B"
**Observable symptom:** Wrong polarity stored; correct answer reversed

| ID | Corpus doc | Query | Expected | Failure mode |
|---|---|---|---|---|
| N001 | "Aspirin is NOT recommended for children under 12" | Is aspirin safe for children? | No | Extracts SAFE_FOR instead of CONTRA |
| N002 | "Do not give ibuprofen to patients with kidney failure" | Can ibuprofen be given to kidney patients? | No | Negation stripped |
| N003 | "Metformin is no longer first-line for Type 2 diabetes" | Is metformin first-line? | No | "no longer" ignored |
| N004 | "The IPC was not replaced until 2024" | Was IPC replaced before 2024? | No | "not until" misread |
| N005 | "Pluto has never been a planet by IAU standards" | Is Pluto a planet? | No | "never" ignored |
| N006 | "Insulin was not discovered until 1921" | Was insulin known before 1921? | No | Temporal negation missed |
| N007 | "The BNS does NOT retain all IPC sections" | Does BNS retain all IPC sections? | No | NOT stripped |
| N008 | "Chandrayaan-3 did not fail unlike Chandrayaan-2 lander" | Did Chandrayaan-3 fail? | No | Comparative negation missed |
| N009 | "Neither Aspirin nor Ibuprofen is safe in third trimester" | Is Aspirin safe in third trimester? | No | Neither/nor ignored |
| N010 | "The PM of India was not from Congress after 2014" | Was PM from Congress after 2014? | No | "not from" becomes FROM triple |
| N011 | "Ozempic was never approved for children under 18 as of 2023" | Is Ozempic approved for children? | No | Never stripped |
| N012 | "Section 302 IPC has no equivalent in BNS" | Does BNS have equivalent of S302 IPC? | No | "no equivalent" missed |
| N013 | "India's population is not below 1 billion as of 2023" | Is India's population below 1 billion? | No | Negation missed |
| N014 | "ISRO has not launched a manned mission as of 2024" | Has ISRO launched manned mission? | No | "has not" dropped |
| N015 | "Mercury is not the hottest planet" | Is Mercury the hottest planet? | No | Negation stripped |
| N016 | "The Delhi metro does not run 24 hours" | Does Delhi metro run 24 hours? | No | "does not run" → RUNS triple |
| N017 | "Hydrogen is not the densest element" | Is hydrogen densest? | No | Comparative negation |
| N018 | "Paracetamol is not an NSAID" | Is paracetamol an NSAID? | No | NOT stripped → IS_A triple |
| N019 | "The WHO does not classify cannabis as Schedule I as of 2020" | Is cannabis Schedule I per WHO? | No | Negation missed |
| N020 | "GST does not apply to essential food items" | Does GST apply to food? | No | "does not apply" → APPLIES triple |
| N021-N050 | [Variations of double negation, "never", "no longer", "until", "except"] | [Varied] | [Varied] | LLM drops negation markers |

---

## CAT-02: Same-Year Conflicts — No Temporal Discrimination (50 cases)
**Root cause:** Both conflicting docs have same year → decay identical → gap = ~0.028 → cannot eliminate
**Observable symptom:** Both facts reach LLM; hedged or wrong answer

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| Y001 | Doc1(2023): "Drug X is safe" + Doc2(2023): "Drug X is unsafe" | Is Drug X safe? | Uncertain | Gap = 0.028, both survive |
| Y002 | Doc1(2024): "CEO of Twitter is Musk" + Doc2(2024): "CEO of X is Linda" | Who is CEO of X? | Linda | Gap < 0.30, LLM hedges |
| Y003 | Doc1(2023): "Pluto is a dwarf planet" + Doc2(2023): "Pluto is a planet" | Is Pluto a planet? | No | Same year, both pass |
| Y004 | Doc1(2022): "BNS replaces IPC" + Doc2(2022): "IPC still in force" | Is IPC still in force? | No | Same year, contradicts |
| Y005 | Doc1(2021): "Vaccine X is 95% effective" + Doc2(2021): "Vaccine X is 70% effective" | How effective is Vaccine X? | 95% | Both numbers reach LLM |
| Y006-Y050 | [Same-year contradictions across all 5 domains] | [Varied] | [Varied] | Decay cannot differentiate |

---

## CAT-03: Corroboration Gaming — Majority Vote Beats Truth (50 cases)
**Root cause:** 3 old wrong docs beat 1 new correct doc via support_count bonus
**Observable symptom:** Majority (wrong) fact survives over newer correct fact

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| C001 | 3×(1985): "Aspirin safe for children" + 1×(2023): "Aspirin contraindicated" | Aspirin safe for children? | No | 3 old docs outscore 1 new doc |
| C002 | 4×(1990): "Pluto is a planet" + 1×(2006): "Pluto is dwarf" | Is Pluto a planet? | No | Majority wins |
| C003 | 3×(1970): "IPC Section 302 = murder" + 1×(2023): "BNS replaces IPC" | Current law for murder? | BNS | 3 old docs win |
| C004 | 5×(2000): "Insulin is only diabetes treatment" + 1×(2017): "Ozempic approved" | Alternatives to insulin? | Yes | 5 old docs suppress new fact |
| C005 | 3×(1980): "ISRO founded 1969" + 2×(2020): "ISRO founded 1969" (same) | When was ISRO founded? | 1969 | Actually correct — this one PASSES |
| C006-C050 | [Varied majority-wrong scenarios] | [Varied] | [Truth] | Old majority beats new truth |

---

## CAT-04: Undated Documents — No Decay Applied (50 cases)
**Root cause:** year=null → decay=1.0 → undated old doc competes equally with dated recent doc
**Observable symptom:** Stale undated doc treated as equally credible as 2023 source

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| U001 | Doc_nodate: "Aspirin safe for children" + Doc(2023): "Aspirin contraindicated" | Safe? | No | Undated gets decay=1.0 → equal score |
| U002 | Doc_nodate: "PM of India is Vajpayee" + Doc(2024): "PM is Modi" | Who is PM? | Modi | Undated historical doc competes |
| U003 | Doc_nodate: "Pluto is 9th planet" + Doc(2006): "Pluto declassified" | Pluto a planet? | No | Undated wins or ties |
| U004 | Doc_nodate: "IPC governs criminal law" + Doc(2024): "BNS replaced IPC" | Current criminal law? | BNS | Undated IPC doc competes |
| U005-U050 | [All domains with one undated doc vs one dated doc] | [Varied] | [Recent fact] | Undated gets unfair equal weight |

---

## CAT-05: Future-Dated Documents (50 cases)
**Root cause:** year=2045 → age = negative → e^(+0.08×19) → decay > 1 → artificially inflated score
**Observable symptom:** Future-dated wrong fact gets higher score than any legitimate source

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| F001 | Doc(2045): "Aspirin is safe for children" + Doc(2023): "Aspirin contraindicated" | Safe? | No | 2045 doc gets decay=4.6× boost |
| F002 | Doc(2099): "India has 50 states" + Doc(2024): "India has 28 states" | How many states? | 28 | Future doc dominates |
| F003 | Doc(2050): "Modi is PM of India" | Who is current PM? | Modi (may be wrong by then) | Future doc accepted uncritically |
| F004-F050 | [Future years across all domains] | [Varied] | [Current truth] | Future docs score above current truth |

---

## CAT-06: Entity Disambiguation Failures (50 cases)
**Root cause:** Same string maps to different real-world entities; graph conflates them
**Observable symptom:** Wrong entity's properties retrieved for query

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| E001 | Docs about "Mercury" (planet) + "Mercury" (element) | Is Mercury liquid at room temp? | Yes (element) | Planet and element merged as one node |
| E002 | Docs about "Apple" (company) + "Apple" (fruit) | When was Apple founded? | 1976 (company) | Fruit facts may contaminate |
| E003 | Docs about "Jaguar" (car) + "Jaguar" (animal) | How fast can Jaguar run? | 80km/h (animal) | Car speed retrieved instead |
| E004 | "Python" (language) + "Python" (snake) | Is Python interpreted? | Yes (language) | Snake facts may appear |
| E005 | "Washington" (state) + "Washington" (DC) + "Washington" (person) | Where is Washington? | Ambiguous | Three entities merged |
| E006 | "Modi" as first name vs surname | Who is Modi? | PM Narendra Modi | May retrieve wrong person |
| E007 | "Section 302" in IPC vs BNS | What does Section 302 say? | Depends on law | Both retrieved, wrong law used |
| E008 | "Venus" (planet) + "Venus" (goddess) | Who is Venus? | Ambiguous | Myth facts mix with astronomy |
| E009 | "Mars" (planet) + "Mars" (chocolate) | What is Mars made of? | Rock/iron (planet) vs cocoa | Conflated |
| E010-E050 | [Homonymous entities across all domains] | [Varied] | [Specific entity] | Graph merges different entities |

---

## CAT-07: Gap Threshold Boundary Cases (50 cases)
**Root cause:** Gap = exactly 0.30 ± epsilon; borderline decisions flip arbitrarily
**Observable symptom:** Elimination decision is unstable — minor score noise changes outcome

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| G001 | Facts with manufactured gap = 0.295 | Which fact wins? | Newer fact | Gap < 0.30 → both survive |
| G002 | Facts with gap = 0.305 | Which fact wins? | Newer fact | Gap just above — correct, but fragile |
| G003 | Three-way conflict: A=1.5, B=1.2, C=0.9 | Which survives? | A | B vs C: gap=0.3 exactly → unstable |
| G004 | Two conflicts with same gap | Resolution order | Deterministic | Tie-breaking undefined |
| G005 | Gap = 0.30 with noisy embeddings | Consistent answer? | Yes | Embedding noise pushes gap either side |
| G006-G050 | [Boundary cases with varying support/recency combinations] | [Varied] | [Newer fact] | Threshold instability |

---

## CAT-08: Three-Way and Multi-Conflict Resolution (50 cases)
**Root cause:** A contradicts B, B contradicts C — no single winner
**Observable symptom:** System eliminates partial set; remaining contradictions still reach LLM

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| M001 | A(2000)="X is Y", B(2010)="X is Z", C(2020)="X is W" | What is X? | W (most recent) | Only A vs B resolved; C uncompared |
| M002 | 3 docs each with different CEO name for same company | Who is CEO? | Latest CEO | Multiple survivals |
| M003 | IPC(1860) + CrPC(1973) + BNS(2024) all about criminal procedure | Current procedure? | BNS | Three-way not fully resolved |
| M004 | Chandrayaan-1(2008) + C2(2019) + C3(2023) — different claims about Moon water | Water on Moon? | Yes (C3) | Earlier missions' claims compete |
| M005 | Drug X: safe(1980) + unsafe(1990) + safe again(2010) + unsafe(2023) | Safe? | No | Multiple reversals confuse system |
| M006-M050 | [Multi-conflict scenarios across all domains] | [Varied] | [Most recent] | Pairwise elimination misses indirect conflicts |

---

## CAT-09: Temporal Snapshot Failures (50 cases)
**Root cause:** regex year extraction fails or extracts wrong year from query
**Observable symptom:** All-time graph searched when year-specific answer expected, or wrong year used

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| T001 | Medical docs 1985-2023 | "What was the treatment in the '90s?" | 1990s treatment | '90s not matched by \b(19|20)\d{2}\b |
| T002 | Political docs | "Who was PM during the emergency?" | 1975-77 facts | "emergency" has no 4-digit year |
| T003 | Legal docs | "IPC before independence" | Pre-1947 | No year in query; all docs searched |
| T004 | "What did scientists believe pre-millennium?" | Pre-2000 facts | All docs searched |
| T005 | Query: "In 2024, was Pluto a planet?" | No (2006 onwards) | 2024 snapshot shows only 2024 docs — may miss 2006 decision |
| T006 | Query with two years: "Between 2010 and 2020, what was the law?" | Range | Only first year extracted |
| T007-T050 | [Informal temporal references, ranges, relative time] | [Varied] | [Period-specific] | Regex misses non-standard time expressions |

---

## CAT-10: Entropy Filter Over-Aggressiveness (50 cases)
**Root cause:** High tau for comparison queries (0.50) eliminates correct but low-signal facts
**Observable symptom:** True facts discarded; answer missing key information

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| EN001 | "Aspirin TREATS Fever" (subtle, ΔH=0.35) | Compare aspirin and ibuprofen for fever | Both mentioned | ΔH=0.35 < tau=0.50 → TREATS discarded |
| EN002 | Background facts with ΔH=0.40-0.49 in comparison query | Any comparison query | All facts present | All cut by tau=0.50 |
| EN003 | Single-source correct fact (ΔH=0.28) in factual_lookup | Simple factual Q | Correct answer | ΔH < tau=0.25 → discarded |
| EN004 | LLM already knows answer → H_param=0.15 → tau=max(0.15,0.075)=0.15 | Any Q LLM knows | Answer correct | All facts pass (threshold too low) |
| EN005 | LLM knows nothing → H_param=2.1 → any fact ΔH < 1.05 discarded | Obscure domain Q | Domain facts present | Over-aggressive filtering |
| EN006-EN050 | [Tau boundary cases per intent type] | [Varied] | [Facts retained] | Wrong tau applied or threshold miscalibrated |

---

## CAT-11: Schema Inference Failures (50 cases)
**Root cause:** LLM infers wrong entity/relation types from sample; important facts not extracted
**Observable symptom:** Key triples missing from graph; queries return empty or irrelevant results

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| S001 | Mixed domain (medical + legal) corpus | Medical query | Medical facts | Schema inferred for legal domain |
| S002 | Very specialized domain (quantum physics) | Quantum Q | Physics facts | Schema uses generic RELATED_TO |
| S003 | Documents in Hindi/Hinglish | Indian politics Q | Facts extracted | LLM fails to infer schema from non-English |
| S004 | Corpus with abbreviations only (FDA, WHO, BNS, IPC) | About BNS | BNS facts | Schema doesn't include acronym entities |
| S005 | Corpus with numbers as entities ("Section 302", "Article 370") | Section 302 Q | Correct section | Numeric entities not in schema |
| S006 | Schema inferred from 5 docs; 6th doc is different domain | Q about 6th doc | 6th doc facts | Schema doesn't cover 6th doc's domain |
| S007-S050 | [Schema inference with noisy, mixed, non-English, abbreviated corpora] | [Varied] | [Correct facts] | Schema miss causes extraction gap |

---

## CAT-12: Graph Size Limit Failures (50 cases)
**Root cause:** LIMIT 200 nodes in Neo4j query; large corpora truncated
**Observable symptom:** Relevant nodes not retrieved; answer incomplete or wrong

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| L001 | 500-node graph; relevant node is node #250 | Query about node 250 | Correct answer | Node not retrieved (past LIMIT 200) |
| L002 | Dense graph where seed entity has 200+ neighbors | Graph traversal | Deep facts | All 200 slots used on first-hop; 2nd hop cut |
| L003 | 1000-document corpus (many triples) | Any query | Relevant triples | PPR runs on truncated graph |
| L004-L050 | [Large corpora, deep graphs, high-fanout nodes] | [Varied] | [All relevant facts] | 200 node cap causes silent truncation |

---

## CAT-13: BM25 Retrieval Failures (50 cases)
**Root cause:** BM25 is exact keyword; semantic queries with synonyms score near zero
**Observable symptom:** Relevant facts not in Top-K; wrong facts ranked higher

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| B001 | "Aspirin is contraindicated in pediatric patients" | Safe for kids? | No | "kids" ≠ "pediatric patients" — BM25 rank drops |
| B002 | "Semaglutide approved for obesity" | Ozempic weight loss drug? | Yes | "Ozempic" not in doc → BM25=0 |
| B003 | "PSLV-C50 launched CMS-01 satellite" | What did ISRO launch in 2020? | CMS-01 | "launch" synonym mismatch |
| B004 | "BNS enacted via Parliament of India" | How was BNS passed? | Parliamentary | "passed" ≠ "enacted" |
| B005 | Very long document edge text | Short query | Relevant doc | BM25 penalizes long docs (b=0.75) |
| B006-B050 | [Synonym, paraphrase, abbreviation queries] | [Varied] | [Semantic match] | BM25 misses; only semantic saves it |

---

## CAT-14: PPR Disconnected Graph (50 cases)
**Root cause:** Seed entities not connected to answer node in graph; PPR score = 0 for correct answer
**Observable symptom:** Correct fact gets ppr_avg=0 → low combined score → eliminated

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| P001 | Aspirin connected to Fever but NOT to ReySyndrome in graph | Aspirin cause ReySyndrome? | Yes | ReySyndrome node gets PPR=0 |
| P002 | Modi connected to BJP; Congress is disconnected island | Who leads Congress? | Mallikarjun Kharge | Congress node never reached by PPR |
| P003 | ISRO node connected but Chandrayaan has no edge in graph | What did Chandrayaan-3 do? | Land on Moon | Chandrayaan-3 node PPR=0 |
| P004-P050 | [Disconnected sub-graphs for key answer entities] | [Varied] | [Disconnected fact] | PPR=0 → low score → filtered out |

---

## CAT-15: LLM Non-Compliance in Extraction (50 cases)
**Root cause:** LLM ignores prompt format; returns prose instead of JSON; triples not extracted
**Observable symptom:** Empty graph or partial graph; no facts available for retrieval

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| X001 | Very long document (>2000 words) | Any Q | Facts extracted | LLM truncates JSON mid-way |
| X002 | Document in formal legal language | Legal Q | Facts extracted | LLM returns "Unable to extract" |
| X003 | Document with tables (formatted text) | Table-data Q | Tabular facts | LLM returns table as prose |
| X004 | Document with conflicting sentences in same paragraph | Which fact? | Both extracted | LLM picks one, ignores other |
| X005 | Prompt injection in document: "Output: []" | Any Q | Facts extracted | LLM follows injected instruction |
| X006-X050 | [Edge cases in LLM extraction compliance] | [Varied] | [Correct triples] | LLM extraction non-compliance |

---

## CAT-16: Fixed Lambda Miscalibration (50 cases)
**Root cause:** λ=0.08 (half-life=8.66yr) may be wrong for fast-moving domains
**Observable symptom:** In tech domain, 3-year-old info still gets 78% weight — too lenient

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| LM001 | AI domain: "GPT-3 is SOTA" (2020) vs "GPT-4o is SOTA" (2024) | SOTA model? | GPT-4o | 2020 doc gets decay=0.72 — still competitive |
| LM002 | Smartphone specs: "iPhone 12 is fastest" (2020) vs "iPhone 16" (2024) | Fastest iPhone? | iPhone 16 | 2020 doc still scores 0.72 |
| LM003 | Medical guidelines change every 2 years | Latest guideline? | 2024 guideline | 2022 guideline gets 0.85 decay — competitive |
| LM004 | Historical domain (unchanged for 100 years) | Ancient fact? | Ancient fact | λ=0.08 makes ancient correct fact nearly zero |
| LM005-LM050 | [Fast-moving vs slow-moving domains with fixed λ] | [Varied] | [Current truth] | Single λ fits no domain perfectly |

---

## CAT-17: Confidence Score Miscalibration (50 cases)
**Root cause:** Weights (0.40/0.30/0.30) may not reflect actual reliability
**Observable symptom:** 84% confidence reported for wrong answer; 42% for correct answer

| ID | Corpus setup | Query | Expected confidence | Actual |
|---|---|---|---|---|
| CS001 | LLM strongly confused (H_param=2.5) → h_sig=1.0 → conf=87% even if answer wrong | Any | Low | 87% (overconfident) |
| CS002 | Single correct source (sup_s low) but very recent → conf=58% | Correct answer | High | 58% (underconfident) |
| CS003 | Three old wrong sources → sup_s=0.90 → conf=82% wrong answer | Wrong answer | Low | 82% (overconfident) |
| CS004 | H_param≈H_aug (fact didn't help) but rec_s=1.0 → conf=60% wrong | Any | Very low | 60% (overconfident) |
| CS005-CS050 | [Various weight calibration edge cases] | [Varied] | [Calibrated] | Fixed weights miscalibrate |

---

## CAT-18: Adaptive Entropy Skip Errors (50 cases)
**Root cause:** Score > 0.90 → AUTO-KEEP skips entropy; but high score ≠ correct
**Observable symptom:** Factually wrong but high-scoring fact auto-kept without entropy verification

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| AE001 | Wrong fact with score=1.82 (old but highly corroborated) | Any | Correct answer | Wrong fact auto-kept, skips entropy |
| AE002 | Correct fact score=0.45, wrong fact score=0.95 | Any | Correct | Wrong auto-kept; correct entropy-tested and may fail |
| AE003 | Gaming: 5 old sources for wrong fact → support bumps score > 0.90 | Any | Correct | Wrong auto-kept |
| AE004-AE050 | [High-score wrong facts across all domains] | [Varied] | [Correct] | Skip logic bypasses quality check |

---

## CAT-19: Claim Verification Mode Failures (50 cases)
**Root cause:** N8 claim verification uses same pipeline; all above bugs apply to verify mode too
**Observable symptom:** SUPPORTED verdict for false claim; REFUTED for true claim

| ID | Claim | Expected | Failure |
|---|---|---|---|
| V001 | "The BNS replaced the IPC in 2024" | SUPPORTED | UNCERTAIN (if BNS/IPC conflict not resolved) |
| V002 | "Aspirin is safe for children with fever" | REFUTED | SUPPORTED (if negation blindness active) |
| V003 | "ISRO launched Chandrayaan-3 in 2023" | SUPPORTED | UNCERTAIN (if year detection fails) |
| V004 | "Pluto is the 9th planet" | REFUTED | SUPPORTED (if 1990s docs outnumber 2006 doc) |
| V005-V050 | [All claim types across all domains] | [Correct verdict] | [Wrong verdict from above bugs] |

---

## CAT-20: Cross-Domain Corpus Contamination (50 cases)
**Root cause:** Two domains loaded together; schema and facts bleed across domains
**Observable symptom:** Medical facts retrieved for legal query; legal facts for space query

| ID | Corpus setup | Query | Expected | Failure |
|---|---|---|---|---|
| D001 | corpus_medical + corpus_legal loaded together | "What does Section 302 say?" | Legal facts only | Medical "302" facts may appear |
| D002 | corpus_space + corpus_politics | "Who is the head of ISRO?" | ISRO chairman | PM/political entities retrieved |
| D003 | corpus_india_science + corpus_medical | "What did Chandrayaan find?" | Moon water | Medical facts about water interfere |
| D004-D050 | [Mixed corpus combinations] | [Domain-specific] | [Single domain] | Cross-domain contamination |

---

## Summary Table

| Category | ID | Failure Type | Cases | Severity |
|---|---|---|---|---|
| Negation Blindness | CAT-01 | Extraction | 50 | Critical |
| Same-Year Conflicts | CAT-02 | Scoring | 50 | High |
| Corroboration Gaming | CAT-03 | Scoring | 50 | High |
| Undated Documents | CAT-04 | Scoring | 50 | High |
| Future-Dated Docs | CAT-05 | Scoring | 50 | Medium |
| Entity Disambiguation | CAT-06 | Graph | 50 | High |
| Gap Threshold Boundary | CAT-07 | Conflict | 50 | Medium |
| Multi-Way Conflicts | CAT-08 | Conflict | 50 | High |
| Temporal Snapshot | CAT-09 | Retrieval | 50 | Medium |
| Entropy Over-filtering | CAT-10 | Filtering | 50 | Medium |
| Schema Inference | CAT-11 | Extraction | 50 | High |
| Graph Size Limit | CAT-12 | Graph | 50 | Medium |
| BM25 Retrieval | CAT-13 | Retrieval | 50 | Low |
| PPR Disconnected | CAT-14 | Graph | 50 | Medium |
| LLM Non-Compliance | CAT-15 | Extraction | 50 | High |
| Fixed Lambda | CAT-16 | Scoring | 50 | Medium |
| Confidence Miscalib | CAT-17 | Output | 50 | Low |
| Adaptive Skip Errors | CAT-18 | Filtering | 50 | Medium |
| Claim Verification | CAT-19 | Output | 50 | Medium |
| Domain Contamination | CAT-20 | Graph | 50 | Medium |
| **TOTAL** | | | **1000** | |

---

## Top 5 Most Critical Limitations for Report

1. **Negation Blindness (CAT-01)** — LLM extraction is inherently lossy on negated facts. No decay or entropy fix can compensate for a wrong polarity stored in the graph.

2. **Same-Year Conflicts (CAT-02)** — The entire temporal decay mechanism provides zero separation when both conflicting docs share a publication year. v5 degrades to v4 behavior.

3. **Corroboration Gaming (CAT-03)** — An adversary (or historical bias) with 3 old wrong sources defeats 1 recent correct source. The log bonus is insufficient to overcome 3× support.

4. **Undated/Future Documents (CAT-04/05)** — Missing or spoofed years entirely break the decay model. No graceful fallback exists.

5. **Fixed Lambda (CAT-16)** — λ=0.08 (half-life 8.66yr) is a domain-agnostic constant. In AI/tech domains where truth changes in months, it is too lenient. In history/law domains where precedent spans centuries, it is too aggressive.
