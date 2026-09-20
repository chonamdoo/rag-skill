# Adversarial review of the source rules

Record of the critical review of Rules 1–4 in SKILL.md at authoring time (2026-09-21). Every item is the skill author's inference and has not been approved by the source (the system prompt supplied in the authoring request, which summarizes an unidentified video). Read this when choosing which items need a user decision during pipeline design, or when explaining why a source rule may not fit the current setting.

Verdicts: **keep** = source rule stays the default; **conditional** = source rule applies only under the stated condition; **conflict** = the author's view is the opposite of the source rule and the conflict is unresolved.

| # | Source rule | Attack | Verdict | Evidence that would settle it |
|---|---|---|---|---|
| 1a | 200-char chunks | "One context" and a fixed character count are different criteria; mid-sentence cuts are possible. | keep (sentence-boundary alignment offered as an option) | Retrieval accuracy with vs. without sentence alignment |
| 1b | 200-char chunks | Tables, code, and lists do not fit a meaning unit in 200 chars. | conditional (apply to prose documents) | Confirm document type |
| 1c | 60-char overlap | 30% duplication raises index and embedding cost; no basis given. | keep | Boundary-question accuracy at overlap 0 / 30 / 60 |
| 1d | Unit of "characters" | Characters vs. tokens undefined. | author decision: characters | Confirm with the source |
| 2a | Keyword search prevents false positives | With whitespace tokenization, Korean particles make "성수점은" ≠ "성수점". | conditional (needs morphological analysis or n-grams) | Check the search engine's tokenizer |
| 2b | Merge both result sets | Fusion method (RRF / weighted sum / filter) undefined; bad weights let one side dominate. | keep (require recording the method) | Evaluate fusion methods |
| 2c | Preventing branch confusion | A metadata filter on branch name is more reliable. | proposal (outside the source rule) | Whether metadata exists |
| 3a | Top-20 | 4× context; lost-in-the-middle risk (Liu et al., 2023). | keep (when k exceeds the budget, apply the author decision in SKILL.md Rule 3 Execution) | LLM context limit, positional utilization |
| 3b | Enough documents for aggregate questions | Retrieval cannot guarantee completeness; raising k never proves "all". | keep + range-limiting phrase mandatory (author decision); structured query is a proposal | Whether structured data exists |
| 4a | Skip reranker at Top-20 | Common practice is the reverse: retrieve wide → rerank → pass few. More candidates raise the reranker's value. The source does not separate retrieved count from LLM-fed count. | **conflict** | Answer quality with vs. without reranker at the same k |
| 4b | Reranker required at Top-5 or fewer | Agreed. But with only 5 candidates the reranker has little to reorder — the first pass must be wider. | keep (recommend ≥ 20 first-pass candidates) | — |
| 4c | 6–19 range | Undefined by the source. | author proposal: reranker on by default, flagged for user confirmation | User decision |
| 4d | Meaning of "secured/extracted" count | First-pass candidates or LLM-fed count undefined. Read as first-pass, the 4b configuration (20 candidates, 5 passed) triggers "skip" and "required" at once. | author decision: LLM-fed count (basis: the "token limit" condition) | Confirm with the source |

## Handling of conflict 4a

Keep the source rule as the default. Reason: the skill's provenance is that rule, and the author's view is general practice, not something validated on this knowledge base. The design output must state (1) which side was chosen, (2) why, and (3) that the opposing view exists. A user report of degraded answers (wrong answers caused by irrelevant chunks) is the trigger to revisit 4a.

## Measured results (2026-09-21, public-document experiment)

The harness in the repository's `eval/` directory ran a real retrieval pipeline to measure the open items above. CI (`.github/workflows/eval.yml`) reproduces the experiment. The results are evidence **for this corpus and these models only**, not general laws.

- Corpus: 11 documents from ko.wikipedia.org (10 prose, 1 table), fetched live with pinned revision ids (oldid) and sha256 verification. CC BY-SA 4.0.
- Questions: 66. Prose 54 (27 keyword-style, 27 paraphrase-style; drafted by an LLM, evidence sentences verified verbatim against the text). Table 12 (8 aggregate, 4 lookup; hand-written).
- Pipeline: the skill's `chunk.py` → BM25 (whitespace vs. character-bigram tokens) + vector search (`intfloat/multilingual-e5-small`, cosine) → RRF → reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- Metrics: recall@k = an evidence chunk is in the top k; MRR = reciprocal rank of the first evidence chunk; coverage@k = fraction of an aggregate question's evidence rows found in the top k. LLM answer quality was not measured.

| Item | Measurement | Effect on verdict |
|---|---|---|
| 1c 60-char overlap | 200/60: recall@20 0.94, MRR 0.81 vs. 200/0: 0.85, 0.71 | Overlap clearly helps; the source rule now has a basis |
| 1a/1c 200-char size | 400/100: recall@5 0.93, MRR 0.85 > 200/60: 0.89, 0.81; 100/30: 0.81, 0.67 | 200 is not optimal here (400 did better). Treat 200 as a starting point, not a ceiling |
| 2a tokenizer | Keyword-only, paraphrased questions, recall@20: whitespace 0.41 vs. bigram 0.85 | Keyword search without morphological analysis or n-grams fails on paraphrases; the conditional verdict is confirmed |
| 2b hybrid effect | recall@20: vector 0.96, hybrid 0.94, keyword (bigram) 0.93; MRR: hybrid 0.81 > vector 0.76 > keyword 0.75 | Hybrid improved ranking (MRR) but not recall@20 over vector alone. "Always combine" is supported as a ranking improvement only |
| 3b aggregate questions | 2–7 evidence rows per question, yet coverage@20 0.59–0.70, coverage@50 0.86–0.93; the top 5 were prose chunks on the same topic, not table rows | Top-20 is incomplete even for small aggregates. The mandatory range-limiting phrase is confirmed; the structured-query proposal is strengthened |
| 1b table document | One row per chunk: coverage@20 0.59 vs. fixed 200/60: 0.70 (coverage@50: 0.93 vs. 0.86) | Row chunking was not better at k = 20. Table rows lack the topic word "national park" and lose to prose chunks. The row-chunking proposal stays unverified |
| 4b reranker with few chunks | Final 5: hybrid top-5 0.85 → top-20 → rerank → 5: 0.89 → top-50 → rerank → 5: 0.94 | Source rule supported; a wider first pass helps more |
| 4a reranker with 20 chunks | top-20 as-is MRR 0.81 → after rerank 0.87 (recall@20 unchanged at 0.94); top-50 → rerank → 20: recall@20 1.00 | Even when all 20 are passed, the reranker improves ordering, and widening to 50 candidates removes misses. "Skip at Top-20" is not supported on this corpus. The conflict stands, but the evidence favours the author's view |

Limits: 11 documents, 66 questions, one small model of each kind. Rankings may change in other domains or with other models. No statistical significance testing. Re-run: `cd eval && python3 fetch.py && python3 run_eval.py`.
