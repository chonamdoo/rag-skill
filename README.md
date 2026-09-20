# rag-skill

An agent skill for the retrieval side of RAG (검색 증강 생성) knowledge bases, plus a behavioral evaluation harness that measured its rules on public documents. The skill's rules come from a supplied system prompt; every rule was put through an adversarial review, and the open questions from that review were tested with a real retrieval pipeline in CI.

| Path | What it is |
|---|---|
| [`rag-retrieval-optimization/SKILL.md`](rag-retrieval-optimization/SKILL.md) | The skill: rules, execution steps, adversarial review, deliverables |
| [`rag-retrieval-optimization/references/adversarial-review.md`](rag-retrieval-optimization/references/adversarial-review.md) | Attack table per rule, verdicts, and measured evidence |
| [`rag-retrieval-optimization/scripts/chunk.py`](rag-retrieval-optimization/scripts/chunk.py) | Fixed-size chunker with overlap (200/60 default) |
| [`eval/`](eval/) | Retrieval pipeline + dataset that tests the rules on ko.wikipedia documents |
| [`.github/workflows/eval.yml`](.github/workflows/eval.yml) | CI: fetches the corpus live, runs the evaluation, publishes the report |

## Install

With the [`skills`](https://github.com/vercel-labs/skills) CLI:

```bash
# interactive: pick agents and scope
npx skills add chonamdoo/rag-skill

# non-interactive, global, Claude Code
npx skills add chonamdoo/rag-skill --skill rag-retrieval-optimization -g -a claude-code -y

# try it once without installing
npx skills use chonamdoo/rag-skill@rag-retrieval-optimization | claude
```

Manual: copy `rag-retrieval-optimization/` into your agent's skills directory (e.g. `~/.claude/skills/` or `~/.agents/skills/`).

## The skill, by keyword

**Chunking & Overlap (청킹 및 겹쳐 자르기)** — Source rule: 200-character chunks, the last 60 characters repeated at the start of the next chunk. `scripts/chunk.py` implements exactly that and emits `start`/`end` offsets for provenance. Review: character count ≠ "one context"; tables and code do not fit; the skill applies 200/60 to prose only and asks before chunking tables.

**Embedding (임베딩) / Vector Search (벡터 검색)** — Cosine similarity over embeddings finds passages that match in meaning, not wording. The evaluation uses `intfloat/multilingual-e5-small`.

**Hybrid Search (하이브리드 검색)** — Vector search plus exact-match keyword search (BM25), fused with RRF. The source rule says "always combine"; the skill additionally requires recording the fusion method and tagging each chunk with the retriever that returned it. Review: whitespace tokenization breaks Korean keyword search (particles), so a morphological analyzer or n-grams is a precondition.

**Top-k (검색 결과 수)** — Default candidate pool raised from 5 to 20; aggregate questions (counts, statistics) must retrieve more. Review: retrieval cannot prove completeness, so aggregate answers must say "aggregated over the retrieved range", and a structured query is proposed when one exists.

**Reranker (리랭커)** — Source rule: skip when 20+ chunks are secured; required when only ≤5 can be passed. The skill reads the threshold as the count passed to the LLM (author decision), defaults to reranking in the undefined 6–19 range, and records the unresolved conflict with common practice (retrieve wide → rerank → pass few).

## What the evaluation built

`eval/` is a self-contained RAG retrieval pipeline used as a test bench for the skill's rules — not a product:

- **Corpus**: 11 ko.wikipedia articles (10 prose, 1 table), pinned by revision id and sha256-verified on every live fetch (`eval/fetch.py`). CC BY-SA 4.0.
- **Dataset**: 66 questions (`eval/dataset.json`) — 54 prose (keyword-style vs. paraphrase-style, evidence sentences verified verbatim), 12 table (8 aggregate, 4 lookup).
- **Pipeline** (`eval/run_eval.py`): the skill's `chunk.py` → BM25 (whitespace vs. character-bigram) + dense vectors → RRF → cross-encoder reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- **Metrics**: recall@k, MRR, and coverage@k for aggregate questions. LLM answer quality is not measured.

Run locally:

```bash
cd eval
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python fetch.py          # live fetch, hash-verified (--offline uses the snapshot)
.venv/bin/python run_eval.py       # writes report.md / results.json (~4 min on an M-series CPU)
```

CI runs the same steps on every push touching the skill or the eval and posts the report to the job summary.

## Results (CI run, 2026-09-21)

| Question | Result | Consequence for the rule |
|---|---|---|
| Does the 60-char overlap help? | recall@20 0.85 → 0.94, MRR 0.71 → 0.81 vs. no overlap | Yes; source rule kept with evidence |
| Is 200 chars the right size? | 400/100 beat 200/60 (recall@5 0.93 vs. 0.89); 100/30 worse | 200 is a starting point, not a ceiling |
| Does the Korean tokenizer matter? | keyword-only recall@20 on paraphrases: whitespace 0.41 vs. bigram 0.85 | Keyword search needs morphology or n-grams |
| Does hybrid beat vector alone? | recall@20 equal (0.94 vs. 0.96); MRR 0.81 vs. 0.76 | Hybrid improves ranking, not recall, here |
| Is Top-20 enough for aggregates? | coverage@20 0.59–0.70 for 2–7 evidence rows; prose chunks crowd out table rows | No; range-limiting phrase is mandatory |
| Reranker when passing 5 chunks? | recall@5 0.85 → 0.89 (20 candidates) → 0.94 (50 candidates) | Source rule confirmed; widen the first pass |
| Reranker when passing 20 chunks? | MRR 0.81 → 0.87 at same recall; 50 → rerank → 20 reaches recall@20 1.00 | "Skip at Top-20" not supported on this corpus; conflict recorded |

Limits: one corpus, one small model per role, no significance testing. Full tables in the CI job summary and in the reference file.

## Provenance

Rules and glossary: a system prompt supplied at authoring time that summarizes an unidentified video. Everything labelled "author" in the skill is inference by the skill's author, not a claim of the source. Corpus text is from ko.wikipedia.org under CC BY-SA 4.0.
