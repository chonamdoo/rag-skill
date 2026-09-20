---
name: rag-retrieval-optimization
description: Design, configure, or review the retrieval side of a RAG (검색 증강 생성) knowledge-base pipeline — chunking with overlap (script included), hybrid (vector + keyword) search, Top-k sizing, and conditional reranker use — and report answers over retrieved chunks with source tags. Use when asked to set up or tune 지식 베이스 검색, 청킹, 하이브리드 검색, Top-k, 리랭커, or to review a RAG retrieval setup. Rules carry an adversarial-review record; apply the source defaults and surface the known tradeoffs rather than silently overriding them.
---

# RAG Knowledge-Base Retrieval Optimization

Document-processing and retrieval rules for finding the documents that actually answer a user's question in a knowledge base. Use it when **designing, configuring, or implementing** a pipeline, when **reviewing** an existing configuration, or when **answering** from retrieval results produced under these rules.

Rules 1–4 restate the system prompt supplied in the authoring request (numbers and conditions unchanged, sentences tidied). That prompt refers to "the video" but the video's title, creator, and publication date were not provided. Under each rule, the **Adversarial review** block is the skill author's inference, not a verified claim of the source. Do not blend the two: the source rule is the default; review items are surfaced to the user/operator for a decision. Full review record and measured evidence: [references/adversarial-review.md](references/adversarial-review.md).

## Terms

The definitions below are translations of the Korean wording supplied in the authoring request; the Korean term is kept in parentheses.

- **Vector Search (벡터 검색)**: A retrieval method that finds the needed information from context and meaning even when the words do not match exactly.
- **Embedding (임베딩)**: A technique that turns the meaning of a sentence into numbers and places it in a virtual space; the video calls this a "map of meaning".
- **Chunking & Overlap (청킹 및 겹쳐 자르기)**: Splitting a long document into pieces of a size that is easy to process (chunking), while repeating part of the previous piece at the start of the next so context at the cut is not lost.
- **Top-k (검색 결과 수)**: The number of candidate documents the AI consults to answer a question. The video stresses the importance of raising this from 5 to 20.
- **Reranker (리랭커)**: A model that re-compares the first-pass candidate documents against the user's question in detail and reorders them so the most relevant documents rank first.
- **Hybrid Search (하이브리드 검색)**: Combining meaning-centred "vector search" with exact-word-match "keyword search" so that each method covers the other's weaknesses.

## Inputs to confirm before starting

If any of these is missing, apply the rules as written and mark the gap as an "assumption" in the output.

- Document type, language, and location (Korean prose? tables, code, or lists mixed in? files or a DB?)
- Retrieval stack: embedding model, vector store, keyword engine and whether it has a Korean tokenizer (morphological analysis vs. whitespace-only)
- Context budget of the answering LLM (can it take all Top-20 chunks?)
- Reranker availability, latency, and cost limits
- Whether expected questions include **aggregate questions** (counts, statistics). If automatic detection is needed, key on words such as "how many", "total", "all", "every", "list" (Korean: "몇 개", "총", "전부", "모든", "목록") — author proposal; the source defines no detection method.

`scripts/chunk.py` is addressed relative to this skill directory. Call it by absolute path from any other working directory.

## Rule 1. Chunking with Overlap

**Source rule**
- Split documents into **200-character** units, each holding one context.
- To avoid breaking context at the cut, **repeat the last 60 characters of the previous chunk at the start of the next** (overlapping).

**Execution**
- Do not cut by hand; run `scripts/chunk.py`: `python3 scripts/chunk.py --size 200 --overlap 60 input.txt` (JSON Lines, each chunk with `index`, `start`, `end`, `text`). The last chunk may be shorter than 200.
- Record the chunk count and the `start`/`end` offsets so retrieval hits can be traced back to the source text.

**Adversarial review (author inference)**
- "One context" and "200 characters" are different criteria. The script enforces only the character count, so sentences can be cut mid-way. Sentence-boundary alignment is not in the source rule → ask the user whether to apply it.
- 200 characters is roughly 1–3 Korean sentences. Tables, code, and long lists do not fit a meaning unit in 200 characters, so the source rule is applied as-is **only to prose**. If the document set is mostly tables or code, do not auto-apply 200/60; ask the user to choose between row/section-based splitting (author proposal) and the source rule.
- 60/200 = 30% overlap grows the index and embedding cost by that much. The source gives no basis for the value.
- The source does not say whether "characters" means characters or tokens. This skill reads it as **characters** (Python `len`) — author decision.

## Rule 2. Hybrid Search

**Source rule**
- Never rely on a single retrieval method; always combine both.
  - **Vector search**: use cosine similarity to find documents close in context and meaning even when the words differ.
  - **Keyword search**: find documents where proper nouns or specific words match exactly, to prevent vector-search false positives (e.g. confusing the Seongsu branch with the Pangyo branch).
- Merge both result sets to produce the final document ranking.

**Execution**
- Run both searches and **record the fusion method explicitly** (e.g. RRF, weighted sum). The source does not fix a fusion method, so mark the choice as an author/executor decision.
- Tag every candidate chunk with which retriever returned it (vector / keyword / both). Used for answer provenance and for checking the fusion.
- If the question contains a proper noun (branch, product, person, code), confirm that exact string went into the keyword query unchanged.

**Adversarial review (author inference)**
- With whitespace tokenization, Korean particles make "성수점은" and "성수점" different tokens. Without a morphological analyzer or n-grams there is no guarantee keyword search performs the false-positive-prevention role.
- Keyword search is fragile to typos and synonyms. "Each covers the other's weaknesses" holds only when the fusion weights are reasonable.
- The Seongsu/Pangyo case is often better handled by a **metadata filter** on branch. When available it is more reliable than keyword search, but it is not in the source rule → offered as a proposal only.

## Rule 3. Top-k Optimization

**Source rule**
- Widen the default candidate pool **from Top-5 to Top-20** so extraction is generous.
- For **aggregate questions** (statistics, counts), secure enough documents that none of the required ones is missed.

**Execution**
- Default k = 20 (first-pass candidate count). For aggregate questions raise k until the required documents are covered (source rule, mandatory). The source gives no number — author proposal: at least the expected number of relevant documents (e.g. number of target items × chunks per item); record the value and its basis.
- If k exceeds the LLM context budget: retrieve k, pass only the budget to the LLM, and if the final count falls under the Rule 4 threshold select with the reranker. Aggregate answers must carry the phrase "aggregated over the retrieved range" — author decision; the source does not cover this case.
- Record "N chunks retrieved, M used" in the answer.

**Adversarial review (author inference)**
- Top-20 quadruples context length and cost. Information in the middle of long contexts is reported to be under-used (Liu et al., 2023, "Lost in the Middle"), so a larger k is not automatically more accurate.
- Aggregate questions **cannot be made complete by retrieval**: no value of k proves that "all" documents were fetched. Therefore the "aggregated over the retrieved range" phrase is mandatory when aggregating via retrieval. Author proposal (outside the source rule; mark as a deviation if adopted): prefer a structured query (DB/metadata aggregation) when one is available.

## Rule 4. Conditional Reranker

**Source rule**
- When **Top-20 or more** documents have been secured, **skip** the reranker pipeline.
- In environments where **only a few documents (Top-5 or fewer)** can be extracted, e.g. because of token limits, **always run the reranker** to compare question and documents and re-rank.

The source does not define whether the "secured/extracted" count is the first-pass candidate count or the number passed to the LLM (it uses the same word as Rule 3). This skill reads it as the **final number of chunks passed to the LLM** — author decision. Basis: the condition is "token limits", which constrain LLM input. If this reading is wrong, the thresholds below change.

**Execution**
- Final chunk count ≤ 5 → reranker required. ≥ 20 → skipped per the source rule. 6–19 → undefined by the source; default to using the reranker (author proposal), flag the range in the output, and raise it for user confirmation.
- When the reranker is on, retrieve a first-pass pool larger than the final count (recommended ≥ 20, author proposal). With only 5 candidates there is nothing to reorder.
- Record "first-pass candidates → after reranker" whenever the reranker runs.

**Adversarial review (author inference) — conflicts with the source rule**
- Common practice is the reverse of the source rule: **retrieve wide (e.g. 50–100) → compress with the reranker → pass a few to the LLM**. The more candidates, the more the reranker is worth. The source appears not to distinguish "retrieved candidates" from "chunks given to the LLM".
- "All 20 go to the LLM, so no reranker" treats the reranker as a **recall fallback** rather than a **precision tool**. Where noisy chunks degrade answers this reading can be wrong.
- The conflict is unresolved. Keep the source rule as the default, but the design output must state the conflict and the reason for the choice. In the public-document experiment (reference file, "Measured results"), the reranker raised MRR from 0.81 to 0.87 even when 20 chunks were passed, and 50 candidates → reranker → 20 reached recall@20 = 1.00 — the evidence favours the author's view, but it is one corpus. Show the user these numbers and let them decide.

## Deliverables and completion criteria

A pipeline design/configuration result must contain all of:

1. Chunking parameters (size, overlap, unit = characters) and resulting chunk count
2. Vector search and keyword search settings, and the fusion method
3. First-pass k, final count passed to the LLM, and the handling of aggregate questions
4. Whether the reranker is used and the Rule 4 basis for that decision
5. Every departure from a source rule with its reason, and the adversarial-review items that need a user decision

For a question-answering task: the source of each chunk used (document, `start`/`end`), whether it came from both retrievers, and, for aggregates, the range-limiting phrase.

## Do not

- Change a source rule silently on the strength of a review item. If you change it, say so in the output.
- Put anything in an answer that was not retrieved. If it is not in the chunks, answer "not in the retrieved range".
- Quote the original source (the video) as fact without having checked it. This skill's provenance stops at "the explanation supplied in the authoring request".

## Provenance and confidence

- Rules 1–4 and term definitions: the system prompt and glossary supplied in the authoring request (skill authored 2026-09-21). The explanation refers to a video whose title, creator, and date were not provided. The rules' effective date and validation basis are unknown.
- Skill author's inferences and proposals (not approved by the source): every adversarial-review item, the "final chunk count" reading of Rule 4, the 6–19 default, the ≥ 20 first-pass recommendation, the "characters" reading, the fusion-method and retriever-tag recording requirements, the aggregate-detection words and k sizing, and the structured-query proposal.
- Measured (reference file, "Measured results"; ko.wikipedia, 11 documents, 66 questions, small models): 60-char overlap helps (recall@20 0.85 → 0.94); 400 chars beat 200; whitespace tokenization fails on paraphrased questions (0.41 vs 0.85 with bigrams); hybrid improves ranking more than recall; Top-20 is insufficient even for small aggregates (coverage@20 ≤ 0.70); the reranker helps at both 5 and 20 final chunks. Evidence is limited to that corpus.
- Still unresolved: the source's basis for the specific 200/60 values, fusion-method choice, row-based chunking for tables (no k=20 gain in the experiment), and generalization to other domains and models.
