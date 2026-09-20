#!/usr/bin/env python3
"""Behavioral evaluation of the rag-retrieval-optimization skill's rules.

Runs a real retrieval pipeline (chunking via the skill's own chunk.py, BM25
keyword search, dense vector search, RRF hybrid fusion, cross-encoder
reranking) over public ko.wikipedia documents (data/corpus.json from fetch.py)
and the questions in dataset.json, and reports retrieval metrics for every
open question left in the skill's adversarial review:

  Q1  chunk size / overlap (rule 1: 200/60 has no stated basis)
  Q2  keyword tokenizer for Korean (rule 2a) and fusion vs single retrievers
  Q3  Top-k for aggregate questions and table documents (rules 1b, 3, 3b)
  Q4  reranker when 20 chunks go to the LLM vs when 5 do (rule 4a/4b)

Relevance is mechanical (see dataset.json "relevance_rule"); this measures
whether the evidence chunk is retrieved, not final LLM answer quality.

    python3 run_eval.py --report report.md --json results.json
"""
import argparse
import importlib.util
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHUNK_PY = HERE.parent / "rag-retrieval-optimization" / "scripts" / "chunk.py"
EMBED_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

spec = importlib.util.spec_from_file_location("skill_chunk", CHUNK_PY)
skill_chunk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skill_chunk)


# ---------------------------------------------------------------- corpus

def build_chunks(docs, size, overlap, table_mode="fixed"):
    """table_mode: 'fixed' = apply size/overlap to the table text like prose;
    'row' = one chunk per table row (the skill's author proposal for 1b)."""
    chunks = []
    for d in docs:
        if d["kind"] == "table" and table_mode == "row":
            pos = 0
            for line in d["text"].split("\n"):
                chunks.append({"doc": d["title"], "text": line, "start": pos, "end": pos + len(line)})
                pos += len(line) + 1
            continue
        for c in skill_chunk.chunk(d["text"], size, overlap):
            chunks.append({"doc": d["title"], "text": c["text"], "start": c["start"], "end": c["end"]})
    return chunks


def relevant(chunk, item):
    if item["kind"] in ("aggregate", "lookup"):
        return [ev for ev in item["evidence"] if ev in chunk["text"]]
    hits = []
    for ev in item["evidence"]:
        need = min(len(ev), 50)
        found = any(ev[i:i + need] in chunk["text"] for i in range(0, len(ev) - need + 1))
        if found:
            hits.append(ev)
    return hits


# ---------------------------------------------------------------- keyword (BM25)

def tok_ws(text):
    return re.findall(r"[0-9A-Za-z가-힣]+", text)


def tok_bigram(text):
    out = []
    for w in tok_ws(text):
        if len(w) == 1:
            out.append(w)
        out.extend(w[i:i + 2] for i in range(len(w) - 1))
    return out


class BM25:
    def __init__(self, texts, tok, k1=1.5, b=0.75):
        self.tok, self.k1, self.b = tok, k1, b
        self.docs = [tok(t) for t in texts]
        self.dl = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = self.dl.mean()
        self.tf = [defaultdict(int) for _ in self.docs]
        df = defaultdict(int)
        for i, d in enumerate(self.docs):
            for w in d:
                self.tf[i][w] += 1
            for w in set(d):
                df[w] += 1
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.docs))
        for w in self.tok(query):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s


# ---------------------------------------------------------------- fusion / metrics

def topn(scores, n):
    return list(np.argsort(-scores)[:n])


def rrf(rankings, k=60):
    s = defaultdict(float)
    for r in rankings:
        for pos, idx in enumerate(r):
            s[idx] += 1.0 / (k + pos + 1)
    return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])]


def metrics(ranked, chunks, item, ks=(5, 20)):
    """recall@k: any relevant chunk in top k. coverage@k: fraction of evidence
    strings with a relevant chunk in top k (aggregate). rr: 1/rank of first hit."""
    m = {}
    first = None
    covered = {k: set() for k in ks}
    for pos, idx in enumerate(ranked):
        hits = relevant(chunks[idx], item)
        if hits and first is None:
            first = pos + 1
        for k in ks:
            if pos < k:
                covered[k].update(hits)
    m["rr"] = 1.0 / first if first else 0.0
    for k in ks:
        m[f"recall@{k}"] = 1.0 if covered[k] else 0.0
        m[f"coverage@{k}"] = len(covered[k]) / len(item["evidence"])
    return m


def mean_by_kind(rows, key):
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        vals = [r[key] for r in rows if r["kind"] == kind]
        out[kind] = sum(vals) / len(vals)
    out["all"] = sum(r[key] for r in rows) / len(rows)
    return out


# ---------------------------------------------------------------- pipeline

class Pipeline:
    def __init__(self, embedder, reranker, docs, size, overlap, table_mode, tok):
        self.chunks = build_chunks(docs, size, overlap, table_mode)
        texts = [c["text"] for c in self.chunks]
        self.bm25 = BM25(texts, tok)
        self.emb = embedder.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=64)
        self.embedder, self.reranker = embedder, reranker
        self._qcache = {}

    def qvec(self, q):
        if q not in self._qcache:
            self._qcache[q] = self.embedder.encode(["query: " + q], normalize_embeddings=True)[0]
        return self._qcache[q]

    def vector(self, q, n):
        return topn(self.emb @ self.qvec(q), n)

    def keyword(self, q, n):
        return topn(self.bm25.scores(q), n)

    def hybrid(self, q, n):
        return rrf([self.vector(q, n), self.keyword(q, n)])[:n]

    def rerank(self, q, cand, n):
        pairs = [(q, self.chunks[i]["text"]) for i in cand]
        s = self.reranker.predict(pairs, batch_size=32)
        order = np.argsort(-np.asarray(s))
        return [cand[i] for i in order[:n]]


def evaluate(pipe, items, retrieve):
    rows = []
    for it in items:
        ranked = retrieve(pipe, it["question"])
        m = metrics(ranked, pipe.chunks, it)
        m["kind"] = it["kind"]
        rows.append(m)
    return rows


def fmt(d):
    return " / ".join(f"{k}={v:.2f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="report.md")
    ap.add_argument("--json", default="results.json")
    a = ap.parse_args()
    t0 = time.time()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    corpus = json.loads((HERE / "data" / "corpus.json").read_text())
    docs = corpus["docs"]
    ds = json.loads((HERE / "dataset.json").read_text())
    items = ds["items"]
    prose = [i for i in items if i["kind"].startswith("prose")]
    table = [i for i in items if not i["kind"].startswith("prose")]
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    reranker = CrossEncoder(RERANK_MODEL, device="cpu")
    results, lines = {}, []
    lines.append("# rag-retrieval-optimization behavioral eval\n")
    lines.append(f"corpus: {corpus['source']}, live fetch={corpus['fetched_live']}, docs={len(docs)}; questions={len(items)} "
                 f"(prose {len(prose)}, table {len(table)}); embed={EMBED_MODEL}; rerank={RERANK_MODEL}; relevance: {ds['relevance_rule']}\n")

    # sanity: every question's evidence is findable under the default chunking
    base = Pipeline(embedder, reranker, docs, 200, 60, "fixed", tok_bigram)
    unfindable = [it["question"] for it in items if not any(relevant(c, it) for c in base.chunks)]
    lines.append(f"sanity: default 200/60 chunks={len(base.chunks)}; questions with no relevant chunk at all: {len(unfindable)}")
    for q in unfindable:
        lines.append(f"  - {q}")
    lines.append("")

    # Q1 chunking (prose, hybrid, k=20)
    lines.append("## Q1 chunk size / overlap (prose questions, hybrid RRF, bigram BM25)\n")
    lines.append("| size/overlap | chunks | recall@5 | recall@20 | MRR |\n|---|---|---|---|---|")
    q1 = {}
    for size, ov in [(200, 60), (200, 0), (100, 30), (400, 100), (400, 0)]:
        p = base if (size, ov) == (200, 60) else Pipeline(embedder, reranker, docs, size, ov, "fixed", tok_bigram)
        rows = evaluate(p, prose, lambda p, q: p.hybrid(q, 20))
        q1[f"{size}/{ov}"] = {"chunks": len(p.chunks), "recall@5": mean_by_kind(rows, "recall@5")["all"],
                              "recall@20": mean_by_kind(rows, "recall@20")["all"], "mrr": mean_by_kind(rows, "rr")["all"]}
        r = q1[f"{size}/{ov}"]
        lines.append(f"| {size}/{ov} | {r['chunks']} | {r['recall@5']:.2f} | {r['recall@20']:.2f} | {r['mrr']:.2f} |")
    results["q1_chunking"] = q1
    lines.append("")

    # Q2 tokenizer and retriever mix (prose, 200/60)
    lines.append("## Q2 retriever mix and Korean tokenizer (prose questions, 200/60)\n")
    lines.append("| retriever | tokenizer | recall@5 | recall@20 | MRR | recall@20 keyword-style | recall@20 paraphrase-style |\n|---|---|---|---|---|---|---|")
    q2 = {}
    ws = Pipeline(embedder, reranker, docs, 200, 60, "fixed", tok_ws)
    for name, p, fn, tk in [("vector", base, lambda p, q: p.vector(q, 20), "-"),
                            ("keyword", ws, lambda p, q: p.keyword(q, 20), "whitespace"),
                            ("keyword", base, lambda p, q: p.keyword(q, 20), "bigram"),
                            ("hybrid", ws, lambda p, q: p.hybrid(q, 20), "whitespace"),
                            ("hybrid", base, lambda p, q: p.hybrid(q, 20), "bigram")]:
        rows = evaluate(p, prose, fn)
        r20 = mean_by_kind(rows, "recall@20")
        q2[f"{name}/{tk}"] = {"recall@5": mean_by_kind(rows, "recall@5")["all"], "recall@20": r20["all"], "mrr": mean_by_kind(rows, "rr")["all"],
                              "recall@20_keyword_q": r20["prose-keyword"], "recall@20_paraphrase_q": r20["prose-paraphrase"]}
        r = q2[f"{name}/{tk}"]
        lines.append(f"| {name} | {tk} | {r['recall@5']:.2f} | {r['recall@20']:.2f} | {r['mrr']:.2f} | {r['recall@20_keyword_q']:.2f} | {r['recall@20_paraphrase_q']:.2f} |")
    results["q2_retrievers"] = q2
    lines.append("")

    # Q3 table doc + aggregate questions
    lines.append("## Q3 table document: fixed 200/60 vs row chunks; Top-k for aggregate questions (hybrid, bigram)\n")
    lines.append("| table chunking | question kind | coverage@5 | coverage@20 | coverage@50 | recall@20 |\n|---|---|---|---|---|---|")
    q3 = {}
    rowp = Pipeline(embedder, reranker, docs, 200, 60, "row", tok_bigram)
    for name, p in [("fixed 200/60", base), ("one row per chunk", rowp)]:
        rows = []
        for it in table:
            ranked = p.hybrid(it["question"], 50)
            m = metrics(ranked, p.chunks, it, ks=(5, 20, 50))
            m["kind"] = it["kind"]
            rows.append(m)
        for kind in ("lookup", "aggregate"):
            sub = [r for r in rows if r["kind"] == kind]
            r = {k: sum(x[k] for x in sub) / len(sub) for k in ("coverage@5", "coverage@20", "coverage@50", "recall@20")}
            q3[f"{name}/{kind}"] = r
            lines.append(f"| {name} | {kind} | {r['coverage@5']:.2f} | {r['coverage@20']:.2f} | {r['coverage@50']:.2f} | {r['recall@20']:.2f} |")
    results["q3_table"] = q3
    lines.append("")
    lines.append("aggregate evidence counts per question: " + ", ".join(f"{it['answer']}" for it in table if it["kind"] == "aggregate") + "\n")

    # Q4 reranker
    lines.append("## Q4 reranker (prose questions, 200/60, hybrid bigram candidates)\n")
    lines.append("| pipeline | final n | recall@5 | recall@20 | MRR |\n|---|---|---|---|---|")
    q4 = {}
    configs = [
        ("hybrid top-20, no reranker (rule 4: skip)", 20, lambda p, q: p.hybrid(q, 20)),
        ("hybrid top-20 -> rerank -> 20 (4a challenge)", 20, lambda p, q: p.rerank(q, p.hybrid(q, 20), 20)),
        ("hybrid top-5, no reranker", 5, lambda p, q: p.hybrid(q, 5)),
        ("hybrid top-20 -> rerank -> 5 (rule 4b)", 5, lambda p, q: p.rerank(q, p.hybrid(q, 20), 5)),
        ("hybrid top-50 -> rerank -> 5", 5, lambda p, q: p.rerank(q, p.hybrid(q, 50), 5)),
        ("hybrid top-50 -> rerank -> 20", 20, lambda p, q: p.rerank(q, p.hybrid(q, 50), 20)),
    ]
    for name, n, fn in configs:
        rows = evaluate(base, prose, fn)
        q4[name] = {"final_n": n, "recall@5": mean_by_kind(rows, "recall@5")["all"], "recall@20": mean_by_kind(rows, "recall@20")["all"], "mrr": mean_by_kind(rows, "rr")["all"]}
        r = q4[name]
        lines.append(f"| {name} | {n} | {r['recall@5']:.2f} | {r['recall@20']:.2f} | {r['mrr']:.2f} |")
    results["q4_reranker"] = q4
    lines.append("")
    lines.append(f"elapsed {time.time() - t0:.0f}s\n")

    Path(a.report).write_text("\n".join(lines))
    Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("\n".join(lines))
    if unfindable:
        print(f"FAIL: {len(unfindable)} questions have no relevant chunk in the corpus", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
