#!/usr/bin/env python3
"""Fetch the evaluation corpus from ko.wikipedia.org by fixed revision id.

Every document in data/snapshot.json carries the revid and sha256 it was
authored against. This script re-downloads each revision from the live site,
rebuilds the same text, and verifies the hash, so the run provably uses the
public source documents rather than only the committed copy.

    python3 fetch.py            # fetch, verify, write data/corpus.json
    python3 fetch.py --offline  # copy snapshot to corpus.json without network
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

API = "https://ko.wikipedia.org/w/api.php"
UA = "rag-skill-eval/0.1 (https://github.com/chonamdoo/rag-skill)"
HERE = Path(__file__).resolve().parent


def api(**params):
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # 429 / transient
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"wikipedia api failed: {url}: {last}")


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.row, self.cell, self.depth = [], None, None, 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            self.tables.append([])
        elif tag == "tr" and self.depth:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1
        elif tag == "tr" and self.row is not None:
            self.tables[-1].append(self.row)
            self.row = None
        elif tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def fetch_prose(title, revid):
    r = api(action="query", prop="extracts", explaintext=1, revids=revid)
    page = next(iter(r["query"]["pages"].values()))
    assert page["title"] == title, (page["title"], title)
    return page["extract"]


def fetch_table(revid, table_index=1):
    r = api(action="parse", oldid=revid, prop="text")
    p = TableParser()
    p.feed(r["parse"]["text"]["*"])
    rows = [[re.sub(r"\[.*?\]", "", c).strip() for c in row] for row in p.tables[table_index]]
    rows = [row for row in rows if row and row[0] != "합계"]
    header, body = rows[0], rows[1:]
    text = "\n".join(" | ".join(f"{h}: {v}" for h, v in zip(header, row)) for row in body)
    return text, header, body


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    snap = json.loads((HERE / "data" / "snapshot.json").read_text())
    out = {"source": snap["source"], "fetched_live": not a.offline, "docs": []}
    for d in snap["docs"]:
        doc = dict(d)
        if not a.offline:
            time.sleep(2)  # be polite; the API rate-limits bursts
            if d["kind"] == "prose":
                text = fetch_prose(d["title"], d["revid"])
            else:
                text, header, body = fetch_table(d["revid"])
                doc["header"], doc["rows"] = header, body
            got = sha(text)
            if got != d["sha256"]:
                print(f"HASH MISMATCH {d['title']} revid={d['revid']}: {got} != {d['sha256']}", file=sys.stderr)
                sys.exit(1)
            doc["text"] = text
            print(f"fetched {d['title']} oldid={d['revid']} sha ok ({len(text)} chars)")
        out["docs"].append(doc)
    (HERE / "data" / "corpus.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote data/corpus.json ({len(out['docs'])} docs, live={not a.offline})")


if __name__ == "__main__":
    main()
