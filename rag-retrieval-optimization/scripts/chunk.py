#!/usr/bin/env python3
"""Split text into fixed-size character chunks with overlap.

Default follows the skill's source rule: 200-character chunks, the last 60
characters of each chunk repeated at the start of the next. Sizes are counted
in Unicode code points (Python len), not tokens, bytes, or grapheme clusters:
normalize input to NFC first if it may contain decomposed Hangul jamo. Line
endings are preserved as-is so start/end offsets index the original text.
When len(text) mod (size - overlap) is small, the final chunk is mostly
overlap; that is what the fixed rule produces.

Usage:
    python3 chunk.py [--size 200] [--overlap 60] [FILE]
Reads FILE (or stdin) and writes one JSON object per line:
    {"index": 0, "start": 0, "end": 200, "text": "..."}
"""
import argparse
import json
import sys


def chunk(text, size, overlap):
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    step = size - overlap
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        out.append({"index": len(out), "start": start, "end": end, "text": text[start:end]})
        if end == n:
            break
        start += step
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="input text file (default: stdin)")
    p.add_argument("--size", type=int, default=200, help="chunk size in characters (default 200)")
    p.add_argument("--overlap", type=int, default=60, help="overlap in characters (default 60)")
    a = p.parse_args()
    if a.size <= 0 or not 0 <= a.overlap < a.size:
        p.error("require size > 0 and 0 <= overlap < size")
    if a.file:
        with open(a.file, encoding="utf-8", newline="") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    for c in chunk(text, a.size, a.overlap):
        sys.stdout.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
