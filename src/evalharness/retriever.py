"""Okapi BM25 retriever in plain Python, so the harness has no required dependencies."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Chunk

STOPWORDS = set(
    "a an and are as at be by can do does for from how i in is it many may much my of on or our "
    "should the their there this to was we what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$@][a-z0-9$.@-]*[a-z0-9]|[a-z0-9]", text.lower().replace("'s", "")) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.title + " " + c.text) for c in chunks]
        self.avg_len = sum(len(d) for d in self.docs) / len(self.docs)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = tokenize(query)
        hits = []
        for chunk, doc in zip(self.chunks, self.docs):
            tf = Counter(doc)
            score = 0.0
            for t in terms:
                if t in tf:
                    num = tf[t] * (self.k1 + 1)
                    den = tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
                    score += self.idf[t] * num / den
            hits.append(Hit(chunk, score))
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return [h for h in hits[:k] if h.score > 0]
