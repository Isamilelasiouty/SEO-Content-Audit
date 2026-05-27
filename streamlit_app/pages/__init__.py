"""
utils/link_utils.py
===================
Shared helper functions used across the SEO engine.
"""
from __future__ import annotations
import re
from collections import defaultdict


_STOPWORDS = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","it","its","this",
    "that","not","no","so","just","more","most","also","very","how","what",
    "when","where","why","who","page","site","web","click","here","read",
})


def tokenise(text: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9\-]*[a-z0-9]|[a-z]{2,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def normalise_url(url: str) -> str:
    return url.strip().split("#")[0].rstrip("/")


def keyword_overlap(kw_a: list[str], kw_b: list[str]) -> float:
    set_a, set_b = set(kw_a), set(kw_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
