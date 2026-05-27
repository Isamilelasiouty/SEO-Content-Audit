"""
seo_engine/internal_links.py
=============================
Core Internal Linking Engine for the SEO Intelligence Dashboard.

Classes:
    PageNode          — represents a single crawled page
    LinkOpportunity   — represents a suggested or existing internal link
    InternalLinkEngine — main engine: builds graph, computes PageRank,
                         scores opportunities, returns graph metrics

Public helper:
    build_pages_from_crawl_data(records) -> list[PageNode]
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageNode:
    """Represents a single crawled page."""
    url: str
    title: str = ""
    body_text: str = ""
    outlinks: list[str] = field(default_factory=list)

    # Computed at engine init
    keywords: list[str] = field(default_factory=list)
    tfidf_vector: dict[str, float] = field(default_factory=dict)


@dataclass
class LinkOpportunity:
    """A suggested (or existing) internal link between two pages."""
    source_url: str
    target_url: str
    anchor_text: str
    relevance_score: float
    semantic_overlap: float
    pagerank_target: float
    reason: str
    is_duplicate: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Text / keyword utilities
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "we", "you", "he",
    "she", "they", "them", "their", "our", "your", "my", "his", "her",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "also",
    "just", "more", "most", "such", "than", "then", "too", "very", "s",
    "page", "site", "website", "web", "click", "here", "read", "learn",
    "about", "how", "what", "when", "where", "why", "which", "who",
})


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords & short tokens."""
    tokens = re.findall(r"[a-z][a-z0-9\-]*[a-z0-9]|[a-z]{2,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def _extract_keywords(page: PageNode, top_n: int = 30) -> list[str]:
    """Return top_n keywords from title + body_text by term frequency."""
    text = f"{page.title} {page.title} {page.body_text}"  # title weighted ×2
    tokens = _tokenise(text)
    if not tokens:
        return []
    freq: dict[str, int] = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    return [kw for kw, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]


def _build_tfidf_vector(page: PageNode, idf: dict[str, float]) -> dict[str, float]:
    """Build a TF-IDF-like vector for a page given pre-computed IDF scores."""
    text = f"{page.title} {page.title} {page.body_text}"
    tokens = _tokenise(text)
    if not tokens:
        return {}
    tf: dict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1.0
    total = len(tokens)
    return {term: (count / total) * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(vec_a.get(k, 0.0) * v for k, v in vec_b.items())
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _keyword_overlap(kw_a: list[str], kw_b: list[str]) -> float:
    """Jaccard overlap between two keyword lists."""
    set_a, set_b = set(kw_a), set(kw_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _suggest_anchor(source: PageNode, target: PageNode) -> str:
    """
    Suggest anchor text for a link from source → target.
    Priority: shared n-gram → top target keyword → cleaned target title.
    """
    src_kw = set(source.keywords)
    tgt_kw = target.keywords

    # Shared keyword (prefer longer ones)
    shared = sorted(src_kw & set(tgt_kw), key=len, reverse=True)
    if shared:
        return shared[0].replace("-", " ").title()

    # Top target keyword
    if tgt_kw:
        return tgt_kw[0].replace("-", " ").title()

    # Cleaned title fallback
    title = target.title.strip()
    if title:
        return title[:60]

    # URL slug fallback
    slug = target.url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return slug.title() if slug else target.url


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class InternalLinkEngine:
    """
    Analyses a corpus of pages and surfaces internal linking opportunities.

    Parameters
    ----------
    pages : list[PageNode]
        All crawled pages.
    min_relevance : float
        Minimum relevance_score for an opportunity to be returned.
    max_suggestions_per_page : int
        Maximum new link suggestions emitted per source page.
    """

    # Relevance formula weights
    _W_COSINE   = 0.55
    _W_OVERLAP  = 0.35
    _W_PAGERANK = 0.10

    def __init__(
        self,
        pages: list[PageNode],
        min_relevance: float = 0.10,
        max_suggestions_per_page: int = 5,
    ) -> None:
        self.pages = pages
        self.min_relevance = min_relevance
        self.max_suggestions_per_page = max_suggestions_per_page

        self._page_map: dict[str, PageNode] = {p.url: p for p in pages}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._pagerank: dict[str, float] = {}

        self._prepare()

    # ── Initialisation ────────────────────────────────────────────────────

    def _prepare(self) -> None:
        """Build graph, compute IDF, keywords, TF-IDF vectors, PageRank."""
        # 1. Build directed graph
        all_urls = set(self._page_map)
        for page in self.pages:
            self._graph.add_node(page.url)
            for link in page.outlinks:
                norm = _normalise_url(link)
                if norm in all_urls and norm != page.url:
                    self._graph.add_edge(page.url, norm)

        # 2. PageRank
        if self._graph.number_of_nodes() > 0:
            try:
                self._pagerank = nx.pagerank(self._graph, alpha=0.85, max_iter=200)
            except nx.PowerIterationFailedConvergence:
                self._pagerank = {n: 1.0 / len(self._graph) for n in self._graph.nodes}
        else:
            self._pagerank = {}

        # 3. IDF over corpus
        doc_freq: dict[str, int] = defaultdict(int)
        for page in self.pages:
            tokens = set(_tokenise(f"{page.title} {page.body_text}"))
            for t in tokens:
                doc_freq[t] += 1
        N = len(self.pages) or 1
        idf = {t: math.log((N + 1) / (df + 1)) + 1.0 for t, df in doc_freq.items()}

        # 4. Keywords + TF-IDF vectors per page
        for page in self.pages:
            page.keywords = _extract_keywords(page)
            page.tfidf_vector = _build_tfidf_vector(page, idf)

    # ── Public API ────────────────────────────────────────────────────────

    def analyze(self) -> list[LinkOpportunity]:
        """
        Return all link opportunities (new + existing) sorted by relevance desc.
        """
        opportunities: list[LinkOpportunity] = []

        for source in self.pages:
            existing_targets = set(
                _normalise_url(u) for u in source.outlinks
                if _normalise_url(u) in self._page_map
            )
            new_count = 0
            candidates: list[LinkOpportunity] = []

            for target in self.pages:
                if target.url == source.url:
                    continue

                cosine  = _cosine_similarity(source.tfidf_vector, target.tfidf_vector)
                overlap = _keyword_overlap(source.keywords, target.keywords)
                pr      = self._pagerank.get(target.url, 0.0)

                # Normalise PageRank bonus (0-1 scale)
                max_pr = max(self._pagerank.values()) if self._pagerank else 1.0
                pr_norm = pr / max_pr if max_pr > 0 else 0.0

                relevance = (
                    self._W_COSINE   * cosine +
                    self._W_OVERLAP  * overlap +
                    self._W_PAGERANK * pr_norm
                )

                if relevance < self.min_relevance:
                    continue

                is_dup = _normalise_url(target.url) in existing_targets
                anchor = _suggest_anchor(source, target)
                reason = _build_reason(cosine, overlap, pr_norm, is_dup)

                candidates.append(LinkOpportunity(
                    source_url=source.url,
                    target_url=target.url,
                    anchor_text=anchor,
                    relevance_score=round(relevance, 6),
                    semantic_overlap=round(overlap, 6),
                    pagerank_target=round(pr, 8),
                    reason=reason,
                    is_duplicate=is_dup,
                ))

            # Sort: new first, then by relevance desc
            candidates.sort(key=lambda o: (o.is_duplicate, -o.relevance_score))

            for opp in candidates:
                if not opp.is_duplicate:
                    if new_count >= self.max_suggestions_per_page:
                        continue
                    new_count += 1
                opportunities.append(opp)

        # Global sort: new opportunities first, then by relevance
        opportunities.sort(key=lambda o: (o.is_duplicate, -o.relevance_score))
        return opportunities

    def graph_metrics(self) -> dict:
        """
        Return a dict of graph-level metrics consumed by the dashboard UI.
        """
        G = self._graph
        nodes = list(G.nodes)
        n = len(nodes)

        # Orphans: nodes with in-degree == 0 AND out-degree == 0
        orphan_pages = [
            node for node in nodes
            if G.in_degree(node) == 0 and G.out_degree(node) == 0
        ]

        # Sinks: nodes with out-degree == 0 (but have inlinks)
        sink_pages = [
            node for node in nodes
            if G.out_degree(node) == 0 and G.in_degree(node) > 0
        ]

        # Average outlinks per page
        avg_outlinks = (G.number_of_edges() / n) if n > 0 else 0.0

        # Average clustering coefficient (undirected proxy)
        try:
            avg_clustering = nx.average_clustering(G.to_undirected())
        except Exception:
            avg_clustering = 0.0

        # Top PageRank pages
        sorted_pr = sorted(self._pagerank.items(), key=lambda x: -x[1])
        top_pagerank_pages = [
            {"url": url, "pagerank": pr} for url, pr in sorted_pr[:20]
        ]

        return {
            "total_pages":               n,
            "total_edges":               G.number_of_edges(),
            "orphan_count":              len(orphan_pages),
            "orphan_pages":              orphan_pages,
            "sink_count":                len(sink_pages),
            "sink_pages":                sink_pages,
            "avg_outlinks_per_page":     round(avg_outlinks, 2),
            "avg_clustering_coefficient": round(avg_clustering, 4),
            "top_pagerank_pages":        top_pagerank_pages,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public helper
# ─────────────────────────────────────────────────────────────────────────────

def build_pages_from_crawl_data(records: list[dict]) -> list[PageNode]:
    """
    Convert a list of raw crawl record dicts into PageNode objects.

    Expected keys per record: url, title (opt), body_text (opt), outlinks (opt list).
    """
    pages = []
    for rec in records:
        url = rec.get("url", "").strip()
        if not url:
            continue
        pages.append(PageNode(
            url=url,
            title=rec.get("title", "") or "",
            body_text=rec.get("body_text", "") or "",
            outlinks=rec.get("outlinks", []) or [],
        ))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_url(url: str) -> str:
    """Strip trailing slash and fragment for deduplication."""
    url = url.strip().split("#")[0].rstrip("/")
    return url


def _build_reason(cosine: float, overlap: float, pr_norm: float, is_dup: bool) -> str:
    """Human-readable reason string for an opportunity."""
    if is_dup:
        return "Already linked"
    parts = []
    if cosine >= 0.4:
        parts.append("high content similarity")
    elif cosine >= 0.2:
        parts.append("moderate content similarity")
    if overlap >= 0.3:
        parts.append("strong keyword overlap")
    elif overlap >= 0.15:
        parts.append("keyword overlap")
    if pr_norm >= 0.7:
        parts.append("high-authority target")
    if not parts:
        parts.append("topical relevance")
    return ", ".join(parts).capitalize()
