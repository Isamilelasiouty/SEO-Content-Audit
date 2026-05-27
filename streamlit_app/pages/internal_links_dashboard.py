"""
streamlit_app/pages/internal_links_dashboard.py
================================================
Internal Linking Opportunity Dashboard

Entry point:
    streamlit run streamlit_app/pages/internal_links_dashboard.py
    (or place under pages/ in a multi-page Streamlit app)
"""

from __future__ import annotations

import io
import sys
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path resolution — works whether run directly or as a sub-page ──────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from seo_engine.internal_links import (
    InternalLinkEngine,
    PageNode,
    build_pages_from_crawl_data,
)
from components.charts import (
    relevance_histogram,
    pagerank_bar,
    opportunity_scatter,
    opps_per_page_bar,
    graph_health_donut,
)
from components.tables import (
    inject_css,
    render_kpi_row,
    section_header,
    build_opportunities_df,
    build_orphans_df,
    build_pagerank_df,
    opportunities_column_config,
    opportunities_download_btn,
    orphans_download_btn,
    df_to_csv_bytes,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Internal Links · SEO Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _parse_csv(uploaded_file) -> list[dict]:
    """Parse a CSV file into crawl records."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return []

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Flexible column mapping
    col_map = {
        "url":        ["url", "address", "page", "page_url", "link"],
        "title":      ["title", "page_title", "name"],
        "body_text":  ["body_text", "content", "text", "body", "copy"],
        "outlinks":   ["outlinks", "out_links", "internal_links", "links_out"],
        "inlinks":    ["inlinks", "in_links", "links_in", "backlinks"],
    }

    def _find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    url_col = _find_col(col_map["url"])
    if not url_col:
        st.error("CSV must contain a 'url' (or 'address'/'page') column.")
        return []

    records = []
    for _, row in df.iterrows():
        outlinks_raw = ""
        ol_col = _find_col(col_map["outlinks"])
        if ol_col:
            outlinks_raw = str(row.get(ol_col, ""))

        outlinks = [u.strip() for u in outlinks_raw.split("|") if u.strip().startswith("http")]

        records.append({
            "url":       str(row[url_col]).strip(),
            "title":     str(row.get(_find_col(col_map["title"]) or "", "")).strip(),
            "body_text": str(row.get(_find_col(col_map["body_text"]) or "", "")).strip(),
            "outlinks":  outlinks,
        })

    return [r for r in records if r["url"].startswith("http")]


def _parse_sitemap(uploaded_file) -> list[dict]:
    """Extract URLs from a sitemap.xml file."""
    try:
        content = uploaded_file.read()
        root = ET.fromstring(content)
    except Exception as e:
        st.error(f"Could not parse sitemap XML: {e}")
        return []

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []
    for loc in root.findall(".//sm:loc", ns):
        text = (loc.text or "").strip()
        if text.startswith("http"):
            urls.append(text)

    if not urls:
        # Try without namespace
        for loc in root.iter("loc"):
            text = (loc.text or "").strip()
            if text.startswith("http"):
                urls.append(text)

    return [{"url": u, "title": "", "body_text": "", "outlinks": []} for u in urls]


def _parse_plain_urls(text: str) -> list[dict]:
    """Parse newline-separated URLs from a text area."""
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("http")]
    return [{"url": u, "title": "", "body_text": "", "outlinks": []} for u in lines]


@st.cache_data(show_spinner=False)
def _run_engine(
    records_json: str,
    min_relevance: float,
    max_suggestions: int,
) -> tuple[dict, list[dict]]:
    """
    Cached engine run.  Accepts JSON string of records so Streamlit can hash it.
    Returns (graph_metrics_dict, opportunities_list_of_dicts).
    """
    import json
    records = json.loads(records_json)
    pages = build_pages_from_crawl_data(records)
    engine = InternalLinkEngine(
        pages,
        min_relevance=min_relevance,
        max_suggestions_per_page=max_suggestions,
    )
    opps = engine.analyze()
    return engine.graph_metrics(), [
        {
            "source_url":       o.source_url,
            "target_url":       o.target_url,
            "anchor_text":      o.anchor_text,
            "relevance_score":  o.relevance_score,
            "semantic_overlap": o.semantic_overlap,
            "pagerank_target":  o.pagerank_target,
            "reason":           o.reason,
            "is_duplicate":     o.is_duplicate,
        }
        for o in opps
    ]


def _demo_records() -> list[dict]:
    """Built-in demo dataset so the dashboard works without a file upload."""
    return [
        {"url": "https://demo.site/seo-guide",        "title": "The Complete SEO Guide 2025",
         "body_text": "Search engine optimisation improves organic visibility. Keyword research on-page optimisation and link building are core SEO pillars. Technical SEO ensures crawlability indexation and site speed.",
         "outlinks": ["https://demo.site/keyword-research"]},
        {"url": "https://demo.site/keyword-research",  "title": "Keyword Research Tutorial",
         "body_text": "Keyword research identifies search terms your audience uses. Use tools to find high-volume low-competition keywords. Long-tail keywords drive targeted organic traffic.",
         "outlinks": []},
        {"url": "https://demo.site/link-building",     "title": "Link Building Strategies",
         "body_text": "Link building acquires backlinks from external sites. Guest posting broken link building and digital PR are effective strategies. Domain authority improves with quality backlinks.",
         "outlinks": ["https://demo.site/seo-guide"]},
        {"url": "https://demo.site/content-marketing", "title": "Content Marketing Strategy",
         "body_text": "Content marketing creates valuable content to attract audiences. A content calendar schedules blog posts. Great content supports SEO by earning backlinks naturally.",
         "outlinks": []},
        {"url": "https://demo.site/technical-seo",     "title": "Technical SEO Checklist",
         "body_text": "Technical SEO covers crawlability site speed Core Web Vitals and structured data. Fix crawl errors in Search Console. Improve page speed for better rankings.",
         "outlinks": ["https://demo.site/seo-guide", "https://demo.site/keyword-research"]},
        {"url": "https://demo.site/on-page-seo",       "title": "On-Page SEO Best Practices",
         "body_text": "On-page SEO optimises individual pages for target keywords. Use keywords in title tags meta descriptions and headings. Internal linking distributes PageRank across the site.",
         "outlinks": ["https://demo.site/keyword-research"]},
        {"url": "https://demo.site/seo-tools",         "title": "Best SEO Tools 2025",
         "body_text": "Ahrefs SEMrush and Moz are leading SEO tools. Use them for keyword research backlink analysis and rank tracking. Google Search Console is free and essential.",
         "outlinks": []},
        {"url": "https://demo.site/python-tutorial",   "title": "Python for Beginners",
         "body_text": "Python is a high-level programming language. Learn variables functions loops and data structures. Python is popular for data science web development and automation.",
         "outlinks": []},
        {"url": "https://demo.site/django-guide",      "title": "Django Web Framework Tutorial",
         "body_text": "Django is a Python web framework. Build web applications using models views and templates. Django includes an ORM for database management and a powerful admin interface.",
         "outlinks": ["https://demo.site/python-tutorial"]},
        {"url": "https://demo.site/isolated-page",     "title": "Isolated Landing Page",
         "body_text": "This page has no inlinks or outlinks making it an orphan in the site graph.",
         "outlinks": []},
    ]


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔗 Internal Links")
    st.markdown("---")

    # ── Data source ──────────────────────────────────────────────────────
    st.markdown("### Data Source")
    data_source = st.radio(
        "Input type",
        ["Demo dataset", "Upload CSV", "Upload Sitemap XML", "Paste URLs"],
        label_visibility="collapsed",
    )

    records: list[dict] = []
    upload_error = False

    if data_source == "Demo dataset":
        records = _demo_records()
        st.caption(f"Using {len(records)}-page demo corpus")

    elif data_source == "Upload CSV":
        f = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        st.caption("Columns: url, title, body_text, outlinks (pipe-separated)")
        if f:
            records = _parse_csv(f)
            if records:
                st.success(f"{len(records)} URLs loaded")

    elif data_source == "Upload Sitemap XML":
        f = st.file_uploader("sitemap.xml", type=["xml"], label_visibility="collapsed")
        if f:
            records = _parse_sitemap(f)
            if records:
                st.success(f"{len(records)} URLs from sitemap")

    elif data_source == "Paste URLs":
        raw = st.text_area("One URL per line", height=160, placeholder="https://example.com/page-1\nhttps://example.com/page-2")
        if raw.strip():
            records = _parse_plain_urls(raw)
            if records:
                st.caption(f"{len(records)} URLs parsed")

    st.markdown("---")

    # ── Engine settings ───────────────────────────────────────────────────
    st.markdown("### Engine Settings")
    min_relevance = st.slider(
        "Min relevance score", 0.0, 0.8, 0.10, 0.01,
        help="Filter out opportunities below this relevance threshold",
    )
    max_suggestions = st.slider(
        "Max suggestions per page", 1, 20, 5,
        help="Cap on new link suggestions per source page",
    )

    st.markdown("---")
    st.markdown("### View Filters")
    show_existing = st.checkbox("Show existing links", value=False,
                                help="Include already-linked pairs in the table")

    st.markdown("---")
    st.caption("SEO Intelligence Dashboard · Internal Links Module")


# ════════════════════════════════════════════════════════════════════════════
# Page header
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="dash-header">'
    '<h1>Internal Link Intelligence</h1>'
    '<p>Semantic opportunity detection · Anchor text suggestions · Graph health analysis</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Empty state ──────────────────────────────────────────────────────────────
if not records:
    st.info("👈  Choose a data source in the sidebar to get started, or select **Demo dataset** for a quick preview.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# Run engine (cached)
# ════════════════════════════════════════════════════════════════════════════

import json as _json

with st.spinner("Analysing link graph…"):
    try:
        metrics, opps_raw = _run_engine(
            records_json=_json.dumps(records, ensure_ascii=False),
            min_relevance=min_relevance,
            max_suggestions=max_suggestions,
        )
    except Exception as e:
        st.error(f"Engine error: {e}")
        st.stop()

# Build DataFrames
all_opps_df  = build_opportunities_df(opps_raw)
new_opps_df  = all_opps_df[all_opps_df["Type"] == "new"]  if not all_opps_df.empty else pd.DataFrame()
dup_opps_df  = all_opps_df[all_opps_df["Type"] == "existing"] if not all_opps_df.empty else pd.DataFrame()
orphans_df   = build_orphans_df(metrics.get("orphan_pages", []))
pagerank_df  = build_pagerank_df(metrics.get("top_pagerank_pages", []))


# ════════════════════════════════════════════════════════════════════════════
# KPI row
# ════════════════════════════════════════════════════════════════════════════

render_kpi_row([
    {"label": "Total Pages",          "value": metrics.get("total_pages", 0),
     "sub": "in corpus", "color": "#00d4ff"},
    {"label": "New Opportunities",    "value": len(new_opps_df),
     "sub": "suggested links", "color": "#7c3aed"},
    {"label": "Existing Links",       "value": len(dup_opps_df),
     "sub": "already linked", "color": "#64748b"},
    {"label": "Orphan Pages",         "value": metrics.get("orphan_count", 0),
     "sub": "no in/out links", "color": "#ef4444"},
    {"label": "Sink Pages",           "value": metrics.get("sink_count", 0),
     "sub": "no outlinks", "color": "#f59e0b"},
    {"label": "Avg Outlinks",         "value": metrics.get("avg_outlinks_per_page", 0),
     "sub": "per page", "color": "#10b981"},
])


# ════════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════════

tab_opps, tab_charts, tab_orphans, tab_pagerank, tab_raw = st.tabs([
    "🔗 Opportunities",
    "📊 Charts",
    "🚨 Orphan Pages",
    "🏆 PageRank",
    "⚙ Raw Data",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Opportunities
# ────────────────────────────────────────────────────────────────────────────
with tab_opps:
    if all_opps_df.empty:
        st.info("No opportunities found. Try lowering the min relevance score.")
    else:
        # ── Filters ──────────────────────────────────────────────────────
        col_search, col_min_rel, col_type = st.columns([3, 2, 2])

        with col_search:
            search_q = st.text_input("🔍 Search URLs or anchor text", placeholder="e.g. seo, guide, keyword…", label_visibility="collapsed")

        with col_min_rel:
            rel_filter = st.slider("Min relevance", 0.0, 1.0, float(min_relevance), 0.01, key="opp_rel_filter", label_visibility="collapsed")

        with col_type:
            type_opts = ["New opportunities", "All (incl. existing)"] if show_existing else ["New opportunities"]
            type_filter = st.selectbox("Type", type_opts, label_visibility="collapsed")

        # Apply filters
        display_df = all_opps_df.copy()

        if type_filter == "New opportunities":
            display_df = display_df[display_df["Type"] == "new"]

        display_df = display_df[display_df["Relevance"] >= rel_filter]

        if search_q:
            q = search_q.lower()
            mask = (
                display_df["Source Page"].str.lower().str.contains(q, na=False) |
                display_df["Target Page"].str.lower().str.contains(q, na=False) |
                display_df["Anchor Text"].str.lower().str.contains(q, na=False)
            )
            display_df = display_df[mask]

        # ── Result count + export ─────────────────────────────────────────
        rcol1, rcol2 = st.columns([5, 1])
        with rcol1:
            section_header(f"{len(display_df)} link opportunities")
        with rcol2:
            opportunities_download_btn(display_df)

        # ── Table ─────────────────────────────────────────────────────────
        st.dataframe(
            display_df,
            column_config=opportunities_column_config(),
            use_container_width=True,
            hide_index=True,
            height=min(600, max(200, len(display_df) * 36 + 60)),
        )


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Charts
# ────────────────────────────────────────────────────────────────────────────
with tab_charts:
    if all_opps_df.empty:
        st.info("No data to visualise.")
    else:
        row1_l, row1_r = st.columns(2)

        with row1_l:
            st.plotly_chart(relevance_histogram(all_opps_df), use_container_width=True, config={"displayModeBar": False})

        with row1_r:
            st.plotly_chart(graph_health_donut(metrics), use_container_width=True, config={"displayModeBar": False})

        row2_l, row2_r = st.columns(2)

        with row2_l:
            st.plotly_chart(opportunity_scatter(all_opps_df), use_container_width=True, config={"displayModeBar": False})

        with row2_r:
            st.plotly_chart(opps_per_page_bar(all_opps_df), use_container_width=True, config={"displayModeBar": False})

        # PageRank bar full-width
        if metrics.get("top_pagerank_pages"):
            st.plotly_chart(pagerank_bar(metrics["top_pagerank_pages"]), use_container_width=True, config={"displayModeBar": False})


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Orphan Pages
# ────────────────────────────────────────────────────────────────────────────
with tab_orphans:
    section_header(f"{len(orphans_df)} orphan pages detected")

    if orphans_df.empty:
        st.success("✅ No orphan pages found — every page has at least one link connection.")
    else:
        st.markdown(
            '<p style="color:var(--text-muted,#64748b);font-size:0.78rem;margin-bottom:0.8rem">'
            "Orphan pages have zero inlinks <em>and</em> zero outlinks. "
            "They are invisible to internal PageRank flow and may not be crawled reliably."
            "</p>",
            unsafe_allow_html=True,
        )

        # ── Search ───────────────────────────────────────────────────────
        orp_search = st.text_input("🔍 Filter orphans", placeholder="e.g. /blog/…", label_visibility="collapsed", key="orp_search")
        filtered_orphans = orphans_df.copy()
        if orp_search:
            filtered_orphans = filtered_orphans[
                filtered_orphans["Orphan Page"].str.lower().str.contains(orp_search.lower(), na=False)
            ]

        exp_col, dl_col = st.columns([5, 1])
        with exp_col:
            pass
        with dl_col:
            orphans_download_btn(filtered_orphans)

        st.dataframe(
            filtered_orphans,
            column_config={"Orphan Page": st.column_config.LinkColumn("Orphan Page")},
            use_container_width=True,
            hide_index=True,
            height=min(500, max(150, len(filtered_orphans) * 36 + 60)),
        )

    # ── Sink pages ───────────────────────────────────────────────────────
    sink_urls = metrics.get("sink_pages", [])
    if sink_urls:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header(f"{len(sink_urls)} sink pages (no outlinks)")
        st.markdown(
            '<p style="color:var(--text-muted,#64748b);font-size:0.78rem;margin-bottom:0.8rem">'
            "Sink pages receive inlinks but pass no PageRank onward — they trap link equity."
            "</p>",
            unsafe_allow_html=True,
        )
        sink_df = pd.DataFrame({"Sink Page": sink_urls})
        st.dataframe(
            sink_df,
            column_config={"Sink Page": st.column_config.LinkColumn("Sink Page")},
            use_container_width=True,
            hide_index=True,
            height=min(400, max(120, len(sink_df) * 36 + 60)),
        )


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — PageRank
# ────────────────────────────────────────────────────────────────────────────
with tab_pagerank:
    section_header("Top Pages by PageRank Authority")

    if pagerank_df.empty:
        st.info("No PageRank data available.")
    else:
        st.markdown(
            '<p style="color:var(--text-muted,#64748b);font-size:0.78rem;margin-bottom:0.8rem">'
            "PageRank reflects the relative authority of each page based on the current internal link structure. "
            "High-PR pages are ideal link targets for distributing equity to weaker pages."
            "</p>",
            unsafe_allow_html=True,
        )

        pr_col1, pr_col2 = st.columns([1, 1])

        with pr_col1:
            st.dataframe(
                pagerank_df,
                column_config={
                    "Page":     st.column_config.LinkColumn("Page"),
                    "PageRank": st.column_config.NumberColumn("PageRank", format="%.6f"),
                },
                use_container_width=True,
                hide_index=True,
            )

        with pr_col2:
            st.plotly_chart(
                pagerank_bar(metrics.get("top_pagerank_pages", []), max_items=10),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Metrics row
        cluster_coeff = metrics.get("avg_clustering_coefficient", 0)
        st.markdown("<br>", unsafe_allow_html=True)
        render_kpi_row([
            {"label": "Total Edges",           "value": metrics.get("total_edges", 0),     "color": "#00d4ff"},
            {"label": "Avg Clustering Coeff",  "value": f"{cluster_coeff:.4f}",             "color": "#7c3aed",
             "sub": "graph density proxy"},
            {"label": "Avg Outlinks / Page",   "value": metrics.get("avg_outlinks_per_page", 0), "color": "#10b981"},
        ])


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — Raw Data
# ────────────────────────────────────────────────────────────────────────────
with tab_raw:
    section_header("Raw Input Records")
    raw_df = pd.DataFrame(records)

    raw_search = st.text_input("🔍 Filter", placeholder="Search URLs…", label_visibility="collapsed", key="raw_search")
    if raw_search and "url" in raw_df.columns:
        raw_df = raw_df[raw_df["url"].str.lower().str.contains(raw_search.lower(), na=False)]

    dl_col, _ = st.columns([1, 5])
    with dl_col:
        st.download_button(
            "⬇ Export raw CSV",
            data=df_to_csv_bytes(raw_df),
            file_name="raw_crawl_data.csv",
            mime="text/csv",
        )

    st.dataframe(raw_df, use_container_width=True, hide_index=True)

    if not all_opps_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("All Opportunities (raw)")
        st.dataframe(all_opps_df, use_container_width=True, hide_index=True)
