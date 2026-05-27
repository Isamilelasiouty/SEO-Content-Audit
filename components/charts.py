"""
components/charts.py
====================
Reusable Plotly chart components for the SEO Intelligence Dashboard.
All charts use the shared dark theme palette and are optimised for
Streamlit Cloud (low-overhead, no heavy rendering).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ---------------------------------------------------------------------------
# Design tokens — single source of truth for all charts
# ---------------------------------------------------------------------------

PALETTE = {
    "bg":           "#0a0e1a",
    "surface":      "#0f1629",
    "border":       "#1e2d4a",
    "accent":       "#00d4ff",
    "accent2":      "#7c3aed",
    "accent3":      "#10b981",
    "warn":         "#f59e0b",
    "danger":       "#ef4444",
    "text":         "#e2e8f0",
    "text_muted":   "#64748b",
    "grid":         "#1e2d4a",
}

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Mono', 'Courier New', monospace", color=PALETTE["text"], size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    hoverlabel=dict(
        bgcolor=PALETTE["surface"],
        bordercolor=PALETTE["border"],
        font_color=PALETTE["text"],
    ),
)

_AXIS_BASE = dict(
    showgrid=True,
    gridcolor=PALETTE["grid"],
    gridwidth=1,
    zeroline=False,
    tickfont=dict(color=PALETTE["text_muted"], size=11),
    linecolor=PALETTE["border"],
)


def _base_layout(**overrides) -> dict:
    layout = {**_LAYOUT_BASE}
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# 1. Relevance score distribution (histogram)
# ---------------------------------------------------------------------------

def relevance_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of opportunity relevance scores (new suggestions only)."""
    # Support both raw column names and renamed display columns
    if "Type" in df.columns and "Relevance" in df.columns:
        data = df[df["Type"] == "new"]["Relevance"]
    elif "is_duplicate" in df.columns and "relevance_score" in df.columns:
        data = df[~df["is_duplicate"]]["relevance_score"]
    elif "Relevance" in df.columns:
        data = df["Relevance"]
    elif "relevance_score" in df.columns:
        data = df["relevance_score"]
    else:
        return _empty_fig("No relevance data available")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=20,
        marker=dict(
            color=PALETTE["accent"],
            opacity=0.85,
            line=dict(color=PALETTE["bg"], width=1),
        ),
        hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title=dict(text="Relevance Score Distribution", font=dict(size=14, color=PALETTE["text"]))),
        xaxis=dict(**_AXIS_BASE, title=dict(text="Relevance Score", font=dict(color=PALETTE["text_muted"]))),
        yaxis=dict(**_AXIS_BASE, title=dict(text="Opportunities", font=dict(color=PALETTE["text_muted"]))),
        bargap=0.05,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Top PageRank pages (horizontal bar)
# ---------------------------------------------------------------------------

def pagerank_bar(top_pages: list[dict], max_items: int = 10) -> go.Figure:
    """Horizontal bar chart of top PageRank pages."""
    items = top_pages[:max_items]
    if not items:
        return _empty_fig("No PageRank data available")

    labels = [_short_url(p["url"]) for p in items]
    values = [p["pagerank"] for p in items]
    full_urls = [p["url"] for p in items]

    max_v = max(values) or 1
    colours = [
        f"rgba(0,212,255,{0.3 + 0.7 * (v / max_v):.2f})"
        for v in values
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colours, line=dict(color=PALETTE["border"], width=1)),
        customdata=full_urls,
        hovertemplate="<b>%{customdata}</b><br>PageRank: %{x:.6f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title=dict(text="Top Pages by PageRank", font=dict(size=14, color=PALETTE["text"]))),
        xaxis=dict(**_AXIS_BASE, title=dict(text="PageRank Score", font=dict(color=PALETTE["text_muted"]))),
        yaxis=dict(autorange="reversed", tickfont=dict(color=PALETTE["text_muted"], size=10), showgrid=False),
        height=max(260, len(items) * 32),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Link opportunity scatter
# ---------------------------------------------------------------------------

def opportunity_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: semantic overlap (x) vs relevance score (y)."""
    if df.empty:
        return _empty_fig("No opportunities to plot")

    # Support both raw and renamed columns
    if "Type" in df.columns and "Overlap" in df.columns and "Relevance" in df.columns:
        new_df = df[df["Type"] == "new"]
        dup_df = df[df["Type"] == "existing"]
        x_col, y_col = "Overlap", "Relevance"
        src_col = "Source Page" if "Source Page" in df.columns else None
        tgt_col = "Target Page" if "Target Page" in df.columns else None
        anc_col = "Anchor Text" if "Anchor Text" in df.columns else None
    elif "is_duplicate" in df.columns:
        new_df = df[~df["is_duplicate"]]
        dup_df = df[df["is_duplicate"]]
        x_col, y_col = "semantic_overlap", "relevance_score"
        src_col, tgt_col, anc_col = "source_url", "target_url", "anchor_text"
    else:
        new_df = df
        dup_df = pd.DataFrame()
        x_col = "Overlap" if "Overlap" in df.columns else "semantic_overlap"
        y_col = "Relevance" if "Relevance" in df.columns else "relevance_score"
        src_col = tgt_col = anc_col = None

    fig = go.Figure()

    if not new_df.empty and x_col in new_df.columns and y_col in new_df.columns:
        custom = None
        if src_col and tgt_col and anc_col and all(c in new_df.columns for c in [src_col, tgt_col, anc_col]):
            custom = new_df[[src_col, tgt_col, anc_col]].values
            hover = (
                "<b>Anchor:</b> %{customdata[2]}<br>"
                "<b>From:</b> %{customdata[0]}<br>"
                "<b>To:</b> %{customdata[1]}<br>"
                "Overlap: %{x:.3f} | Relevance: %{y:.3f}<extra></extra>"
            )
        else:
            hover = "Overlap: %{x:.3f} | Relevance: %{y:.3f}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=new_df[x_col],
            y=new_df[y_col],
            mode="markers",
            name="New opportunity",
            marker=dict(color=PALETTE["accent"], size=8, opacity=0.75, line=dict(color=PALETTE["bg"], width=1)),
            customdata=custom,
            hovertemplate=hover,
        ))

    if not dup_df.empty and x_col in dup_df.columns and y_col in dup_df.columns:
        fig.add_trace(go.Scatter(
            x=dup_df[x_col],
            y=dup_df[y_col],
            mode="markers",
            name="Existing link",
            marker=dict(color=PALETTE["text_muted"], size=6, opacity=0.4, symbol="x"),
            hovertemplate="Existing link<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(title=dict(text="Opportunities: Overlap vs Relevance", font=dict(size=14, color=PALETTE["text"]))),
        xaxis=dict(**_AXIS_BASE, title=dict(text="Semantic Overlap", font=dict(color=PALETTE["text_muted"])), range=[-0.02, 1.02]),
        yaxis=dict(**_AXIS_BASE, title=dict(text="Relevance Score", font=dict(color=PALETTE["text_muted"])), range=[-0.02, 1.02]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["border"], font=dict(color=PALETTE["text_muted"], size=11)),
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Opportunities per source page (bar)
# ---------------------------------------------------------------------------

def opps_per_page_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart — pages with most new linking opportunities."""
    if df.empty:
        return _empty_fig("No data")

    # Support both raw and renamed columns
    if "Type" in df.columns and "Source Page" in df.columns:
        new_df = df[df["Type"] == "new"]
        src_col = "Source Page"
    elif "is_duplicate" in df.columns and "source_url" in df.columns:
        new_df = df[~df["is_duplicate"]]
        src_col = "source_url"
    else:
        return _empty_fig("No new opportunities found")

    if new_df.empty:
        return _empty_fig("No new opportunities found")

    counts = (
        new_df.groupby(src_col)
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="count")
    )
    counts["label"] = counts[src_col].apply(_short_url)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["label"],
        y=counts["count"],
        marker=dict(color=PALETTE["accent2"], opacity=0.85, line=dict(color=PALETTE["bg"], width=1)),
        customdata=counts[src_col].values,
        hovertemplate="<b>%{customdata}</b><br>Opportunities: %{y}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title=dict(text="New Opportunities by Source Page", font=dict(size=14, color=PALETTE["text"]))),
        xaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False, linecolor=PALETTE["border"], tickangle=-35, tickfont=dict(color=PALETTE["text_muted"], size=9)),
        yaxis=dict(**_AXIS_BASE, title=dict(text="Count", font=dict(color=PALETTE["text_muted"]))),
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Graph health donut
# ---------------------------------------------------------------------------

def graph_health_donut(metrics: dict) -> go.Figure:
    """Donut chart showing page health breakdown."""
    total   = metrics.get("total_pages", 0)
    orphans = metrics.get("orphan_count", 0)
    sinks   = metrics.get("sink_count", 0)
    healthy = max(total - orphans - sinks, 0)

    if total == 0:
        return _empty_fig("No graph data")

    fig = go.Figure(go.Pie(
        labels=["Healthy", "Orphan pages", "Sink pages"],
        values=[healthy, orphans, sinks],
        hole=0.62,
        marker=dict(colors=[PALETTE["accent3"], PALETTE["danger"], PALETTE["warn"]], line=dict(color=PALETTE["bg"], width=2)),
        textfont=dict(color=PALETTE["text"], size=11),
        hovertemplate="%{label}: %{value} pages (%{percent})<extra></extra>",
    ))

    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:10px'>pages</span>",
        x=0.5, y=0.5,
        font=dict(size=16, color=PALETTE["text"]),
        showarrow=False,
    )

    fig.update_layout(
        **_base_layout(title=dict(text="Link Graph Health", font=dict(size=14, color=PALETTE["text"]))),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["text_muted"], size=11), orientation="v"),
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_url(url: str, max_len: int = 40) -> str:
    label = str(url).replace("https://", "").replace("http://", "")
    return label if len(label) <= max_len else "…" + label[-(max_len - 1):]


def _empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=PALETTE["text_muted"], size=13))
    fig.update_layout(**_base_layout(), height=220)
    return fig
