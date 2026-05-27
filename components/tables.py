"""
components/tables.py
====================
Styled, filterable table helpers for the SEO Intelligence Dashboard.
Returns DataFrames ready for st.dataframe() / st.data_editor(),
plus pre-built column configs and CSS injection utilities.
"""

from __future__ import annotations

import io
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# CSS injection — call once per page load
# ---------------------------------------------------------------------------

_DASHBOARD_CSS = """
<style>
/* ── Google Fonts ────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root palette ────────────────────────────────────── */
:root {
  --bg:            #0a0e1a;
  --surface:       #0f1629;
  --surface2:      #131d35;
  --border:        #1e2d4a;
  --accent:        #00d4ff;
  --accent2:       #7c3aed;
  --accent3:       #10b981;
  --warn:          #f59e0b;
  --danger:        #ef4444;
  --text:          #e2e8f0;
  --text-muted:    #64748b;
  --text-faint:    #334155;
  --radius:        8px;
}

/* ── App shell ───────────────────────────────────────── */
.stApp { background: var(--bg); }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.main .block-container { padding-top: 1.5rem; max-width: 1280px; }

/* ── Typography ──────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'DM Mono', 'Courier New', monospace; color: var(--text); }
h1,h2,h3,h4,h5 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

/* ── Page header ─────────────────────────────────────── */
.dash-header {
  padding: 1.5rem 0 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.dash-header h1 {
  font-size: 1.9rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 0.2rem;
}
.dash-header p { color: var(--text-muted); font-size: 0.82rem; margin: 0; }

/* ── KPI cards ───────────────────────────────────────── */
.kpi-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.5rem; }
.kpi-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  flex: 1 1 140px;
  min-width: 130px;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--accent-color, var(--accent));
}
.kpi-label { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700; color: var(--text); line-height: 1; }
.kpi-sub   { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.25rem; }

/* ── Section headers ─────────────────────────────────── */
.section-head {
  font-family: 'Syne', sans-serif;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent);
  padding: 0.6rem 0 0.4rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.75rem;
}

/* ── Tag pills ───────────────────────────────────────── */
.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.04em;
}
.tag-new      { background: rgba(0,212,255,0.12); color: var(--accent);  border: 1px solid rgba(0,212,255,0.3); }
.tag-existing { background: rgba(100,116,139,0.15); color: var(--text-muted); border: 1px solid var(--border); }
.tag-orphan   { background: rgba(239,68,68,0.12);  color: var(--danger); border: 1px solid rgba(239,68,68,0.3); }
.tag-sink     { background: rgba(245,158,11,0.12); color: var(--warn);   border: 1px solid rgba(245,158,11,0.3); }

/* ── Streamlit widget overrides ──────────────────────── */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stMultiSelect"] {
  background: var(--surface2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
}
div[data-testid="stSlider"] .stSlider { color: var(--accent); }

/* Dataframe */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--radius); }
.stDataFrame thead { background: var(--surface2); }
.stDataFrame th   { color: var(--text-muted) !important; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
.stDataFrame td   { font-size: 0.78rem; }

/* Buttons */
.stButton > button {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.78rem !important;
  transition: border-color 0.2s, color 0.2s;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

/* Primary / download button */
.stDownloadButton > button {
  background: rgba(0,212,255,0.1) !important;
  border: 1px solid rgba(0,212,255,0.4) !important;
  color: var(--accent) !important;
  border-radius: var(--radius) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.78rem !important;
}

/* Expander */
.streamlit-expanderHeader {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-size: 0.82rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-radius: 6px 6px 0 0 !important;
  font-size: 0.78rem !important;
  padding: 6px 14px !important;
  font-family: 'DM Mono', monospace !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* Upload zone */
div[data-testid="stFileUploader"] section {
  background: var(--surface2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: var(--radius) !important;
}
div[data-testid="stFileUploader"] section:hover { border-color: var(--accent) !important; }

/* Alerts */
div[data-testid="stAlert"] { border-radius: var(--radius) !important; font-size: 0.8rem; }

/* Sidebar labels */
section[data-testid="stSidebar"] label { font-size: 0.75rem !important; color: var(--text-muted) !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--accent);
  margin-bottom: 0.5rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    """Inject the dashboard CSS — call once at the top of each page."""
    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI card HTML builder
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str | int | float, sub: str = "",
             accent_color: str = "#00d4ff") -> str:
    """Return an HTML string for a single KPI card."""
    return (
        f'<div class="kpi-card" style="--accent-color:{accent_color}">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def render_kpi_row(cards: list[dict]) -> None:
    """
    Render a row of KPI cards.

    Each dict: { label, value, sub (opt), color (opt) }
    """
    html = '<div class="kpi-grid">'
    for c in cards:
        html += kpi_card(
            label=c["label"],
            value=c["value"],
            sub=c.get("sub", ""),
            accent_color=c.get("color", "#00d4ff"),
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str) -> None:
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def build_opportunities_df(opportunities: list[dict]) -> pd.DataFrame:
    """
    Convert raw opportunity dicts → display DataFrame.
    Keeps only columns that are useful in the UI.
    """
    if not opportunities:
        return pd.DataFrame()

    df = pd.DataFrame(opportunities)

    # Friendly column names
    rename = {
        "source_url":       "Source Page",
        "target_url":       "Target Page",
        "anchor_text":      "Anchor Text",
        "relevance_score":  "Relevance",
        "semantic_overlap": "Overlap",
        "pagerank_target":  "Target PR",
        "reason":           "Reason",
        "is_duplicate":     "Existing Link",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Friendly "Type" column
    if "Existing Link" in df.columns:
        df["Type"] = df["Existing Link"].map({True: "existing", False: "new"})

    # Round floats
    for col in ("Relevance", "Overlap", "Target PR"):
        if col in df.columns:
            df[col] = df[col].round(4)

    # Column order
    ordered = ["Type", "Source Page", "Target Page", "Anchor Text",
                "Relevance", "Overlap", "Target PR", "Reason"]
    df = df[[c for c in ordered if c in df.columns]]

    return df


def build_orphans_df(orphan_urls: list[str]) -> pd.DataFrame:
    if not orphan_urls:
        return pd.DataFrame(columns=["Orphan Page"])
    return pd.DataFrame({"Orphan Page": orphan_urls})


def build_pagerank_df(top_pages: list[dict]) -> pd.DataFrame:
    if not top_pages:
        return pd.DataFrame(columns=["Page", "PageRank"])
    df = pd.DataFrame(top_pages)
    df.columns = ["Page", "PageRank"]
    df["PageRank"] = df["PageRank"].round(6)
    return df


# ---------------------------------------------------------------------------
# Streamlit column config helpers
# ---------------------------------------------------------------------------

def opportunities_column_config() -> dict:
    return {
        "Type": st.column_config.TextColumn("Type", width="small"),
        "Source Page": st.column_config.LinkColumn("Source Page", display_text=r"https?://[^/]+(.+)"),
        "Target Page": st.column_config.LinkColumn("Target Page", display_text=r"https?://[^/]+(.+)"),
        "Anchor Text": st.column_config.TextColumn("Anchor Text", width="medium"),
        "Relevance":   st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1, format="%.3f"),
        "Overlap":     st.column_config.ProgressColumn("Overlap",   min_value=0, max_value=1, format="%.3f"),
        "Target PR":   st.column_config.NumberColumn("Target PR",   format="%.6f"),
        "Reason":      st.column_config.TextColumn("Reason", width="large"),
    }


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def opportunities_download_btn(df: pd.DataFrame, label: str = "⬇ Export CSV") -> None:
    """Render a download button for the opportunities DataFrame."""
    if df.empty:
        return
    st.download_button(
        label=label,
        data=df_to_csv_bytes(df),
        file_name="internal_link_opportunities.csv",
        mime="text/csv",
        use_container_width=False,
    )


def orphans_download_btn(df: pd.DataFrame) -> None:
    if df.empty:
        return
    st.download_button(
        label="⬇ Export Orphans CSV",
        data=df_to_csv_bytes(df),
        file_name="orphan_pages.csv",
        mime="text/csv",
    )
