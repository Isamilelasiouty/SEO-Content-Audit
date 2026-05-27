# SEO Intelligence Dashboard

An AI-powered internal linking and SEO analysis platform built for content teams.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
streamlit run streamlit_app/pages/internal_links_dashboard.py
```

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → set **Main file path** to `streamlit_app/pages/internal_links_dashboard.py`
4. Add `ANTHROPIC_API_KEY` in the Secrets section

## Project Structure

```
seo_engine/        — core analysis engines
components/        — reusable UI components (charts, tables, CSS)
utils/             — helper functions
streamlit_app/     — Streamlit pages
tests/             — pytest test suite
```
