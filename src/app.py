"""
One entry point for every dashboard:

    make dash        # = python -m streamlit run src/app.py

Pages: E4 end-to-end funnel (default), E1 classification, pipeline database.
All bind to localhost only (``.streamlit/config.toml``).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Financial AI Pipeline", page_icon="📊", layout="wide")

page = st.navigation(
    [
        st.Page(HERE / "e4_dashboard.py", title="E4 · end-to-end", default=True),
        st.Page(HERE / "e1_dashboard.py", title="E1 · classification"),
        st.Page(HERE / "dashboard.py", title="Pipeline database"),
    ]
)
page.run()
