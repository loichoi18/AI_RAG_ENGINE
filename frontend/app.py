"""Streamlit dashboard for the RAG engine.

A thin client over the FastAPI ``/v1/query`` endpoint (so it exercises the same
path real callers use). Lets a reviewer ask questions, inspect the grounded
answer with citations / confidence / retrieved chunks and scores, and compare
**hybrid vs dense** retrieval side by side.

Run:  streamlit run frontend/app.py
Configure the API URL via the RAG_API_URL environment variable.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
USERS = ["admin", "engineering", "hr"]


def ask(query: str, mode: str, user: str, top_k: int) -> dict[str, Any]:
    """Call the query endpoint and return the parsed response."""
    payload = {"query": query, "mode": mode, "user": user, "top_k": top_k}
    resp = requests.post(f"{API_URL}/v1/query", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def render_answer(result: dict[str, Any]) -> None:
    """Render a single answer result."""
    if result.get("refused"):
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])
    st.caption(
        f"confidence: {result['confidence']:.3f} · "
        f"latency: {result['latency_ms']} ms · "
        f"citations: {', '.join(result['citations']) or 'none'}"
    )
    with st.expander(f"Retrieved chunks ({len(result['retrieved_chunks'])})"):
        for chunk in result["retrieved_chunks"]:
            st.markdown(
                f"**{chunk['document_id']}** · score `{chunk['score']:.3f}`\n\n{chunk['text']}"
            )


st.set_page_config(page_title="RAG Engine", layout="wide")
st.title("RAG Engine — Internal Knowledge Search")
st.caption(f"API: {API_URL}")

with st.sidebar:
    st.header("Settings")
    user = st.selectbox("User (ACL)", USERS, index=0)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    compare = st.toggle("Compare Hybrid vs Dense", value=False)

question = st.text_input("Ask a question", placeholder="e.g. How do we deploy services?")

if st.button("Ask", type="primary") and question:
    try:
        if compare:
            col_hybrid, col_dense = st.columns(2)
            with col_hybrid:
                st.subheader("Hybrid")
                render_answer(ask(question, "hybrid", user, top_k))
            with col_dense:
                st.subheader("Dense")
                render_answer(ask(question, "dense", user, top_k))
        else:
            render_answer(ask(question, "hybrid", user, top_k))
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
