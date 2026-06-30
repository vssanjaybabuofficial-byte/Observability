"""Streamlit dashboard for chronicle traces.

Launch::

    chronicle-dashboard path/to/trace.jsonl

or programmatically::

    from chronicle.dashboard import launch
    launch("path/to/trace.jsonl")

Or from any Chronicle::

    c.dump("trace.jsonl"); launch("trace.jsonl")
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def launch(trace_path: Optional[str] = None) -> int:
    """Spawn `streamlit run` on this module."""
    here = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(here)]
    if trace_path:
        cmd += ["--", "--trace", str(trace_path)]
    return subprocess.call(cmd)


def cli() -> None:
    """Console entry-point. Accepts an optional positional trace path."""
    p = argparse.ArgumentParser(description="Launch the chronicle dashboard.")
    p.add_argument("trace", nargs="?", default=None, help="Path to a JSONL trace.")
    args = p.parse_args()
    raise SystemExit(launch(args.trace))


def _format_dur(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    if ms < 60_000:
        return f"{ms/1000:.2f} s"
    return f"{ms/60_000:.2f} min"


def _run_streamlit_app() -> None:
    """The Streamlit page itself. Only imports streamlit lazily so the rest of
    the package remains import-light.
    """
    import streamlit as st
    import pandas as pd
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--trace", default=None)
    args, _ = p.parse_known_args()

    st.set_page_config(page_title="chronicle", layout="wide")
    st.title("chronicle - agent trace dashboard")
    st.caption("Merkle-DAG agent traceability  |  causal . integrity . semantic")

    from chronicle import Chronicle

    if args.trace and Path(args.trace).exists():
        trace_path = args.trace
    else:
        st.sidebar.subheader("Load trace")
        uploaded = st.sidebar.file_uploader("Upload a JSONL trace", type=["jsonl", "json"])
        if uploaded is None:
            st.info("No trace loaded. Provide --trace or upload a JSONL file in the sidebar.")
            return
        import tempfile
        tf = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tf.write(uploaded.read())
        tf.close()
        trace_path = tf.name

    c = Chronicle.load(trace_path)
    stats = c.stats()
    ok, bad = c.verify()

    # ---- top metric cards ----
    cols = st.columns(6)
    cols[0].metric("Events", stats["total_events"])
    cols[1].metric("LLM calls", stats["llm_calls"])
    cols[2].metric("Tool calls", stats["tool_calls"])
    cols[3].metric("Tokens", f'{stats["total_tokens"]:,}')
    cols[4].metric("LLM latency", _format_dur(stats["llm_duration_ms"]))
    cols[5].metric("Est. cost", f'${stats["estimated_cost_usd"]:.4f}')

    if ok:
        st.success(f"Integrity OK . Merkle root: `{c.root()}`")
    else:
        st.error(f"INTEGRITY FAILED . Bad event ids: {bad}")

    tabs = st.tabs(["Overview", "LLM calls", "Tool calls", "DAG", "Search", "Raw"])

    # ---- Overview ----
    with tabs[0]:
        st.subheader("Event mix")
        by_kind = stats["by_kind"]
        if by_kind:
            df = pd.DataFrame({"kind": list(by_kind), "count": list(by_kind.values())})
            st.bar_chart(df.set_index("kind"))

        if stats["by_model"]:
            st.subheader("Per-model breakdown")
            rows = []
            for model, row in stats["by_model"].items():
                rows.append({
                    "model": model,
                    "calls": row["calls"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "duration_ms": row["duration_ms"],
                    "cost_usd": row["cost_usd"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ---- LLM calls ----
    with tabs[1]:
        rows = []
        evts = list(c)
        by_id = {e.id: e for e in evts}
        for e in evts:
            if e.kind == "llm_result":
                call = by_id.get(e.parents[0]) if e.parents else None
                rows.append({
                    "id": e.id[:10],
                    "model": e.payload.get("model"),
                    "input_tokens": e.payload.get("input_tokens", 0),
                    "output_tokens": e.payload.get("output_tokens", 0),
                    "duration_ms": e.telemetry.get("duration_ms", 0.0),
                    "cost_usd": e.telemetry.get("cost_usd", 0.0),
                    "text": (e.payload.get("text") or "")[:140],
                    "call_args": (str(call.payload.get("args")) if call else ""),
                })
        if rows:
            df = pd.DataFrame(rows).sort_values("duration_ms", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No llm_result events in this trace.")

    # ---- Tool calls ----
    with tabs[2]:
        rows = []
        for e in c:
            if e.kind == "tool_result":
                rows.append({
                    "id": e.id[:10],
                    "tool": e.payload.get("name"),
                    "duration_ms": e.telemetry.get("duration_ms", 0.0),
                    "result": str(e.payload.get("result"))[:160],
                })
        if rows:
            df = pd.DataFrame(rows).sort_values("duration_ms", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.subheader("Tool latency distribution")
            st.bar_chart(df.set_index("tool")[["duration_ms"]])
        else:
            st.info("No tool_result events in this trace.")

    # ---- DAG ----
    with tabs[3]:
        import io
        import matplotlib.pyplot as plt
        import networkx as nx
        g = nx.DiGraph()
        for e in c:
            g.add_node(e.id[:6], label=f"{e.kind}\n{e.actor[:20]}")
            for parent in e.parents:
                g.add_edge(parent[:6], e.id[:6])
        try:
            pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
        except Exception:
            pos = nx.spring_layout(g, seed=42)

        fig, ax = plt.subplots(figsize=(12, max(4, len(g) * 0.4)))
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1800, node_color="#cfe2ff", edgecolors="#1f4e79")
        nx.draw_networkx_edges(g, pos, ax=ax, arrowsize=14, edge_color="#666")
        labels = {n: g.nodes[n].get("label", n) for n in g.nodes}
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
        ax.axis("off")
        st.pyplot(fig)

    # ---- Search ----
    with tabs[4]:
        q = st.text_input("Semantic search", value="")
        if q:
            hits = c.search(q, k=10)
            for evt, score in hits:
                with st.container(border=True):
                    st.markdown(f"**{evt.kind}** . `{evt.actor}` . score `{score:+.3f}`")
                    st.code(str(evt.payload)[:600], language="json")

    # ---- Raw ----
    with tabs[5]:
        for e in c:
            with st.expander(f"{e.kind} | {e.actor} | {e.id[:12]}"):
                st.json({"payload": dict(e.payload), "telemetry": dict(e.telemetry),
                         "meta": dict(e.meta), "parents": list(e.parents)})


# Streamlit runs this module as __main__ when invoked via `streamlit run`.
if __name__ == "__main__":
    _run_streamlit_app()
