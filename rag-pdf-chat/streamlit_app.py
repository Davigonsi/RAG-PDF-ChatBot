"""Streamlit frontend for the RAG PDF Chat application."""

import json

import requests
import streamlit as st

_API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for more spacious UI ─────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
        .stChatMessage { padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
        section[data-testid="stSidebar"] { width: 360px !important; }
        .uploaded-file-tag {
            background: #1e3a5f; color: #90caf9; border-radius: 6px;
            padding: 4px 10px; font-size: 0.82rem; margin: 3px 0; display: inline-block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processed_names" not in st.session_state:
    st.session_state.processed_names = set()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG PDF Chat")
    st.caption("Powered by Claude + Pinecone")
    st.divider()

    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDFs at once. Each will be indexed separately.",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.processed_names
        ]
        if new_files:
            for uf in new_files:
                with st.spinner(f"Processing {uf.name}..."):
                    try:
                        response = requests.post(
                            f"{_API_BASE}/upload",
                            files={
                                "file": (uf.name, uf.getvalue(), "application/pdf")
                            },
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.processed_names.add(uf.name)
                            st.session_state.uploaded_files.append(
                                {"name": data["filename"], "chunks": data["total_chunks"]}
                            )
                            st.success(f"✓ {data['filename']} — {data['total_chunks']} chunks")
                        else:
                            detail = response.json().get("detail", response.text)
                            st.error(f"Failed: {uf.name} — {detail}")
                    except Exception as e:
                        st.error(f"Failed: {uf.name} — {e}")

    st.divider()

    if st.session_state.uploaded_files:
        st.subheader("Indexed Documents")
        for doc in st.session_state.uploaded_files:
            st.markdown(
                f'<div class="uploaded-file-tag">📎 {doc["name"]} &nbsp;·&nbsp; {doc["chunks"]} chunks</div>',
                unsafe_allow_html=True,
            )
        st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.session_state.processed_names = set()
            st.rerun()

# ── Main area header ──────────────────────────────────────────────────────────
if not st.session_state.uploaded_files:
    st.markdown(
        """
        <div style="text-align:center; padding: 5rem 2rem; color: #888;">
            <h2>👈 Upload PDFs from the sidebar to get started</h2>
            <p>You can upload multiple documents and ask questions across all of them.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    names = ", ".join(d["name"] for d in st.session_state.uploaded_files)
    st.markdown(
        f"<p style='color:#888; font-size:0.85rem;'>Asking across: <b>{names}</b></p>",
        unsafe_allow_html=True,
    )

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.write(f"📄 **{s['source']}** — Page {s['page']}")

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Ask anything about your uploaded PDFs..." if st.session_state.uploaded_files
    else "Upload a PDF first to start chatting..."
)

if prompt:
    if not st.session_state.uploaded_files:
        st.warning("Please upload at least one PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Claude is thinking..."):
            try:
                response = requests.post(
                    f"{_API_BASE}/ask",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"query": prompt, "top_k": 5}),
                )
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["answer"])
                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for s in data["sources"]:
                                st.write(f"📄 **{s['source']}** — Page {s['page']}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data.get("sources", []),
                        }
                    )
                else:
                    detail = response.json().get("detail", response.text)
                    st.error(f"Error: {detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {detail}", "sources": []}
                    )
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {e}", "sources": []}
                )

    st.rerun()
