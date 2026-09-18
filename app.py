"""
AI Detective — AI Knowledge Investigation Assistant
====================================================
Streamlit front-end + serving layer for the RAG pipeline built in the
"mid-term-project" notebook (PDF incident reports + structured crime
records -> multilingual-e5-base embeddings -> FAISS IndexFlatIP ->
LLM answer generation with conversation memory & query rewriting).

This app loads the three artifacts produced at the end of the notebook
(config.json, index.faiss, metadata.pkl) and reproduces the exact same
retrieval pipeline. Because a 12B local LLM (Mistral-Nemo-Instruct) is
not deployable on free/CPU-only hosting like Streamlit Community Cloud,
answer generation here uses the Anthropic Claude API instead — the
retrieval side (embeddings + FAISS) is untouched from the notebook.
"""

import json
import os
import pickle

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
APP_TITLE = "AI Detective"
APP_SUBTITLE = "AI Knowledge Investigation Assistant"
DEFAULT_ARTIFACTS_DIR = "model"  # folder holding config.json / index.faiss / metadata.pkl
GENERATION_MODEL = "claude-sonnet-5"
REWRITE_MODEL = "claude-haiku-4-5-20251001"

EXAMPLE_QUESTIONS = [
    "Tell me about INC-001.",
    "What vehicle was mentioned?",
    "Was it mentioned in another case?",
    "Does that prove the incidents are connected?",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 950px; }
    .ai-detective-header {
        display: flex; align-items: center; gap: 0.75rem;
        padding-bottom: 0.25rem; border-bottom: 1px solid rgba(120,120,120,0.25);
        margin-bottom: 1rem;
    }
    .ai-detective-header h1 { margin: 0; font-size: 1.9rem; }
    .ai-detective-header p { margin: 0; opacity: 0.7; font-size: 0.95rem; }
    .evidence-card {
        border: 1px solid rgba(120,120,120,0.25); border-radius: 10px;
        padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; font-size: 0.85rem;
    }
    .evidence-score {
        display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px;
        background: rgba(46,164,79,0.15); font-weight: 600; font-size: 0.75rem;
    }
    .stChatMessage { font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    artifacts_dir = st.text_input(
        "Artifacts folder",
        value=os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR),
        help="Folder containing config.json, index.faiss and metadata.pkl "
             "(this folder must sit next to app.py in the repo).",
    )

    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            help="Only needed if ANTHROPIC_API_KEY isn't set in Streamlit secrets.",
        )

    st.divider()
    top_k = st.slider("Evidence chunks to retrieve (top_k)", 1, 10, 5)
    temperature = st.slider("Answer creativity (temperature)", 0.0, 1.0, 0.5, 0.05)
    show_sources = st.checkbox("Show retrieved evidence under each answer", value=True)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag_history = []
        st.rerun()

    st.divider()
    st.markdown("### 💡 Try asking")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"example_{q}"):
            st.session_state.pending_question = q


# --------------------------------------------------------------------------
# Load artifacts (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading FAISS index & metadata...")
def load_artifacts(directory: str):
    config_path = os.path.join(directory, "config.json")
    index_path = os.path.join(directory, "index.faiss")
    metadata_path = os.path.join(directory, "metadata.pkl")

    for p in (config_path, index_path, metadata_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return config, index, metadata


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name, device="cpu")


# --------------------------------------------------------------------------
# RAG pipeline — mirrors the notebook exactly (retrieval side unchanged)
# --------------------------------------------------------------------------
def search_index(query, model, index, metadata, top_k=5):
    q = model.encode([f"query: {query}"], convert_to_numpy=True)
    faiss.normalize_L2(q)
    scores, ids = index.search(q, top_k)
    return [dict(metadata[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


def build_context(history, max_turns=6):
    return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in history[-max_turns:])


def build_rag_prompt(question, evidence, history_text):
    evidence_block = "\n\n".join(
        f"[{i}] ({e['source_name']}, "
        f"{'page ' + str(e['page']) if e['source_type'] == 'pdf' else 'row ' + str(e['row_number'])}, "
        f"incident {e['incident_id']}, score {e['score']:.2f})\n{e['text']}"
        for i, e in enumerate(evidence, 1)
    ) or "(no evidence retrieved)"

    return f"""You are AI Detective. Answer using ONLY the evidence and conversation below.
- Never invent facts; say so if the evidence is insufficient.
- A shared detail (e.g. same vehicle) across cases is only a POSSIBLE connection, never proof.
- Mention relevant incident IDs when synthesizing multiple sources.

Conversation so far:
{history_text or "(none)"}

Retrieved evidence:
{evidence_block}

Question: {question}
Answer:"""


def get_client(key):
    if anthropic is None:
        raise RuntimeError("The 'anthropic' package is not installed.")
    return anthropic.Anthropic(api_key=key)


def generate_text(client, prompt, model=GENERATION_MODEL, max_tokens=500, temperature=0.5):
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def contextualize_query(client, history, question):
    if not history:
        return question
    prompt = f"""Conversation so far:
{build_context(history, 4)}

New question: "{question}"
Rewrite it as one standalone question (resolve pronouns like "it"/"that"). Reply with ONLY the rewritten question.
Rewritten question:"""
    rewritten = generate_text(
        client, prompt, model=REWRITE_MODEL, max_tokens=60, temperature=0.3
    ).split("\n")[0].strip()
    return rewritten or question


def rag_answer(client, question, history, index, embed_model, metadata, top_k=5, temperature=0.5):
    contextualized = contextualize_query(client, history, question)
    evidence = search_index(contextualized, embed_model, index, metadata, top_k=top_k)
    prompt = build_rag_prompt(question, evidence, build_context(history))
    answer = generate_text(client, prompt, temperature=temperature)
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return {
        "answer": answer,
        "original_question": question,
        "contextualized_question": contextualized,
        "sources": [
            {k: e[k] for k in ("source_type", "source_name", "page", "row_number", "incident_id", "score")}
            for e in evidence
        ],
    }


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="ai-detective-header">
        <div style="font-size:2.2rem;">🕵️</div>
        <div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE} — retrieval-augmented Q&A over incident reports &amp; case records</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load everything, fail gracefully with clear instructions
# --------------------------------------------------------------------------
try:
    config, index, metadata = load_artifacts(artifacts_dir)
except FileNotFoundError as e:
    st.error(
        f"Couldn't find the artifacts folder or a file inside it: `{e}`.\n\n"
        f"Make sure a folder named **`{artifacts_dir}`** containing "
        f"`config.json`, `index.faiss` and `metadata.pkl` sits next to `app.py` "
        f"in your GitHub repo, then redeploy."
    )
    st.stop()

if not api_key:
    st.warning(
        "Add your **Anthropic API key** in the sidebar (or set `ANTHROPIC_API_KEY` "
        "in Streamlit secrets) to start chatting."
    )
    st.stop()

embed_model = load_embedding_model(config.get("embedding_model", "intfloat/multilingual-e5-base"))
client = get_client(api_key)

with st.sidebar:
    st.divider()
    st.markdown("### 📦 Knowledge base")
    st.caption(f"Embedding model: `{config.get('embedding_model', 'n/a')}`")
    st.caption(f"Knowledge items: **{config.get('num_knowledge_items', index.ntotal)}**")
    st.caption(f"Index type: `{config.get('index_type', 'FAISS')}`")
    st.caption(f"Built: {config.get('created_at', 'n/a')}")

# --------------------------------------------------------------------------
# Chat state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str, "sources": [...]}]
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []  # buffer memory fed into contextualize_query / prompt

if not st.session_state.messages:
    st.info(
        "Ask about a specific incident (e.g. *\"What happened in INC-001?\"*), a detail "
        "across cases, or a follow-up like *\"was it mentioned elsewhere?\"* — the assistant "
        "keeps conversation memory and answers strictly from the retrieved evidence."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s) used"):
                for s in msg["sources"]:
                    loc = f"page {s['page']}" if s["source_type"] == "pdf" else f"row {s['row_number']}"
                    st.markdown(
                        f"""<div class="evidence-card">
                        <span class="evidence-score">score {s['score']:.2f}</span>
                        &nbsp; <b>{s['source_name']}</b> ({loc}) — incident <code>{s['incident_id']}</code>
                        </div>""",
                        unsafe_allow_html=True,
                    )

# --------------------------------------------------------------------------
# Input handling (chat box or example-question button)
# --------------------------------------------------------------------------
question = st.chat_input("Ask the AI Detective about a case...")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Investigating..."):
            try:
                result = rag_answer(
                    client,
                    question,
                    st.session_state.rag_history,
                    index,
                    embed_model,
                    metadata,
                    top_k=top_k,
                    temperature=temperature,
                )
            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")
                st.stop()

        st.markdown(result["answer"])
        if result["contextualized_question"] != result["original_question"]:
            st.caption(f"🔎 Understood as: *{result['contextualized_question']}*")
        if show_sources and result["sources"]:
            with st.expander(f"📎 {len(result['sources'])} source(s) used"):
                for s in result["sources"]:
                    loc = f"page {s['page']}" if s["source_type"] == "pdf" else f"row {s['row_number']}"
                    st.markdown(
                        f"""<div class="evidence-card">
                        <span class="evidence-score">score {s['score']:.2f}</span>
                        &nbsp; <b>{s['source_name']}</b> ({loc}) — incident <code>{s['incident_id']}</code>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
