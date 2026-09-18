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
answer generation here uses the Google Gemini API instead — the
retrieval side (embeddings + FAISS) is untouched from the notebook.

Everything below is intentionally hard-coded (no sidebar, no settings
UI): the end user only ever sees a question box and an answer. The
API key comes ONLY from Streamlit secrets (GEMINI_API_KEY) — it is
never entered by the user.
"""

import json
import os
import pickle

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

# ==========================================================================
# Fixed configuration — edit these values in code, never exposed in the UI
# ==========================================================================
APP_TITLE = "AI Detective"
APP_SUBTITLE = "AI Knowledge Investigation Assistant"
ARTIFACTS_DIR = "model"  # folder holding config.json / index.faiss / metadata.pkl

# "-latest" aliases auto-track Google's current recommended model, so this
# app doesn't need a code change every time a preview model is retired.
GENERATION_MODEL = "gemini-flash-latest"
REWRITE_MODEL = "gemini-flash-lite-latest"

# ضع مفتاح Gemini بتاعك هنا مباشرة (من https://aistudio.google.com/apikey)
GEMINI_API_KEY = "AQ.Ab8RN6LwhqHr1NhZT2eafT3kFknxdnInCSmd_uHLBp_R8KqzRQ"

TOP_K = 5              # evidence chunks retrieved per question
TEMPERATURE = 0.5      # answer generation temperature
SHOW_SOURCES = True    # show the retrieved evidence under each answer

st.set_page_config(page_title=APP_TITLE, page_icon="🕵️", layout="centered")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 820px; }
    .ai-detective-header {
        display: flex; align-items: center; gap: 0.75rem;
        padding-bottom: 0.5rem; border-bottom: 1px solid rgba(120,120,120,0.25);
        margin-bottom: 1.25rem;
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
# Load artifacts (cached) — silent to the user, no folder/setting exposed
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
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
    if genai is None:
        raise RuntimeError("The 'google-genai' package is not installed.")
    return genai.Client(api_key=key)


def generate_text(client, prompt, model=GENERATION_MODEL, max_tokens=500, temperature=0.5):
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return (resp.text or "").strip()


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


def rag_answer(client, question, history, index, embed_model, metadata, top_k=TOP_K, temperature=TEMPERATURE):
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
            <p>{APP_SUBTITLE}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Startup checks — developer-facing errors only, nothing for the end user
# to configure
# --------------------------------------------------------------------------
try:
    config, index, metadata = load_artifacts(ARTIFACTS_DIR)
except FileNotFoundError as e:
    st.error(
        f"Setup error: couldn't find `{e}`. Make sure the `{ARTIFACTS_DIR}/` folder "
        f"(config.json, index.faiss, metadata.pkl) is committed next to app.py."
    )
    st.stop()

api_key = GEMINI_API_KEY
if not api_key or api_key == "ضع_مفتاحك_هنا":
    st.error("Setup error: ضع مفتاح Gemini بتاعك في متغيّر GEMINI_API_KEY أول app.py.")
    st.stop()

embed_model = load_embedding_model(config.get("embedding_model", "intfloat/multilingual-e5-base"))
client = get_client(api_key)

# --------------------------------------------------------------------------
# Chat state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str, "sources": [...]}]
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []  # buffer memory fed into contextualize_query / prompt

if not st.session_state.messages:
    st.caption("اسأل عن أي حادثة أو تفصيلة، وهيجاوبك بناءً على الأدلة المسترجعة فقط.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and SHOW_SOURCES and msg.get("sources"):
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
# The ONLY thing the user interacts with: type a question, get an answer.
# --------------------------------------------------------------------------
question = st.chat_input("اكتب سؤالك هنا...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث..."):
            try:
                result = rag_answer(
                    client,
                    question,
                    st.session_state.rag_history,
                    index,
                    embed_model,
                    metadata,
                )
            except Exception as e:
                st.error(f"حصل خطأ أثناء توليد الإجابة: {e}")
                st.stop()

        st.markdown(result["answer"])
        if SHOW_SOURCES and result["sources"]:
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