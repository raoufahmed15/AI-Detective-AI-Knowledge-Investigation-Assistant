"""
AI Detective — AI Knowledge Investigation Assistant
====================================================

Streamlit front-end + serving layer for the RAG pipeline built in the
"mid-term-project" notebook.

Pipeline:
PDF incident reports + structured crime records
-> multilingual-e5-base embeddings
-> FAISS IndexFlatIP
-> Gemini LLM answer generation
-> conversation memory + query rewriting

The API key is loaded ONLY from Streamlit secrets:
    GEMINI_API_KEY

It is never hard-coded in this file.
"""

# ============================================================================
# Imports
# ============================================================================

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


# ============================================================================
# Fixed configuration
# ============================================================================

APP_TITLE = "AI Detective"
APP_SUBTITLE = "AI Knowledge Investigation Assistant"

ARTIFACTS_DIR = "model"

# Gemini models
GENERATION_MODEL = "gemini-flash-latest"
REWRITE_MODEL = "gemini-flash-lite-latest"

TOP_K = 5
TEMPERATURE = 0.5
SHOW_SOURCES = True


# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🕵️",
    layout="centered",
)


# ============================================================================
# Styling
# ============================================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2.5rem;
        max-width: 820px;
    }

    .ai-detective-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(120,120,120,0.25);
        margin-bottom: 1.25rem;
    }

    .ai-detective-header h1 {
        margin: 0;
        font-size: 1.9rem;
    }

    .ai-detective-header p {
        margin: 0;
        opacity: 0.7;
        font-size: 0.95rem;
    }

    .evidence-card {
        border: 1px solid rgba(120,120,120,0.25);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }

    .evidence-score {
        display: inline-block;
        padding: 0.05rem 0.5rem;
        border-radius: 999px;
        background: rgba(46,164,79,0.15);
        font-weight: 600;
        font-size: 0.75rem;
    }

    .stChatMessage {
        font-size: 0.95rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Gemini API key
# ============================================================================

def get_api_key():
    """
    Load Gemini API key from Streamlit secrets.

    Expected:
        GEMINI_API_KEY = "AQ.Ab8RN6LwhqHr1NhZT2eafT3kFknxdnInCSmd_uHLBp_R8KqzRQ"
    """

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        st.error(
            "Setup error: GEMINI_API_KEY is missing from Streamlit Secrets."
        )
        st.info(
            "Add GEMINI_API_KEY to Streamlit Secrets, then restart the app."
        )
        st.stop()

    return str(api_key).strip()


# ============================================================================
# Load artifacts
# ============================================================================

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_artifacts(directory: str):

    config_path = os.path.join(directory, "config.json")
    index_path = os.path.join(directory, "index.faiss")
    metadata_path = os.path.join(directory, "metadata.pkl")

    for path in (config_path, index_path, metadata_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return config, index, metadata


# ============================================================================
# Embedding model
# ============================================================================

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedding_model(model_name: str):

    return SentenceTransformer(
        model_name,
        device="cpu",
    )


# ============================================================================
# RAG retrieval
# ============================================================================

def search_index(
    query,
    model,
    index,
    metadata,
    top_k=5,
):
    """
    Search the FAISS index using multilingual-e5-base.
    """

    q = model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
    )

    faiss.normalize_L2(q)

    scores, ids = index.search(
        q,
        top_k,
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx == -1:
            continue

        results.append(
            {
                **metadata[idx],
                "score": float(score),
            }
        )

    return results


# ============================================================================
# Conversation context
# ============================================================================

def build_context(
    history,
    max_turns=6,
):
    """
    Convert conversation history into plain text.
    """

    return "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history[-max_turns:]
    )


# ============================================================================
# RAG prompt
# ============================================================================

def build_rag_prompt(
    question,
    evidence,
    history_text,
):

    evidence_block = "\n\n".join(
        f"[{i}] "
        f"({e['source_name']}, "
        f"{'page ' + str(e['page']) if e['source_type'] == 'pdf' else 'row ' + str(e['row_number'])}, "
        f"incident {e['incident_id']}, "
        f"score {e['score']:.2f})\n"
        f"{e['text']}"
        for i, e in enumerate(evidence, 1)
    )

    if not evidence_block:
        evidence_block = "(no evidence retrieved)"

    return f"""
You are AI Detective.

Answer the user's question using ONLY the retrieved evidence
and the conversation below.

Rules:

- Never invent facts.
- If the evidence is insufficient, clearly say that the evidence is insufficient.
- A shared detail across cases is only a POSSIBLE connection, never proof.
- Do not treat similarity as proof of identity or causation.
- Mention relevant incident IDs when synthesizing multiple sources.
- Keep the answer concise but informative.
- Prefer direct evidence over assumptions.
- If multiple pieces of evidence conflict, explicitly mention the conflict.

Conversation so far:
{history_text or "(none)"}

Retrieved evidence:
{evidence_block}

Question:
{question}

Answer:
""".strip()


# ============================================================================
# Gemini client
# ============================================================================

def get_client():
    """
    Create the official Google GenAI client.

    The API key comes from Streamlit Secrets.
    """

    if genai is None:
        raise RuntimeError(
            "The 'google-genai' package is not installed. "
            "Run: pip install -U google-genai"
        )

    api_key = get_api_key()

    return genai.Client(
        api_key=api_key
    )


# ============================================================================
# Gemini text generation
# ============================================================================

def generate_text(
    client,
    prompt,
    model,
    max_tokens=500,
    temperature=0.5,
):
    """
    Generate text using Gemini.
    """

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()


# ============================================================================
# Query contextualization
# ============================================================================

def contextualize_query(
    client,
    history,
    question,
):

    if not history:
        return question

    prompt = f"""
Conversation so far:
{build_context(history, 4)}

New question:
"{question}"

Rewrite the new question as ONE standalone question.

Resolve pronouns such as:
- it
- that
- this
- they
- them
- he
- she

Use the previous conversation only when necessary.

Reply with ONLY the rewritten question.

Rewritten question:
""".strip()

    rewritten = generate_text(
        client=client,
        prompt=prompt,
        model=REWRITE_MODEL,
        max_tokens=60,
        temperature=0.3,
    )

    rewritten = rewritten.split("\n")[0].strip()

    return rewritten or question


# ============================================================================
# Main RAG answer
# ============================================================================

def rag_answer(
    client,
    question,
    history,
    index,
    embed_model,
    metadata,
    top_k=TOP_K,
    temperature=TEMPERATURE,
):

    # ------------------------------------------------------------
    # 1. Rewrite question using conversation history
    # ------------------------------------------------------------

    contextualized = contextualize_query(
        client,
        history,
        question,
    )

    # ------------------------------------------------------------
    # 2. Retrieve evidence
    # ------------------------------------------------------------

    evidence = search_index(
        contextualized,
        embed_model,
        index,
        metadata,
        top_k=top_k,
    )

    # ------------------------------------------------------------
    # 3. Build grounded RAG prompt
    # ------------------------------------------------------------

    prompt = build_rag_prompt(
        question,
        evidence,
        build_context(history),
    )

    # ------------------------------------------------------------
    # 4. Generate final answer
    # ------------------------------------------------------------

    answer = generate_text(
        client=client,
        prompt=prompt,
        model=GENERATION_MODEL,
        max_tokens=500,
        temperature=temperature,
    )

    # ------------------------------------------------------------
    # 5. Update conversation memory
    # ------------------------------------------------------------

    history.extend(
        [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
    )

    # ------------------------------------------------------------
    # 6. Return result
    # ------------------------------------------------------------

    return {
        "answer": answer,
        "original_question": question,
        "contextualized_question": contextualized,
        "sources": [
            {
                key: evidence_item[key]
                for key in (
                    "source_type",
                    "source_name",
                    "page",
                    "row_number",
                    "incident_id",
                    "score",
                )
            }
            for evidence_item in evidence
        ],
    }


# ============================================================================
# Header
# ============================================================================

st.markdown(
    f"""
    <div class="ai-detective-header">

        <div style="font-size:2.2rem;">
            🕵️
        </div>

        <div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Startup checks
# ============================================================================

try:

    config, index, metadata = load_artifacts(
        ARTIFACTS_DIR
    )

except FileNotFoundError as error:

    st.error(
        f"Setup error: couldn't find `{error}`."
    )

    st.info(
        f"Make sure the `{ARTIFACTS_DIR}/` folder contains:\n\n"
        "- config.json\n"
        "- index.faiss\n"
        "- metadata.pkl"
    )

    st.stop()


# ============================================================================
# Load embedding model
# ============================================================================

embed_model = load_embedding_model(
    config.get(
        "embedding_model",
        "intfloat/multilingual-e5-base",
    )
)


# ============================================================================
# Initialize Gemini client
# ============================================================================

try:

    client = get_client()

except Exception as error:

    st.error(
        f"Gemini initialization failed: {error}"
    )

    st.stop()


# ============================================================================
# Chat state
# ============================================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "rag_history" not in st.session_state:

    st.session_state.rag_history = []


# ============================================================================
# Empty state
# ============================================================================

if not st.session_state.messages:

    st.caption(
        "اسأل عن أي حادثة أو تفصيلة، "
        "وهيجاوبك بناءً على الأدلة المسترجعة فقط."
    )


# ============================================================================
# Render previous messages
# ============================================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and SHOW_SOURCES
            and message.get("sources")
        ):

            with st.expander(
                f"📎 {len(message['sources'])} source(s) used"
            ):

                for source in message["sources"]:

                    if source["source_type"] == "pdf":
                        location = f"page {source['page']}"
                    else:
                        location = f"row {source['row_number']}"

                    st.markdown(
                        f"""
                        <div class="evidence-card">

                            <span class="evidence-score">
                                score {source['score']:.2f}
                            </span>

                            &nbsp;

                            <b>{source['source_name']}</b>

                            ({location})

                            —

                            incident
                            <code>{source['incident_id']}</code>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================================
# Chat input
# ============================================================================

question = st.chat_input(
    "اكتب سؤالك هنا..."
)


# ============================================================================
# Handle question
# ============================================================================

if question:

    # ------------------------------------------------------------
    # Add user message
    # ------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ------------------------------------------------------------
    # Generate assistant response
    # ------------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("جاري البحث..."):

            try:

                result = rag_answer(
                    client=client,
                    question=question,
                    history=st.session_state.rag_history,
                    index=index,
                    embed_model=embed_model,
                    metadata=metadata,
                    top_k=TOP_K,
                    temperature=TEMPERATURE,
                )

            except Exception as error:

                st.error(
                    f"حصل خطأ أثناء توليد الإجابة:\n\n{error}"
                )

                st.stop()

        # --------------------------------------------------------
        # Show answer
        # --------------------------------------------------------

        st.markdown(
            result["answer"]
        )

        # --------------------------------------------------------
        # Show sources
        # --------------------------------------------------------

        if (
            SHOW_SOURCES
            and result["sources"]
        ):

            with st.expander(
                f"📎 {len(result['sources'])} source(s) used"
            ):

                for source in result["sources"]:

                    if source["source_type"] == "pdf":
                        location = f"page {source['page']}"
                    else:
                        location = f"row {source['row_number']}"

                    st.markdown(
                        f"""
                        <div class="evidence-card">

                            <span class="evidence-score">
                                score {source['score']:.2f}
                            </span>

                            &nbsp;

                            <b>{source['source_name']}</b>

                            ({location})

                            —

                            incident
                            <code>{source['incident_id']}</code>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ------------------------------------------------------------
    # Save assistant message
    # ------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )