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

Gemini API key is configured directly in this file for the live demo.
"""

# ============================================================================
# Imports
# ============================================================================

import json
import os
import pickle

import requests
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

# Keep credentials in environment variables or Streamlit secrets.
# Example:
#   export GEMINI_API_KEY="..."
#   setx GEMINI_API_KEY "..."   # Windows PowerShell
# or create .streamlit/secrets.toml with GEMINI_API_KEY = "..."
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Use a currently supported Gemini model for new users.
# Google has deprecated older 2.5 models; keep a compatibility alias so stale env values
# automatically upgrade to the supported 3.6 family.
DEFAULT_GENERATION_MODEL = "gemini-3.6-flash"
DEFAULT_REWRITE_MODEL = "gemini-3.6-flash"


def resolve_model_name(model_name, fallback):
    """Map deprecated Gemini aliases to the latest supported model name."""
    value = (model_name or "").strip()
    if not value:
        return fallback

    aliases = {
        "gemini-2.5-flash": "gemini-3.6-flash",
        "models/gemini-2.5-flash": "models/gemini-3.6-flash",
        "gemini-2.5-flash-lite": "gemini-3.6-flash",
        "models/gemini-2.5-flash-lite": "models/gemini-3.6-flash",
    }

    normalized = value.lower()
    return aliases.get(normalized, value)


GENERATION_MODEL = resolve_model_name(
    os.getenv("GEMINI_GENERATION_MODEL") or os.getenv("GEMINI_MODEL"),
    DEFAULT_GENERATION_MODEL,
)
REWRITE_MODEL = resolve_model_name(
    os.getenv("GEMINI_REWRITE_MODEL") or os.getenv("GEMINI_MODEL"),
    DEFAULT_REWRITE_MODEL,
)

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

# Current google-genai releases support the 2026 Gemini authentication changes.
# Streamlit Cloud should install google-genai==2.24.0 from requirements.txt.


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
    """Load the Gemini API key from environment variables or Streamlit secrets."""

    try:
        secrets_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secrets_key = ""

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or secrets_key
        or GEMINI_API_KEY
    )

    if not api_key or not str(api_key).strip():
        st.error(
            "Gemini API key is missing. Add it to your environment as GEMINI_API_KEY "
            "or create .streamlit/secrets.toml with GEMINI_API_KEY = \"...\" and restart the app."
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
    """Create the official Google GenAI client."""

    api_key = get_api_key()

    if genai is None:
        return {"api_key": api_key, "sdk": None}

    return {
        "api_key": api_key,
        "sdk": genai.Client(api_key=api_key),
    }


# ============================================================================
# Gemini text generation
# ============================================================================

def _generate_text_rest(
    api_key,
    prompt,
    model,
    max_tokens=500,
    temperature=0.5,
):
    """
    Direct Gemini REST fallback using the documented x-goog-api-key header.
    This bypasses SDK credential/header handling.
    """

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        json={
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        },
        timeout=60,
    )

    if not response.ok:
        try:
            details = response.json()
        except Exception:
            details = response.text

        raise RuntimeError(
            f"Gemini REST error {response.status_code}: {details}"
        )

    payload = response.json()
    candidates = payload.get("candidates") or []

    text_parts = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            part_text = part.get("text")
            if part_text:
                text_parts.append(part_text)

    text = "\n".join(text_parts).strip()

    if not text:
        raise RuntimeError("Gemini REST returned an empty response.")

    return text


def generate_text(
    client,
    prompt,
    model,
    max_tokens=500,
    temperature=0.5,
):
    """
    Generate text with the official SDK, then fall back to direct REST
    if the SDK request fails at the authentication layer.
    """

    api_key = client["api_key"]
    sdk_client = client.get("sdk")

    if sdk_client is not None and genai_types is not None:
        try:
            response = sdk_client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            text = getattr(response, "text", None)

            if text:
                return text.strip()

        except Exception as sdk_error:
            sdk_message = str(sdk_error)

            # If the key is wrong or the model name is invalid, the REST layer can
            # still fail. Keep the message actionable for app users.
            if "401" not in sdk_message and "UNAUTHENTICATED" not in sdk_message and "404" not in sdk_message and "not found" not in sdk_message.lower():
                raise

            try:
                return _generate_text_rest(
                    api_key=api_key,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as rest_error:
                rest_message = str(rest_error)
                raise RuntimeError(
                    "Gemini authentication or model setup failed. Check that your API key is valid "
                    "and that the model name is available for your Google AI project. "
                    "The app expects a working GEMINI_API_KEY and a current Gemini model such as "
                    "gemini-3.6-flash.\n\n"
                    f"SDK error: {sdk_message}\n\n"
                    f"REST error: {rest_message}"
                ) from rest_error

    return _generate_text_rest(
        api_key=api_key,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


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