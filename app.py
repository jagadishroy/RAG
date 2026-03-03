import streamlit as st
import streamlit.components.v1 as components

import json
import os
import time
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anthropic
from anthropic import Anthropic


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="InsureIQ — RAG Claims Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling (keeps the “better before” card layout feel)
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #f7f6f2; }

.header-banner {
    background: linear-gradient(135deg, #1a3c5e 0%, #0d5c3a 100%);
    padding: 28px 32px; border-radius: 14px; margin-bottom: 18px; color: white;
}
.header-banner h1 { font-size: 28px; margin: 0; letter-spacing: .2px; }
.header-banner p  { margin: 6px 0 0; opacity: .92; }

.card {
    background: white; border-radius: 14px; padding: 18px 18px 14px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 12px rgba(0,0,0,0.045);
}
.card h3 { margin: 0 0 10px; font-size: 16px; color: #14324a; }
.mono { font-family: 'DM Mono', monospace; font-size: 12px; }

.badge {
    display:inline-block; padding: 2px 10px; border-radius: 999px;
    background: rgba(13,92,58,0.12); color: #0d5c3a; font-size: 12px;
    border: 1px solid rgba(13,92,58,0.20);
    margin-left: 8px;
}
.small-note { color: rgba(0,0,0,0.55); font-size: 12px; }
hr { border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 18px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header-banner">
  <h1>InsureIQ <span class="badge">RAG Claims Assistant</span></h1>
  <p>Ask questions about policy documents. Answers cite the retrieved sources.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar — API + options
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

QUERY_LIMIT = 10
st.session_state.setdefault("query_count", 0)

st.sidebar.markdown("**Session query limit:**")
st.sidebar.progress(min(st.session_state.query_count / QUERY_LIMIT, 1.0))
st.sidebar.caption(f"{st.session_state.query_count}/{QUERY_LIMIT} queries used")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Anthropic API Key")

api_key = None
if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
elif os.getenv("ANTHROPIC_API_KEY"):
    api_key = os.getenv("ANTHROPIC_API_KEY")

manual_key = st.sidebar.text_input(
    "Enter API key (optional)",
    type="password",
    help="If not set in Streamlit Secrets, you can paste it here for this session.",
)
if manual_key.strip():
    api_key = manual_key.strip()

st.sidebar.markdown("### 🧠 Model")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
st.sidebar.code(MODEL)

st.sidebar.markdown("### 🔎 Retrieval")
TOP_K = st.sidebar.slider("Top-k sources", 3, 15, 8, 1)
MIN_SCORE = st.sidebar.slider(
    "Min relevance (BM25 score)", 0.0, 15.0, 0.0, 0.5,
    help="Keep at 0.0 for demo reliability; raise if you want stricter evidence."
)

st.sidebar.markdown("### 📈 Demo Visual")
show_workflow = st.sidebar.toggle("Show Workflow Demo (Interactive)", value=False)

# Optional debug helpers
show_pipeline = st.sidebar.toggle("Show pipeline log", value=True)
show_doc_preview = st.sidebar.toggle("Show source previews", value=True)

# ─────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WORKFLOW_HTML_PATH = BASE_DIR / "ai_workflow_visuals.html"

# ─────────────────────────────────────────────
# Workflow demo renderer
# ─────────────────────────────────────────────
def render_workflow_demo(height: int = 900) -> None:
    if not WORKFLOW_HTML_PATH.exists():
        st.warning("Workflow demo HTML not found: ai_workflow_visuals.html")
        return
    html = WORKFLOW_HTML_PATH.read_text(encoding="utf-8")
    components.html(html, height=height, scrolling=True)


# ─────────────────────────────────────────────
# Tokenization + BM25 retrieval (much stronger than token overlap)
# ─────────────────────────────────────────────
STOPWORDS = set("""
a an the and or but if then else this that those these to from in on at for of with without
is are was were be been being as by it its they them their you your i we our he she his her
""".split())

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return [t for t in tokens if t and t not in STOPWORDS]

def bm25_build(docs: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
    """
    Returns a lightweight BM25 index:
      - doc_tokens: list[list[str]]
      - doc_freq: term -> df
      - idf: term -> idf
      - doc_tf: list[dict[str,int]]
      - doc_len: list[int]
      - avgdl: float
    """
    doc_tokens: List[List[str]] = []
    doc_tf: List[Dict[str, int]] = []
    doc_len: List[int] = []
    df: Dict[str, int] = {}

    for d in docs:
        toks = tokenize(d.get("text", ""))
        doc_tokens.append(toks)
        doc_len.append(len(toks))
        tf: Dict[str, int] = {}
        seen = set()
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
        doc_tf.append(tf)

    N = max(1, len(docs))
    avgdl = (sum(doc_len) / N) if N else 0.0

    idf: Dict[str, float] = {}
    for term, dfi in df.items():
        # classic BM25 idf with +0.5 smoothing
        idf[term] = math.log(1 + (N - dfi + 0.5) / (dfi + 0.5))

    return {
        "k1": k1,
        "b": b,
        "doc_tf": doc_tf,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "idf": idf,
    }

def bm25_score(query: str, idx, doc_i: int) -> float:
    q_terms = tokenize(query)
    if not q_terms:
        return 0.0

    k1 = idx["k1"]
    b = idx["b"]
    tf = idx["doc_tf"][doc_i]
    dl = idx["doc_len"][doc_i]
    avgdl = idx["avgdl"] or 1.0
    idf = idx["idf"]

    score = 0.0
    for term in q_terms:
        if term not in tf:
            continue
        f = tf[term]
        w = idf.get(term, 0.0)
        denom = f + k1 * (1 - b + b * (dl / avgdl))
        score += w * (f * (k1 + 1)) / (denom if denom else 1.0)
    return float(score)

def retrieve_bm25(query: str, docs: List[Dict[str, Any]], idx, top_k: int) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, int]] = []
    for i in range(len(docs)):
        s = bm25_score(query, idx, i)
        scored.append((s, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, i in scored[:top_k]:
        d = docs[i]
        out.append({
            "id": d.get("id", i),
            "title": d.get("title", f"Doc {i+1}"),
            "text": d.get("text", ""),
            "score": float(s),
            "source": d.get("source", d.get("title", f"Doc {i+1}")),
        })
    return out


# ─────────────────────────────────────────────
# JSON → chunking (so your quick questions actually match something)
# ─────────────────────────────────────────────
def flatten_json(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """
    Flattens nested dict/list into (path, text) pairs.
    Keeps paths to preserve meaning and improve retrieval.
    """
    items: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            items.extend(flatten_json(v, new_prefix))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            items.extend(flatten_json(v, new_prefix))
    else:
        # primitive
        txt = "" if obj is None else str(obj)
        items.append((prefix, txt))
    return items

def chunk_pairs(pairs: List[Tuple[str, str]], max_chars: int = 1800) -> List[str]:
    """
    Builds readable chunks like:
      path: value
    with a size cap for RAG prompt safety.
    """
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for path, val in pairs:
        line = f"{path}: {val}".strip()
        if not line or line.endswith(":"):
            continue
        # avoid extremely long values collapsing the UI
        if len(line) > 1000:
            line = line[:1000] + "…"
        add_len = len(line) + 1
        if buf and (cur + add_len) > max_chars:
            chunks.append("\n".join(buf))
            buf = []
            cur = 0
        buf.append(line)
        cur += add_len
    if buf:
        chunks.append("\n".join(buf))
    return chunks

@st.cache_data(show_spinner=False)
def load_policy_corpus() -> List[Dict[str, Any]]:
    """
    Loads all *.json files in the same folder as app.py
    and converts each into multiple chunks for retrieval.
    """
    corpus: List[Dict[str, Any]] = []
    json_files = sorted([p for p in BASE_DIR.glob("*.json") if p.is_file()])

    for fp in json_files:
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            # If JSON parsing fails, skip quietly (keeps the app resilient)
            continue

        title = fp.stem.replace("_", " ").title()

        # If file is already an array of “chunk-like” objects
        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and ("text" in raw[0] or "content" in raw[0]):
            for i, item in enumerate(raw):
                txt = item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)
                corpus.append({
                    "id": f"{fp.stem}_{i}",
                    "title": title,
                    "text": str(txt),
                    "source": fp.name
                })
            continue

        # Otherwise flatten and chunk
        pairs = flatten_json(raw)
        chunks = chunk_pairs(pairs, max_chars=1800)

        # If flattening produced too little (rare), fallback to whole dump
        if not chunks:
            chunks = [json.dumps(raw, indent=2, ensure_ascii=False)]

        for i, ch in enumerate(chunks):
            corpus.append({
                "id": f"{fp.stem}_{i}",
                "title": title,
                "text": ch,
                "source": fp.name
            })

    return corpus

@st.cache_data(show_spinner=False)
def build_retrieval_index(corpus: List[Dict[str, Any]]):
    return bm25_build(corpus)

corpus = load_policy_corpus()
bm25_idx = build_retrieval_index(corpus) if corpus else None


# ─────────────────────────────────────────────
# Prompt context builder (tight & safe)
# ─────────────────────────────────────────────
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))

def build_context(retrieved: List[Dict[str, Any]]) -> str:
    parts = []
    for idx, r in enumerate(retrieved, start=1):
        title = r.get("title", f"Source {idx}")
        source = r.get("source", title)
        text = (r.get("text") or "").strip()
        parts.append(
            f"[SOURCE {idx}] {title}\n"
            f"(origin: {source})\n"
            f"{text}\n"
        )
    joined = "\n---\n".join(parts)
    if len(joined) > MAX_CONTEXT_CHARS:
        joined = joined[:MAX_CONTEXT_CHARS] + "\n\n[TRUNCATED: context shortened for model input safety]"
    return joined


# ─────────────────────────────────────────────
# LLM call with robust error surfacing (no more “mystery redactions”)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are InsureIQ, an expert insurance claims specialist assistant.

You MUST answer using ONLY the provided policy documents.
If the documents do not contain the answer, say so clearly and ask what document or detail is missing.

Requirements:
1) Directly answer the member’s question.
2) Cite evidence using [SOURCE N] markers.
3) If relevant, explain deductible/coinsurance/copay impacts.
4) Call out exclusions, pre-auth, in-network vs out-of-network conditions when applicable.
5) If you do any calculation, show the math.
6) Keep the tone precise, practical, and non-speculative.
"""

def generate_answer(query: str, retrieved: List[Dict[str, Any]], client: Anthropic) -> str:
    context = build_context(retrieved)
    user_message = f"""Member Question: {query}

Relevant Policy Documents:
{context}

Return a structured answer with citations to the sources above."""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=900,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text
    except anthropic.APIStatusError as e:
        # Shows actual status code + whatever the SDK exposes (better than Streamlit’s redaction alone)
        st.error(f"Anthropic API error {e.status_code}: {getattr(e, 'message', str(e))}")
        # If Streamlit Cloud logs are available, this typically contains a helpful payload
        try:
            st.code(getattr(e, "response", None))
        except Exception:
            pass
        raise
    except anthropic.APIError as e:
        st.error(f"Anthropic API error: {str(e)}")
        raise


# ─────────────────────────────────────────────
# Sample queries (your “quick questions” + complex)
# ─────────────────────────────────────────────
SAMPLE_QUERIES = [
    {"label": "🏥 In-network vs out-of-network cost", "q": "What is my out-of-network financial exposure for a surgery? Include balance billing risk and any caps, if stated."},
    {"label": "💊 Prescription coverage question", "q": "Does the plan cover insulin and what are the copays? Also note any formulary or prior authorization requirements."},
    {"label": "🧾 Deductible & coinsurance", "q": "How do deductible and coinsurance apply for an MRI? Explain what I pay before and after the deductible is met."},
    {"label": "🚑 ER coverage", "q": "If I go to an out-of-network emergency room, is it covered as in-network? What conditions apply?"},
]


# ─────────────────────────────────────────────
# Main layout (NO tabs → avoids empty tab bars)
# ─────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask a policy question")

    # Sample query buttons
    sq_cols = st.columns(2)
    for i, item in enumerate(SAMPLE_QUERIES):
        with sq_cols[i % 2]:
            if st.button(item["label"], use_container_width=True):
                st.session_state["query"] = item["q"]

    query_input = st.text_area(
        "Your question",
        value=st.session_state.get("query", ""),
        height=120,
        placeholder="E.g., Is a knee replacement covered out-of-network? What will I owe?",
    )

    run_btn = st.button("🔎 Retrieve + Answer", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown("### 📚 Corpus status")
    if corpus:
        st.write(f"Loaded **{len(corpus)}** chunks from JSON files in this repo folder.")
        st.caption("This app reads all *.json files next to app.py and auto-chunks them for retrieval.")
        with st.expander("Show loaded JSON files", expanded=False):
            files = [p.name for p in sorted(BASE_DIR.glob("*.json"))]
            st.write(files)
    else:
        st.warning("No JSON corpus loaded. Ensure your policy JSON files are in the same folder as app.py.")
    st.markdown("</div>", unsafe_allow_html=True)

    if show_workflow:
        st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown("### 📈 Workflow Demo (Interactive)")
        render_workflow_demo(height=860)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Answer")
    if st.session_state.get("answer"):
        st.markdown(st.session_state["answer"])
    else:
        st.info("Ask a question to generate an answer.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown("### 🧩 Retrieved Sources")
    retrieved_state = st.session_state.get("retrieved", [])
    if retrieved_state:
        for i, r in enumerate(retrieved_state, start=1):
            st.markdown(
                f"**[SOURCE {i}] {r.get('title','')}**  \n"
                f"<span class='mono'>score={r.get('score',0):.3f} | origin={r.get('source','')}</span>",
                unsafe_allow_html=True,
            )
            if show_doc_preview:
                txt = (r.get("text") or "").strip()
                st.caption(txt[:340] + ("…" if len(txt) > 340 else ""))
            st.write("")
    else:
        st.warning("No sources retrieved yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    if show_pipeline:
        st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown("### 🧪 Pipeline Log")
        if st.session_state.get("pipeline_log"):
            st.json(st.session_state["pipeline_log"])
        else:
            st.caption("Pipeline steps will appear after you run a query.")
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────
if run_btn:
    st.session_state["query"] = query_input

    if not api_key:
        st.error("⚠️ No API key configured. Add ANTHROPIC_API_KEY to Streamlit Secrets or enter it in the sidebar.")
    elif st.session_state.query_count >= QUERY_LIMIT:
        st.warning(f"⏳ You’ve reached the session limit of {QUERY_LIMIT} queries. Refresh the page to start a new session.")
    elif not corpus or not bm25_idx:
        st.error("No policy documents loaded. Add your JSON policy files next to app.py.")
    else:
        st.session_state.query_count += 1

        t0 = time.time()
        pipeline_log = []

        # Step 1 — retrieve
        retrieved = retrieve_bm25(query_input, corpus, bm25_idx, top_k=TOP_K)
        # Score filter (keep 0.0 by default for demo reliability)
        retrieved_f = [r for r in retrieved if r["score"] >= MIN_SCORE]

        pipeline_log.append({
            "step": "retrieve_bm25",
            "requested_top_k": TOP_K,
            "min_score": MIN_SCORE,
            "returned": len(retrieved),
            "kept_after_filter": len(retrieved_f),
            "top_scores": [round(r["score"], 3) for r in retrieved[:min(5, len(retrieved))]],
        })

        st.session_state["retrieved"] = retrieved_f

        # Step 2 — generate (only if some evidence exists)
        if not retrieved_f:
            st.session_state["answer"] = "No answer generated because no sources were retrieved with sufficient score."
            pipeline_log.append({"step": "generate", "status": "skipped_no_sources"})
        else:
            client = Anthropic(api_key=api_key)
            try:
                answer = generate_answer(query_input, retrieved_f, client)
                st.session_state["answer"] = answer
                pipeline_log.append({"step": "generate", "status": "ok"})
            except Exception:
                # Error already shown in UI by generate_answer
                st.session_state["answer"] = "Generation failed. See the error above for details."
                pipeline_log.append({"step": "generate", "status": "failed"})

        pipeline_log.append({
            "step": "timing",
            "seconds_total": round(time.time() - t0, 3),
        })
        st.session_state["pipeline_log"] = pipeline_log

        # force UI refresh with updated session state
        st.rerun()


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "Tip: If Streamlit still redacts errors, set the environment variable ANTHROPIC_LOG=debug "
    "and check your deployment logs for the underlying API response."
)
