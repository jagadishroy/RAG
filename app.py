import streamlit as st
import streamlit.components.v1 as components
import json
import os
import math
import time
import re
from pathlib import Path
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
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #f7f6f2; }

.header-banner {
    background: linear-gradient(135deg, #1a3c5e 0%, #0d5c3a 100%);
    padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; color: white;
}
.header-banner h1 { font-size: 28px; margin: 0; letter-spacing: .3px; }
.header-banner p  { margin: 6px 0 0; opacity: .9; }

.card {
    background: white; border-radius: 12px; padding: 18px 18px 14px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 10px rgba(0,0,0,0.04);
}
.card h3 { margin: 0 0 10px; font-size: 16px; color: #14324a; }
.mono { font-family: 'DM Mono', monospace; font-size: 12px; }

.badge {
    display:inline-block; padding: 2px 8px; border-radius: 999px;
    background: rgba(13,92,58,0.10); color: #0d5c3a; font-size: 12px;
    border: 1px solid rgba(13,92,58,0.20);
    margin-left: 8px;
}
.pill {
    display:inline-block; padding: 6px 10px; border-radius: 999px;
    background: rgba(26,60,94,0.08); color: #1a3c5e; font-size: 12px;
    border: 1px solid rgba(26,60,94,0.15);
    margin-right: 6px; margin-bottom: 6px;
}

.small-note { color: rgba(0,0,0,0.55); font-size: 12px; }
hr { border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>InsureIQ <span class="badge">RAG Claims Assistant</span></h1>
  <p>Ask questions about policy documents. Answers cite the retrieved sources.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — settings & API key
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

QUERY_LIMIT = 10
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

st.sidebar.markdown("**Session query limit:**")
st.sidebar.progress(min(st.session_state.query_count / QUERY_LIMIT, 1.0))
st.sidebar.caption(f"{st.session_state.query_count}/{QUERY_LIMIT} queries used")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Anthropic API Key")

# Prefer Streamlit Secrets → env var → manual input
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

# Model choice: default to a current stable Sonnet alias; override via env if desired.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))  # rough safety cap

st.sidebar.markdown("### 🧠 Model")
st.sidebar.code(MODEL)

st.sidebar.markdown("### 🧾 Retrieval settings")
TOP_K = st.sidebar.slider("Top-k chunks", min_value=3, max_value=15, value=8, step=1)
MIN_SCORE = st.sidebar.slider("Min score threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

# ─────────────────────────────────────────────
# Data / corpus loading (simple local JSON store)
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CORPUS_PATH = DATA_DIR / "corpus.json"

def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

corpus = load_corpus(DEFAULT_CORPUS_PATH)

# ─────────────────────────────────────────────
# Minimal retrieval utilities (simple token overlap score)
# ─────────────────────────────────────────────
STOPWORDS = set("""
a an the and or but if then else this that those these to from in on at for of with without
is are was were be been being as by it its they them their you your i we our he she his her
""".split())

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return [t for t in tokens if t and t not in STOPWORDS]

def score_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    q = set(query_tokens)
    d = set(doc_tokens)
    return len(q.intersection(d)) / max(1, len(q))

def retrieve(query: str, docs: list[dict], top_k: int = 8) -> list[dict]:
    qt = tokenize(query)
    scored = []
    for i, d in enumerate(docs):
        text = d.get("text", "") or ""
        dt = tokenize(text)
        s = score_overlap(qt, dt)
        scored.append((s, i, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, i, d in scored[:top_k]:
        out.append({
            "id": d.get("id", i),
            "title": d.get("title", f"Document {i+1}"),
            "text": d.get("text", ""),
            "score": float(s),
            "source": d.get("source", d.get("title", f"Document {i+1}")),
        })
    return out

# ─────────────────────────────────────────────
# Prompt context builder
# ─────────────────────────────────────────────
def build_context(retrieved: list[dict]) -> str:
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
    # Rough guardrail: keep prompt sizes manageable (tokenization varies by model)
    if len(joined) > MAX_CONTEXT_CHARS:
        joined = joined[:MAX_CONTEXT_CHARS] + "\n\n[TRUNCATED: context shortened for model input safety]"
    return joined

def generate_answer(query: str, retrieved: list[dict], client: Anthropic) -> str:
    context = build_context(retrieved)
    system_prompt = """You are InsureIQ, an expert insurance claims specialist assistant with deep knowledge of health insurance policy, coverage rules, ACA regulations, and claims procedures.

You answer questions using ONLY the provided policy documents. Your answers must:
1. Directly address the member's specific question
2. Cite which source document(s) you are drawing from (use [SOURCE N] notation)
3. Be clear about financial implications when relevant (deductibles, coinsurance, balance billing)
4. Flag important exclusions or conditions that apply
5. Recommend in-network or Center of Excellence options when appropriate for cost savings
6. Be factual, precise, and helpful — not overly cautious or vague
7. If the answer involves calculations (e.g., OON financial exposure), show the math step-by-step

Format your response with:
- A direct answer to the question
- Supporting details from the policy documents
- Any important caveats or exclusions
- A recommendation if appropriate"""

    user_message = f"""Member Question: {query}

Relevant Policy Documents:
{context}

Please provide a comprehensive, accurate answer based strictly on the above policy documents."""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
    except anthropic.APIStatusError as e:
        # 4xx/5xx with structured payload. Streamlit may redact the original message;
        # surface enough detail to debug without leaking secrets.
        st.error(f"Anthropic API error {e.status_code}: {getattr(e, 'message', str(e))}")
        try:
            st.code(getattr(e, "response", None))
        except Exception:
            pass
        raise
    except anthropic.APIError as e:
        st.error(f"Anthropic API error: {str(e)}")
        raise

# ─────────────────────────────────────────────
# Sample Queries
# ─────────────────────────────────────────────
SAMPLE_QUERIES = [
    {"label": "🏥 In-network vs out-of-network cost", "q": "What is my out-of-network financial exposure for a surgery?"},
    {"label": "💊 Prescription coverage question", "q": "Does the plan cover insulin and what are the copays?"},
    {"label": "🧾 Deductible & coinsurance", "q": "How do deductible and coinsurance apply for an MRI?"},
    {"label": "🚑 ER coverage", "q": "If I go to the emergency room out-of-network, is it covered?"},
]

# ─────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask a policy question")

    # Sample queries
    sq_cols = st.columns(2)
    for i, item in enumerate(SAMPLE_QUERIES):
        with sq_cols[i % 2]:
            if st.button(item["label"]):
                st.session_state.query = item["q"]

    query_input = st.text_area(
        "Your question",
        value=st.session_state.get("query", ""),
        height=120,
        placeholder="E.g., Is a knee replacement covered if done out-of-network? What will I owe?",
    )

    run_btn = st.button("🔎 Retrieve + Answer", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown("### 📚 Corpus status")
    if corpus:
        st.write(f"Loaded **{len(corpus)}** document chunks from `data/corpus.json`.")
        st.caption("If you see weak retrieval, ensure your corpus chunks contain enough policy text and metadata.")
    else:
        st.warning("No corpus loaded. Add `data/corpus.json` with policy chunks (list of {id,title,text,source}).")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Answer")
    answer_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown("### 🧩 Retrieved Sources")
    retrieved_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown("### 🧪 Pipeline Log")
    log_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("⚠️ No API key configured. Add ANTHROPIC_API_KEY to Streamlit Secrets or enter it in the sidebar.")
    elif st.session_state.query_count >= QUERY_LIMIT:
        st.warning(
            f"⏳ You've reached the session limit of {QUERY_LIMIT} queries. "
            "Refresh the page to start a new session."
        )
    else:
        st.session_state.query_count += 1
        st.session_state.query = query_input

        client = Anthropic(api_key=api_key)

        t_start = time.time()
        pipeline_log = []

        # Step 1 — Tokenize
        tokens = tokenize(query_input)
        pipeline_log.append({
            "step": "tokenize",
            "tokens_preview": tokens[:20],
            "token_count": len(tokens),
        })

        # Step 2 — Retrieve
        retrieved = retrieve(query_input, corpus, top_k=TOP_K)
        # Apply score filter
        retrieved_f = [r for r in retrieved if r["score"] >= MIN_SCORE]
        pipeline_log.append({
            "step": "retrieve",
            "requested_top_k": TOP_K,
            "min_score": MIN_SCORE,
            "returned": len(retrieved),
            "kept_after_filter": len(retrieved_f),
            "top_scores": [round(r["score"], 3) for r in retrieved[:min(5, len(retrieved))]],
        })

        # Show retrieved in UI
        if retrieved_f:
            with retrieved_box.container():
                for i, r in enumerate(retrieved_f, start=1):
                    st.markdown(
                        f"**[SOURCE {i}] {r['title']}**  \n"
                        f"<span class='mono'>score={r['score']:.3f} | origin={r.get('source','')}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption((r.get("text", "") or "")[:300] + ("..." if len((r.get("text","") or "")) > 300 else ""))
                    st.write("")
        else:
            retrieved_box.warning("No sources passed the score threshold. Try lowering the min score or improving corpus chunking.")

        # Step 3 — Generate answer (only if we have some evidence)
        if retrieved_f:
            try:
                answer = generate_answer(query_input, retrieved_f, client)
                answer_box.markdown(answer)
            except Exception:
                # Error details are already displayed in generate_answer()
                answer_box.error("Generation failed. See the error above for details.")
        else:
            answer_box.info("No answer generated because no sources were retrieved with sufficient score.")

        pipeline_log.append({
            "step": "timing",
            "seconds_total": round(time.time() - t_start, 3),
        })

        log_box.json(pipeline_log)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "Tip: If you still see redacted Anthropic errors in Streamlit, enable SDK debug logs by setting "
    "`ANTHROPIC_LOG=debug` in your deployment environment variables."
)
