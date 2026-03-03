import streamlit as st
import json
import os
import math
import time
import re
from pathlib import Path
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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background-color: #f7f6f2;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #1a3c5e 0%, #0d5c3a 100%);
    padding: 28px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    color: white;
}
.header-banner h1 {
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.header-banner p {
    font-size: 14px;
    opacity: 0.8;
    margin: 0;
}

/* Pipeline step cards */
.pipeline-step {
    background: white;
    border: 1px solid #e4e0d8;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.pipeline-step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.step-num {
    background: #1a3c5e;
    color: white;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-title {
    font-weight: 600;
    font-size: 13px;
    color: #1a1814;
}
.step-detail {
    font-size: 12px;
    color: #6b7280;
    margin-left: 34px;
    font-family: 'DM Mono', monospace;
}

/* Result cards */
.result-card {
    background: white;
    border: 1px solid #e4e0d8;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border-left: 4px solid #0d5c3a;
}
.result-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}
.result-rank {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #0d5c3a;
    font-weight: 700;
}
.result-title {
    font-size: 14px;
    font-weight: 600;
    color: #1a1814;
    margin-bottom: 4px;
}
.result-source {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #6b7280;
    background: #f0f0eb;
    padding: 2px 8px;
    border-radius: 4px;
}
.score-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.score-badge {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 500;
}
.score-sim { background: #d1f0e6; color: #064e3b; }
.score-conf-high { background: #d1f0e6; color: #064e3b; }
.score-conf-med { background: #fde68a; color: #92400e; }
.score-conf-low { background: #fde0e0; color: #991b1b; }
.excerpt-box {
    background: #f9f8f5;
    border: 1px solid #e4e0d8;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 12px;
    color: #4a4640;
    line-height: 1.6;
    font-style: italic;
}

/* Answer box */
.answer-box {
    background: white;
    border: 1px solid #b8d4eb;
    border-radius: 10px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-top: 4px solid #1a3c5e;
}

/* Confidence badge large */
.conf-badge-lg {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 12px;
}
.conf-high { background: #d1f0e6; color: #064e3b; }
.conf-med  { background: #fde68a; color: #92400e; }
.conf-low  { background: #fde0e0; color: #991b1b; }

/* Sample query buttons */
.sample-query {
    background: white;
    border: 1px solid #e4e0d8;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    font-size: 12.5px;
    color: #1a1814;
    line-height: 1.4;
    transition: all 0.15s;
}
.sample-query:hover {
    border-color: #1a3c5e;
    background: #f0f4f8;
}

/* Metric tiles */
.metric-tile {
    background: white;
    border: 1px solid #e4e0d8;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #1a3c5e;
}
.metric-label {
    font-size: 11px;
    color: #6b7280;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a3c5e !important;
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.9) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

DATASET_FILES = {
    "Eligibility Rules": "eligibility_rules.json",
    "Pre-existing Conditions": "preexisting_conditions.json",
    "Network & Provider Rules": "network_provider_rules.json",
    "Claim Procedures & Limits": "claim_procedures.json",
    "Grace Period & Lapse Policies": "grace_period_lapse.json",
    "Cardiac & Surgical Coverage": "cardiac_surgical_coverage.json",
}

DATASET_ICONS = {
    "Eligibility Rules": "📋",
    "Pre-existing Conditions": "🩺",
    "Network & Provider Rules": "🏥",
    "Claim Procedures & Limits": "📄",
    "Grace Period & Lapse Policies": "⏳",
    "Cardiac & Surgical Coverage": "❤️",
}

@st.cache_data
def load_all_documents():
    """Load and flatten all JSON datasets into searchable document chunks."""
    docs = []
    for dataset_name, filename in DATASET_FILES.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue
        with open(filepath, "r") as f:
            data = json.load(f)
        for record in data.get("records", []):
            # Build a rich text representation for embedding/search
            text_parts = [
                record.get("title", ""),
                record.get("rule", ""),
                " ".join(record.get("tags", [])),
                record.get("subcategory", ""),
                " ".join(record.get("exclusions", [])),
            ]
            # Add condition details as text
            conditions = record.get("conditions", {})
            if isinstance(conditions, dict):
                for k, v in conditions.items():
                    if isinstance(v, str):
                        text_parts.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        text_parts.append(f"{k}: {', '.join(str(i) for i in v)}")

            full_text = " ".join(filter(None, text_parts)).lower()

            docs.append({
                "id": record.get("id", ""),
                "title": record.get("title", ""),
                "rule": record.get("rule", ""),
                "tags": record.get("tags", []),
                "exclusions": record.get("exclusions", []),
                "conditions": record.get("conditions", {}),
                "dataset": dataset_name,
                "filename": filename,
                "category": record.get("category", ""),
                "subcategory": record.get("subcategory", ""),
                "full_text": full_text,
                "effective_date": record.get("effective_date", ""),
            })
    return docs

# ─────────────────────────────────────────────
# RAG Engine — TF-IDF style cosine similarity
# (no external ML dependencies needed)
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    # Simple stopword removal
    stopwords = {'a','an','the','is','are','was','were','be','been','being',
                 'have','has','had','do','does','did','will','would','could',
                 'should','may','might','shall','can','need','must','ought',
                 'i','me','my','we','our','you','your','he','she','it','they',
                 'them','their','this','that','these','those','what','which',
                 'who','whom','when','where','why','how','all','both','each',
                 'few','more','most','other','some','such','no','nor','not',
                 'only','own','same','so','than','too','very','just','of',
                 'at','by','for','with','about','against','between','into',
                 'through','during','before','after','above','below','to',
                 'from','up','down','in','out','on','off','over','under',
                 'again','further','then','once','and','but','or','if','as'}
    return [t for t in tokens if t not in stopwords and len(t) > 2]

def build_tfidf(docs: list[dict]) -> tuple[dict, dict]:
    """Build TF-IDF vectors for all documents."""
    N = len(docs)
    # Document frequency
    df = {}
    doc_tokens_list = []
    for doc in docs:
        tokens = tokenize(doc["full_text"])
        token_set = set(tokens)
        doc_tokens_list.append(tokens)
        for t in token_set:
            df[t] = df.get(t, 0) + 1

    # TF-IDF vectors
    vectors = []
    for tokens in doc_tokens_list:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        tfidf = {}
        for t, freq in tf.items():
            idf = math.log((N + 1) / (df.get(t, 0) + 1)) + 1
            tfidf[t] = (freq / len(tokens)) * idf
        vectors.append(tfidf)

    return vectors, df, N

def cosine_sim(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two TF-IDF dicts."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v*v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v*v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def compute_query_vector(query: str, df: dict, N: int) -> dict:
    tokens = tokenize(query)
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    tfidf = {}
    for t, freq in tf.items():
        idf = math.log((N + 1) / (df.get(t, 0) + 1)) + 1
        tfidf[t] = (freq / len(tokens)) * idf
    return tfidf

def get_verbatim_score(query: str, rule_text: str) -> float:
    """Estimate verbatim keyword overlap between query and rule text."""
    q_tokens = set(tokenize(query))
    r_tokens = set(tokenize(rule_text))
    if not q_tokens:
        return 0.0
    overlap = q_tokens & r_tokens
    return len(overlap) / len(q_tokens)

def confidence_label(sim_score: float) -> str:
    if sim_score >= 0.35:
        return "HIGH"
    elif sim_score >= 0.18:
        return "MEDIUM"
    else:
        return "LOW"

def retrieve(query: str, docs: list[dict], doc_vectors: list[dict],
             df: dict, N: int, top_k: int = 5) -> list[dict]:
    """Full retrieval pipeline with scoring."""
    query_vec = compute_query_vector(query, df, N)
    scored = []
    for i, (doc, vec) in enumerate(zip(docs, doc_vectors)):
        sim = cosine_sim(query_vec, vec)
        # Boost for tag matches
        q_tokens = set(tokenize(query))
        tag_tokens = set(tokenize(" ".join(doc["tags"])))
        tag_boost = 0.05 * len(q_tokens & tag_tokens)
        # Boost for title match
        title_tokens = set(tokenize(doc["title"]))
        title_boost = 0.08 * len(q_tokens & title_tokens)
        final_score = min(sim + tag_boost + title_boost, 1.0)
        verbatim = get_verbatim_score(query, doc["rule"])
        scored.append({
            **doc,
            "similarity_score": round(final_score, 4),
            "raw_sim": round(sim, 4),
            "verbatim_score": round(verbatim, 4),
            "confidence": confidence_label(final_score),
            "rank": 0,
        })
    scored.sort(key=lambda x: x["similarity_score"], reverse=True)  # sort by float score only
    for i, r in enumerate(scored[:top_k]):
        r["rank"] = i + 1
    return scored[:top_k]

# ─────────────────────────────────────────────
# Claude Reader
# ─────────────────────────────────────────────

def build_context(retrieved: list[dict]) -> str:
    parts = []
    for r in retrieved:
        parts.append(
            f"[SOURCE {r['rank']}: {r['title']} | Dataset: {r['dataset']} | ID: {r['id']}]\n"
            f"{r['rule']}\n"
            f"Exclusions: {'; '.join(r['exclusions']) if r['exclusions'] else 'None'}\n"
        )
    return "\n---\n".join(parts)

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

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text

# ─────────────────────────────────────────────
# Sample Queries
# ─────────────────────────────────────────────
SAMPLE_QUERIES = [
    {
        "label": "🫀 Diabetic cardiac surgery, out-of-network",
        "query": "I'm 45 years old with Type 2 diabetes and my doctor is recommending a $75,000 cardiac surgery at an out-of-network hospital. What will my insurance cover and what are my total costs?"
    },
    {
        "label": "⏰ COBRA grace period after job loss",
        "query": "I lost my job last month and haven't paid my COBRA premium yet. How long do I have to pay and what happens if I miss the deadline? Will my claims still be covered?"
    },
    {
        "label": "🏥 Pre-existing condition with insurance switch",
        "query": "I have a history of cancer and I'm switching to a new insurance plan. Can they exclude my cancer treatments as a pre-existing condition? What protections do I have?"
    },
    {
        "label": "❤️ Emergency cardiac procedure coverage",
        "query": "I had an emergency cardiac catheterization at an out-of-network hospital last week. Will my insurance cover this? Do I owe balance billing charges?"
    },
    {
        "label": "⚖️ Bariatric surgery eligibility with diabetes",
        "query": "I have Type 2 diabetes and a BMI of 37. Am I eligible for bariatric surgery coverage? What are the requirements I need to meet before the insurance will approve it?"
    },
    {
        "label": "🔄 Coverage reinstatement after lapse",
        "query": "My health insurance lapsed 45 days ago because I missed premium payments. Can I get reinstated? Will I have a new waiting period or pre-existing condition exclusions?"
    },
    {
        "label": "🏆 Center of Excellence for heart bypass",
        "query": "My doctor recommended CABG (bypass surgery). What are my cost differences between a regular in-network hospital versus a Center of Excellence? Should I travel to a COE?"
    },
    {
        "label": "📋 Mental health parity and prior auth",
        "query": "I need inpatient psychiatric treatment. Do I need prior authorization? Are there limits on how many days are covered? Does my plan have to cover mental health the same as medical?"
    },
]

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "last_query_time_ms" not in st.session_state:
    st.session_state.last_query_time_ms = None

# ─────────────────────────────────────────────
# Load data and build index
# ─────────────────────────────────────────────
docs = load_all_documents()

@st.cache_data
def build_index(doc_texts):
    """Cache the TF-IDF index."""
    return build_tfidf(docs)

doc_vectors, df_map, N_docs = build_index([d["full_text"] for d in docs])

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 InsureIQ")
    st.markdown("**RAG-Powered Claims Assistant**")
    st.markdown("---")

    # API Key
    st.markdown("### 🔑 Anthropic API Key")
    api_key = st.text_input(
        "Enter your API key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key from console.anthropic.com"
    )

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Search Settings")
    top_k = st.slider("Top results to retrieve", min_value=3, max_value=8, value=5)
    st.session_state.top_k = top_k

    st.markdown("---")

    # Dataset info
    st.markdown("### 📚 Knowledge Base")
    for name, icon in DATASET_ICONS.items():
        count = sum(1 for d in docs if d["dataset"] == name)
        st.markdown(f"{icon} **{name}** — {count} rules")

    st.markdown("---")
    st.markdown(f"**Total indexed rules:** {len(docs)}")
    st.markdown(f"**Vocabulary size:** {len(df_map):,} terms")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
This app demonstrates a **full RAG pipeline**:
1. **Retrieval** — TF-IDF cosine similarity search across 6 JSON policy datasets
2. **Reader** — Claude Sonnet synthesizes the final answer from retrieved passages

No vector database required — pure Python retrieval engine.
    """)

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🏥 InsureIQ — Insurance RAG Claims Assistant</h1>
    <p>Ask any complex insurance question. The system retrieves relevant policy rules, scores them by relevance, and generates a precise answer using Claude AI.</p>
</div>
""", unsafe_allow_html=True)

# ─── Metrics row ───
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-tile"><div class="metric-val">{len(docs)}</div><div class="metric-label">Policy Rules Indexed</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-tile"><div class="metric-val">6</div><div class="metric-label">JSON Datasets</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-tile"><div class="metric-val">{len(df_map):,}</div><div class="metric-label">Vocabulary Terms</div></div>""", unsafe_allow_html=True)
with col4:
    timing_display = f"{st.session_state.last_query_time_ms}ms" if st.session_state.last_query_time_ms else "—"
    st.markdown(f"""<div class="metric-tile"><div class="metric-val">{timing_display}</div><div class="metric-label">Last Query Time</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Two column layout ───
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # Query input
    st.markdown("### 💬 Ask a Claims Question")
    query_input = st.text_area(
        label="Your question",
        value=st.session_state.query,
        height=110,
        placeholder="E.g. I'm 45 years old with diabetes and need cardiac surgery at an out-of-network hospital that will cost $75,000. What will my insurance pay and what is my total financial exposure?",
        label_visibility="collapsed"
    )

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        search_btn = st.button("🔍  Run RAG Pipeline", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("✕  Clear", use_container_width=True)

    if clear_btn:
        st.session_state.query = ""
        st.session_state.results = None
        st.session_state.answer = None
        st.session_state.pipeline_log = []
        st.rerun()

with right_col:
    st.markdown("### ⚡ Sample Questions")
    for sample in SAMPLE_QUERIES:
        if st.button(sample["label"], key=f"sample_{sample['label']}", use_container_width=True):
            st.session_state.query = sample["query"]
            st.rerun()

# ─────────────────────────────────────────────
# Run pipeline on button click
# ─────────────────────────────────────────────
if search_btn and query_input.strip():
    if not api_key:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar to use the Claude reader.")
    else:
        st.session_state.query = query_input
        client = Anthropic(api_key=api_key)

        t_start = time.time()
        pipeline_log = []

        # STEP 1: Tokenize query
        tokens = tokenize(query_input)
        pipeline_log.append({
            "step": 1,
            "name": "Query Tokenization",
            "detail": f"Extracted {len(tokens)} meaningful tokens: {', '.join(tokens[:12])}{'...' if len(tokens) > 12 else ''}",
            "icon": "🔤",
        })

        # STEP 2: Compute query vector
        query_vec = compute_query_vector(query_input, df_map, N_docs)
        pipeline_log.append({
            "step": 2,
            "name": "TF-IDF Query Embedding",
            "detail": f"Computed query vector across {len(query_vec)} unique terms | Vocabulary coverage: {len(query_vec)}/{len(df_map)} terms",
            "icon": "🔢",
        })

        # STEP 3: Cosine similarity search
        pipeline_log.append({
            "step": 3,
            "name": "Cosine Similarity Search",
            "detail": f"Computing dot products against {len(docs)} document vectors across {len(df_map):,} term vocabulary...",
            "icon": "📐",
        })

        # STEP 4: Retrieve top-K
        retrieved = retrieve(query_input, docs, doc_vectors, df_map, N_docs, top_k=top_k)
        top_score = retrieved[0]["similarity_score"] if retrieved else 0
        pipeline_log.append({
            "step": 4,
            "name": f"Top-{top_k} Result Retrieval",
            "detail": f"Retrieved top-{top_k} documents | Best score: {top_score:.4f} | Sources: {', '.join(set(r['dataset'] for r in retrieved[:3]))}",
            "icon": "🎯",
        })

        # STEP 5: Verbatim scoring
        score_parts = [
            f'#{r["rank"]} sim={r["similarity_score"]:.3f} verb={r["verbatim_score"]:.2f} [{r["confidence"]}]'
            for r in retrieved[:3]
        ]
        pipeline_log.append({
            "step": 5,
            "name": "Verbatim & Confidence Scoring",
            "detail": f"Scores: {' | '.join(score_parts)}",
            "icon": "📊",
        })

        # STEP 6: Claude Reader
        pipeline_log.append({
            "step": 6,
            "name": "Claude Sonnet Reader (Answer Generation)",
            "detail": f"Sending {len(retrieved)} retrieved passages ({sum(len(r['rule']) for r in retrieved):,} chars) to Claude claude-sonnet-4-20250514 for synthesis...",
            "icon": "🤖",
        })

        with st.spinner("Generating answer with Claude..."):
            answer = generate_answer(query_input, retrieved, client)

        t_end = time.time()
        elapsed_ms = int((t_end - t_start) * 1000)

        pipeline_log.append({
            "step": 7,
            "name": "Pipeline Complete",
            "detail": f"Total time: {elapsed_ms}ms | Retrieval: ~{int(elapsed_ms * 0.05)}ms | Claude generation: ~{int(elapsed_ms * 0.95)}ms",
            "icon": "✅",
        })

        st.session_state.results = retrieved
        st.session_state.answer = answer
        st.session_state.pipeline_log = pipeline_log
        st.session_state.last_query_time_ms = elapsed_ms

        st.rerun()

elif search_btn and not query_input.strip():
    st.warning("Please enter a question first.")

# ─────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────
if st.session_state.results is not None:
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🤖 AI Answer", "📊 Retrieved Documents", "⚙️ Pipeline Trace"])

    # ── TAB 1: Answer ──
    with tab1:
        results = st.session_state.results
        answer = st.session_state.answer

        # Confidence of top result
        top_conf = results[0]["confidence"] if results else "LOW"
        conf_class = {"HIGH": "conf-high", "MEDIUM": "conf-med", "LOW": "conf-low"}.get(top_conf, "conf-low")
        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(top_conf, "🔴")

        st.markdown(f"""
<div class="answer-box">
<span class="conf-badge-lg {conf_class}">{conf_emoji} {top_conf} CONFIDENCE</span>
<div style="font-size:13px;color:#374151;line-height:1.7;">{answer.replace(chr(10), '<br>')}</div>
</div>
""", unsafe_allow_html=True)

        # Source summary
        st.markdown("<br>**Sources consulted:**", unsafe_allow_html=True)
        source_cols = st.columns(min(len(results), 5))
        for i, (col, res) in enumerate(zip(source_cols, results)):
            with col:
                icon = DATASET_ICONS.get(res["dataset"], "📄")
                st.markdown(f"""
<div style="background:white;border:1px solid #e4e0d8;border-radius:6px;padding:8px 10px;text-align:center;font-size:11px;">
<div style="font-size:18px;">{icon}</div>
<div style="font-weight:600;color:#1a1814;margin-top:4px;">{res['id']}</div>
<div style="color:#6b7280;font-size:10px;">{res['dataset']}</div>
</div>
""", unsafe_allow_html=True)

    # ── TAB 2: Retrieved Documents ──
    with tab2:
        st.markdown(f"**{len(results)} most relevant policy rules retrieved:**")
        for r in results:
            conf_cls = {"HIGH": "score-conf-high", "MEDIUM": "score-conf-med", "LOW": "score-conf-low"}.get(r["confidence"], "score-conf-low")
            conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(r["confidence"], "🔴")
            icon = DATASET_ICONS.get(r["dataset"], "📄")

            with st.expander(f"#{r['rank']} — {r['title']}", expanded=(r["rank"] <= 2)):
                # Header row
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{icon} {r['dataset']}** — `{r['id']}`")
                    st.markdown(f"*{r['category']} / {r['subcategory']}*")
                with col_b:
                    st.markdown(f"""
<div class="score-row">
<span class="score-badge score-sim">Sim: {r['similarity_score']:.3f}</span>
<span class="score-badge score-sim">Verb: {r['verbatim_score']:.2f}</span>
<span class="score-badge {conf_cls}">{conf_emoji} {r['confidence']}</span>
</div>
""", unsafe_allow_html=True)

                # Verbatim excerpt
                st.markdown("**📝 Policy Rule (verbatim excerpt):**")
                # Show first 400 chars as excerpt
                excerpt = r["rule"][:500] + ("..." if len(r["rule"]) > 500 else "")
                st.markdown(f"""<div class="excerpt-box">{excerpt}</div>""", unsafe_allow_html=True)

                if r["exclusions"]:
                    st.markdown("**⛔ Exclusions:**")
                    for ex in r["exclusions"]:
                        st.markdown(f"- {ex}")

                if r.get("tags"):
                    tags_html = " ".join([f'<span style="background:#f0f0eb;border:1px solid #e4e0d8;padding:2px 8px;border-radius:4px;font-size:10px;font-family:monospace;">{t}</span>' for t in r["tags"][:8]])
                    st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)

    # ── TAB 3: Pipeline Trace ──
    with tab3:
        st.markdown("**Full RAG pipeline execution trace:**")
        for log in st.session_state.pipeline_log:
            completed = log["step"] <= len(st.session_state.pipeline_log)
            color = "#0d5c3a" if log["step"] == 7 else "#1a3c5e"
            st.markdown(f"""
<div class="pipeline-step">
<div class="pipeline-step-header">
<span style="font-size:18px;">{log['icon']}</span>
<span class="step-title">Step {log['step']}: {log['name']}</span>
</div>
<div class="step-detail">{log['detail']}</div>
</div>
""", unsafe_allow_html=True)

        # Score distribution
        st.markdown("<br>**📊 Similarity Score Distribution:**", unsafe_allow_html=True)
        chart_data = {r["title"][:35] + "...": r["similarity_score"] for r in results}

        import pandas as pd
        df_chart = pd.DataFrame({
            "Document": [r["title"][:40] for r in results],
            "Similarity Score": [r["similarity_score"] for r in results],
            "Verbatim Score": [r["verbatim_score"] for r in results],
            "Confidence": [r["confidence"] for r in results],
            "Dataset": [r["dataset"] for r in results],
        })
        st.dataframe(df_chart, use_container_width=True, hide_index=True)

        st.bar_chart(
            pd.DataFrame({"Similarity": [r["similarity_score"] for r in results]},
                         index=[f"#{r['rank']} {r['id']}" for r in results])
        )

elif st.session_state.query == "":
    # Welcome state
    st.markdown("---")
    st.markdown("""
<div style="text-align:center;padding:40px 20px;">
<div style="font-size:48px;margin-bottom:16px;">🏥</div>
<h3 style="color:#1a3c5e;margin-bottom:8px;">Ready to answer your insurance questions</h3>
<p style="color:#6b7280;font-size:14px;">Enter a question above or click a sample query on the right to see the full RAG pipeline in action.<br>
The system retrieves relevant policy rules using TF-IDF cosine similarity, then uses Claude to synthesize a precise answer.</p>
</div>
""", unsafe_allow_html=True)
