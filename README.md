# 🏥 InsureIQ — RAG-Powered Insurance Claims Assistant

A full **Retrieval-Augmented Generation (RAG)** application for insurance claims Q&A, built with Streamlit and Claude AI.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  RETRIEVAL ENGINE (Pure Python TF-IDF)  │
│  1. Query tokenization & TF-IDF vector  │
│  2. Cosine similarity vs 48 policy docs │
│  3. Tag + title boost scoring           │
│  4. Top-K retrieval + verbatim scoring  │
│  5. Confidence labeling (HIGH/MED/LOW)  │
└─────────────────────────────────────────┘
    │  Top-K passages
    ▼
┌─────────────────────────────────────────┐
│  READER (Claude Sonnet via Anthropic)   │
│  6. Context assembly from passages      │
│  7. Prompted answer synthesis           │
│  8. Source citation in response         │
└─────────────────────────────────────────┘
    │
    ▼
Final Answer + Pipeline Trace + Scores
```

## Knowledge Base — 6 JSON Datasets (48 rules)

| Dataset | Rules | Coverage |
|---------|-------|----------|
| Eligibility Rules | 8 | Age, employment, dependents, HDHP/HSA, reinstatement |
| Pre-existing Conditions | 8 | ACA protections, diabetes, cardiac, cancer, mental health |
| Network & Provider Rules | 8 | In/out-of-network, No Surprises Act, tiered network, COE |
| Claim Procedures & Limits | 8 | Filing deadlines, PA requirements, COB, appeals, subrogation |
| Grace Period & Lapse Policies | 8 | COBRA timelines, grace periods, reinstatement, open enrollment |
| Cardiac & Surgical Coverage | 8 | CABG, TAVR, bariatric, transplant, ICD, PCI, OON calculations |

## Deployment — Streamlit Cloud (Free)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "InsureIQ RAG app"
git remote add origin https://github.com/YOUR_USERNAME/insureiq-rag.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click **"Deploy"**
6. Your shareable URL will be: `https://YOUR_APP_NAME.streamlit.app`

### Step 3: Secrets (Optional — for pre-configured API key)
In Streamlit Cloud dashboard → App Settings → Secrets:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```
Then in `app.py`, add this line after the `api_key = st.text_input(...)` line:
```python
if not api_key:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
```

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
insureiq-rag/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── data/
    ├── eligibility_rules.json      # 8 eligibility rules
    ├── preexisting_conditions.json # 8 pre-existing condition rules
    ├── network_provider_rules.json # 8 network/OON rules
    ├── claim_procedures.json       # 8 claim procedure rules
    ├── grace_period_lapse.json     # 8 grace period rules
    └── cardiac_surgical_coverage.json # 8 cardiac/surgical rules
```

## Sample Questions to Try
- "I'm 45 with diabetes needing $75K cardiac surgery out-of-network. What are my costs?"
- "I lost my job and missed my COBRA payment. What happens to my coverage?"
- "Can a new insurance plan exclude my cancer as a pre-existing condition?"
- "I had an emergency at an out-of-network ER. Am I protected from balance billing?"
- "What are the requirements for insurance to cover bariatric surgery?"
