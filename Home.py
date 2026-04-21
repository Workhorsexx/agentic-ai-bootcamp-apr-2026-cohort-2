"""
LLM Bootcamp Project - Home Page
"""

import streamlit as st

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LLM Bootcamp Project",
    page_icon=None,
    layout="centered"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .stApp { background-color: #ffffff; }

    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 860px !important;
    }

    .page-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #111111;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
        text-align: center;
    }
    .page-sub {
        font-size: 0.9rem;
        color: #555555;
        max-width: 500px;
        margin: 0 auto 1.5rem;
        line-height: 1.6;
        text-align: center;
    }

    .info-banner {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #1d4ed8;
        margin-bottom: 1.75rem;
        text-align: center;
    }

    .divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 0 0 1.5rem;
    }

    .sec-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 1rem;
    }

    /* 2-column grid */
    .module-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .module-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .module-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        background: #f3f4f6;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #6b7280;
    }

    .module-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111111;
        margin: 0;
    }

    .module-desc {
        font-size: 0.8rem;
        color: #6b7280;
        line-height: 1.55;
        flex: 1;
    }

    .module-footer {
        border-top: 1px solid #f3f4f6;
        padding-top: 0.75rem;
        margin-top: 0.25rem;
    }

    /* API section */
    .api-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 0.75rem;
    }

    /* Override Streamlit button inside cards */
    .stButton > button {
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 0.3rem 0.9rem !important;
        margin: 0 !important;
        width: auto !important;
    }
    .stButton > button:hover {
        background: #dbeafe !important;
        border-color: #93c5fd !important;
    }

    /* Save keys button override */
    .save-btn > button {
        background: #111111 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 1.2rem !important;
        width: 100% !important;
        margin-top: 0.25rem !important;
    }
    .save-btn > button:hover {
        background: #333333 !important;
    }

    .footer-note {
        text-align: center;
        font-size: 0.7rem;
        color: #d1d5db;
        padding-top: 0.75rem;
        border-top: 1px solid #f3f4f6;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""
if "tavily_key" not in st.session_state:
    st.session_state.tavily_key = ""

# ============================================================================
# HERO
# ============================================================================

st.markdown('<div class="page-title">AI Chatbot Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Hands-on modules exploring core LLM application patterns — from basic inference to retrieval-augmented generation and tool use.</div>', unsafe_allow_html=True)

# Info banner
openai_set = bool(st.session_state.openai_key)
tavily_set = bool(st.session_state.tavily_key)
if not openai_set:
    st.markdown('<div class="info-banner">Enter your API keys below to get started.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="info-banner">API keys saved — select a module below to begin.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ============================================================================
# API KEYS
# ============================================================================

st.markdown('<div class="api-label">API Keys</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    openai_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-...",
        value=st.session_state.openai_key,
    )
with col2:
    tavily_input = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-...",
        value=st.session_state.tavily_key,
    )

with st.container():
    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save API Keys"):
        errors = []
        if openai_input and not openai_input.startswith("sk-"):
            errors.append("Invalid OpenAI key format.")
        if tavily_input and not tavily_input.startswith("tvly-"):
            errors.append("Invalid Tavily key format.")
        if errors:
            st.error(" ".join(errors))
        else:
            st.session_state.openai_key = openai_input
            st.session_state.tavily_key = tavily_input
            st.success("Keys saved.")
    st.markdown('</div>', unsafe_allow_html=True)

k1, k2 = st.columns(2)
with k1:
    if st.session_state.openai_key:
        st.success("OpenAI connected")
    else:
        st.warning("OpenAI not set")
with k2:
    if st.session_state.tavily_key:
        st.success("Tavily connected")
    else:
        st.warning("Tavily not set — needed for Search Chat")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ============================================================================
# MODULE GRID
# ============================================================================

st.markdown('<div class="sec-label">Modules</div>', unsafe_allow_html=True)

modules = [
    {
        "num": "01",
        "title": "Basic AI Chat",
        "desc": "Direct conversational interface with a language model. No tools, no retrieval — just the base model.",
        "page": "pages/1_Basic_Chatbot.py"
    },
    {
        "num": "02",
        "title": "Search-Enabled Chat",
        "desc": "Extends base chat with live web search via Tavily, enabling access to current information.",
        "page": "pages/2_Chatbot_Agent.py"
    },
    {
        "num": "03",
        "title": "RAG — Retrieval-Augmented Generation",
        "desc": "Upload documents and query them using semantic search. Grounds model responses in your own sources.",
        "page": "pages/3_Chat_with_your_Data.py"
    },
    {
        "num": "04",
        "title": "MCP Chatbot",
        "desc": "Tool-use via the Model Context Protocol. Connects the model to structured external services and APIs.",
        "page": "pages/4_MCP_Agent.py"
    },
]

# Render in 2-column pairs
for i in range(0, len(modules), 2):
    col1, col2 = st.columns(2)
    for col, m in zip([col1, col2], modules[i:i+2]):
        with col:
            with st.container(border=True):
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;">
                    <span class="module-number">{m['num']}</span>
                    <span class="module-title">{m['title']}</span>
                </div>
                <div class="module-desc" style="margin-bottom:0.85rem;">{m['desc']}</div>
                """, unsafe_allow_html=True)
                if st.button("Open module", key=m["page"]):
                    st.switch_page(m["page"])

# ============================================================================
# FOOTER
# ============================================================================

st.markdown('<div class="footer-note">LLM Bootcamp · Session Project</div>', unsafe_allow_html=True)