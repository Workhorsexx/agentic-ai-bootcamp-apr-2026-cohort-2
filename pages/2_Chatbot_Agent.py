"""
Chatbot Agent - AI agent with live web search via Tavily.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import streamlit as st
import html
import re
import os

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
# Wikipedia and ArXiv: Tools for encyclopedia and research papers
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langgraph.prebuilt import create_react_agent

_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")


def _render_assistant_text(text: str) -> None:
    text = text or ""
    parts: list[str] = []
    last = 0
    for m in _MD_LINK.finditer(text):
        parts.append(html.escape(text[last : m.start()]))
        label, raw_url = m.group(1), m.group(2).strip()
        if raw_url.startswith(("http://", "https://")):
            href = html.escape(raw_url, quote=True)
            parts.append(
                f'<a href="{href}" target="_blank" rel="noopener noreferrer">'
                f"{html.escape(label)}</a>"
            )
        else:
            parts.append(html.escape(m.group(0)))
        last = m.end()
    parts.append(html.escape(text[last:]))
    st.markdown(
        f'<div style="white-space: pre-wrap;">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Chatbot Agent",
    page_icon=None,
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background-color: #ffffff; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 860px !important;
    }
    .page-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #111111;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .page-caption {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 1.25rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 0 0 1.25rem;
    }
    .info-banner {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.82rem;
        color: #1d4ed8;
        margin-bottom: 1.25rem;
    }
    .warn-banner {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.82rem;
        color: #92400e;
        margin-bottom: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">Multi-Tool Chatbot Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="page-caption">Multi-Tool Chatbot Agent</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []


# =============================================================================
# CHECK API KEYS FROM HOME PAGE
# =============================================================================

openai_key = st.session_state.get("openai_key", "")
tavily_key = st.session_state.get("tavily_key", "")

missing = []
if not openai_key:
    missing.append("OpenAI")
if not tavily_key:
    missing.append("Tavily")

if missing:
    st.markdown(f"""
    <div class="warn-banner">
        Missing API keys: <strong>{", ".join(missing)}</strong>.
        Please go back to the Home page and save your keys first.
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Home"):
        st.switch_page("Home.py")
    st.stop()
else:
    st.markdown("""
    <div class="info-banner">OpenAI and Tavily keys loaded — ready to search and chat.</div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("**Search-Enabled Chat**")
    st.caption("AI agent with live web search via Tavily.")
    st.divider()
    if st.session_state.openai_key and st.session_state.tavily_key:
        st.subheader("🛠️ Available Tools")
        st.write("✅ **Tavily Search** - Web search")
        st.write("✅ **Wikipedia** - Encyclopedia")
        st.write("✅ **ArXiv** - Research papers")
    
    if st.session_state.openai_key or st.session_state.tavily_key:
        if st.button("Change API Keys"):
            # Reset everything to start fresh
            st.session_state.openai_key = ""
            st.session_state.tavily_key = ""
            st.rerun()


# =============================================================================
# CREATE AGENT
# =============================================================================

if not st.session_state.agent:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create Tavily search tool
    search_tool = TavilySearch(max_results=3)
    


    # Create Wikipedia tool
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500
        ),
        name="wikipedia",
        description="""Search Wikipedia for encyclopedia articles, historical information, 
        biographies, and general knowledge. Best for: 'Who was...', 'What is...', 
        'History of...', 'Explain...' queries."""
    )
    
    # Create ArXiv tool
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500
        ),
        name="arxiv",
        description="""Search ArXiv for academic papers, research articles, and scientific 
        publications. Best for: 'Latest research on...', 'Papers about...', 
        'Scientific studies on...' queries."""
    )
    
    # Create agent with all tools
    tools = [search_tool, wikipedia, arxiv]
    st.session_state.agent = create_react_agent(llm, tools)


# =============================================================================
# GREETING
# =============================================================================

if not st.session_state.agent_messages:
    greeting = "Hi! My name is Assistant. I can search the web for up-to-date information. What would you like to know today?"
    st.session_state.agent_messages.append({"role": "assistant", "content": greeting})

# =============================================================================
# CHAT HISTORY
# =============================================================================

for message in st.session_state.agent_messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_assistant_text(message["content"])
        else:
            st.write(message["content"])
print(st.session_state)


# =============================================================================
# USER INPUT
# =============================================================================

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.agent_messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            response = st.session_state.agent.invoke({
                "messages": st.session_state.agent_messages
            })
            response_text = response["messages"][-1].content
            _render_assistant_text(response_text)
            st.session_state.agent_messages.append({
                "role": "assistant",
                "content": response_text
            })
