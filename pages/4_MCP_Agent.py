"""
MCP Agent - AI agent with Model Context Protocol server integration.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import streamlit as st
import asyncio
import os
import nest_asyncio

nest_asyncio.apply()

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="MCP Agent",
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

st.markdown('<div class="page-title">MCP Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="page-caption">AI agent connected to external tools via the Model Context Protocol.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "mcp_server_url" not in st.session_state:
    st.session_state.mcp_server_url = ""

if "mcp_api_key" not in st.session_state:
    st.session_state.mcp_api_key = ""

if "mcp_agent" not in st.session_state:
    st.session_state.mcp_agent = None

if "mcp_messages" not in st.session_state:
    st.session_state.mcp_messages = []


# =============================================================================
# CHECK OPENAI KEY FROM HOME PAGE
# =============================================================================

openai_key = st.session_state.get("openai_key", "")

if not openai_key:
    st.markdown("""
    <div class="warn-banner">
        No OpenAI API key found. Please go back to the Home page and save your key first.
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Home"):
        st.switch_page("Home.py")
    st.stop()


# =============================================================================
# MCP SERVER CONFIGURATION
# =============================================================================

# MCP server URL is specific to this page — ask for it here if not set
if not st.session_state.mcp_server_url:
    st.markdown("""
    <div class="info-banner">
        Enter your MCP server URL to connect. The API key is optional and only
        required for servers that use Authorization headers (e.g. Zapier MCP).
    </div>
    """, unsafe_allow_html=True)

    server_url = st.text_input("MCP Server URL", placeholder="https://your-mcp-server.com")
    mcp_api_key = st.text_input("MCP API Key (optional)", type="password", placeholder="leave blank if not required")

    if st.button("Connect to MCP Server"):
        if server_url and (server_url.startswith("http://") or server_url.startswith("https://")):
            st.session_state.mcp_server_url = server_url
            st.session_state.mcp_api_key = mcp_api_key
            st.rerun()
        else:
            st.error("Please enter a valid URL starting with http:// or https://")
    st.stop()
else:
    st.markdown(f"""
    <div class="info-banner">
        OpenAI key loaded · MCP server connected: <strong>{st.session_state.mcp_server_url[:50]}...</strong>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("**MCP Chatbot**")
    st.caption("Connected to an external MCP tool server.")
    st.divider()
    if st.session_state.mcp_server_url:
        st.caption(f"Server: `{st.session_state.mcp_server_url[:30]}...`")
    if st.button("Clear chat"):
        st.session_state.mcp_messages = []
        st.rerun()
    if st.button("Change MCP Server"):
        st.session_state.mcp_server_url = ""
        st.session_state.mcp_api_key = ""
        st.session_state.mcp_agent = None
        st.session_state.mcp_messages = []
        st.rerun()
    if st.button("Home"):
        st.switch_page("Home.py")


# =============================================================================
# INITIALIZE MCP AGENT
# =============================================================================

if not st.session_state.mcp_agent:
    with st.spinner("Initializing MCP agent..."):
        os.environ["OPENAI_API_KEY"] = openai_key

        async def init_agent():
            server_cfg = {
                "url": st.session_state.mcp_server_url,
                "transport": "streamable_http",
            }
            if st.session_state.mcp_api_key:
                server_cfg["headers"] = {
                    "Authorization": f"Bearer {st.session_state.mcp_api_key}"
                }
            client = MultiServerMCPClient({"server": server_cfg})
            tools = await client.get_tools()
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            return create_react_agent(llm, tools)

        try:
            st.session_state.mcp_agent = asyncio.get_event_loop().run_until_complete(init_agent())
        except ExceptionGroup as eg:
            for exc in eg.exceptions:
                st.error(f"MCP sub-error: {type(exc).__name__}: {exc}")
            st.session_state.mcp_server_url = ""
            st.session_state.mcp_agent = None
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialize MCP agent: {type(e).__name__}: {e}")
            st.session_state.mcp_server_url = ""
            st.session_state.mcp_agent = None
            st.stop()


# =============================================================================
# GREETING
# =============================================================================

if not st.session_state.mcp_messages:
    greeting = "Hi! My name is Assistant. I'm connected to an MCP server and ready to use external tools. What can I help you with today?"
    st.session_state.mcp_messages.append({"role": "assistant", "content": greeting})

# =============================================================================
# CHAT HISTORY
# =============================================================================

for message in st.session_state.mcp_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =============================================================================
# USER INPUT
# =============================================================================

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.mcp_messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Processing with MCP tools..."):
            async def run_agent():
                return await st.session_state.mcp_agent.ainvoke({
                    "messages": st.session_state.mcp_messages
                })

            try:
                response = asyncio.get_event_loop().run_until_complete(run_agent())
                response_text = response["messages"][-1].content
                st.write(response_text)
                st.session_state.mcp_messages.append({
                    "role": "assistant",
                    "content": response_text
                })
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.mcp_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })