"""
Build PromptGenerator with LangGraph
"""

# =============================================================================
# IMPORTS
# =============================================================================

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

# Pydantic: For defining data structures
from pydantic import BaseModel
from typing import List, Literal, TypedDict
from langchain_core.messages import BaseMessage


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Prompt Generator",
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

st.markdown('<div class="page-title">AI Prompt Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="page-caption">A friendly AI assistant that can help you generate prompts.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# Define LangGraph state
class State(TypedDict):
    messages: List[BaseMessage]


# =============================================================================
# SESSION STATE
# =============================================================================

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prompt_generator" not in st.session_state:
    st.session_state.prompt_generator = None


# =============================================================================
# CHECK API KEY FROM HOME PAGE
# =============================================================================

openai_key = st.session_state.get("openai_key", "")

if not openai_key:
    st.markdown("""
    <div class="warn-banner">
        No API key found. Please go back to the Home page and save your OpenAI API key first.
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Home"):
        st.switch_page("Home.py")
    st.stop()
else:
    st.markdown("""
    <div class="info-banner">OpenAI key loaded from Home — ready to chat.</div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("**Basic AI Prompt Generator**")
    st.caption("Simple LLM conversation to generate prompt based on your goal, requirements, contraints and structure")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.prompt_generator = None
        st.session_state.llm = None
        st.rerun()
    if st.button("Home"):
        st.switch_page("Home.py")


# =============================================================================
# INITIALIZE AI
# =============================================================================

if not st.session_state.llm:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=openai_key
    )


# =============================================================================
# BUILD 2-STATE PROMPT GENERATOR GRAPH
# =============================================================================

if st.session_state.llm and not st.session_state.prompt_generator:

    # Define requirements structure
    class PromptInstructions(BaseModel):
        """Structure for collecting prompt requirements"""
        objective: str
        variables: List[str]
        constraints: List[str]
        requirements: List[str]
    
    # Bind tool to LLM
    llm_with_tool = st.session_state.llm.bind_tools([PromptInstructions])

    # STATE 1: Gather requirements through conversation
    def gather_requirements(state: State):
        """Ask questions to understand what prompt the user needs"""
        system_prompt = """Help the user create a custom AI prompt through friendly conversation.

        You need to understand:
        1. Purpose: What do they want the AI to help with?
        2. Information needed: What details will they provide each time?
        3. Things to avoid: What should the AI NOT do?
        4. Must include: What should the AI always do?

        RULES:
        - Ask ONE question at a time in plain language
        - No technical terms like variables or parameters
        - Be conversational and friendly

        When you have all information, call the tool."""
        
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}

    # Transition node between states
    def add_tool_message(state: State):
        """Add confirmation message after requirements are collected"""
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content="Requirements collected! Generating prompt...",
                    tool_call_id=tool_call_id
                )
            ]
        }

    # STATE 2: Generate the actual prompt
    def generate_prompt(state: State):
        """Create a professional prompt based on collected requirements"""
        tool_args = None
        post_tool_messages = []
        
        # Extract requirements from tool call
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_args = msg.tool_calls[0]["args"]
            elif isinstance(msg, ToolMessage):
                continue
            elif tool_args:
                post_tool_messages.append(msg)
        
        if tool_args:
            requirements_text = f"""
            Objective: {tool_args.get('objective', 'Not specified')}
            Variables: {', '.join(tool_args.get('variables', []))}
            Constraints: {', '.join(tool_args.get('constraints', []))}
            Requirements: {', '.join(tool_args.get('requirements', []))}
            """
            
            system_msg = SystemMessage(content=f"""Create a prompt template based on:

            {requirements_text}

            Guidelines:
            - Make it clear and specific
            - Use {{{{variable_name}}}} format for variables
            - Address all constraints and requirements
            - Use professional prompt engineering techniques""")
            
            messages = [system_msg] + post_tool_messages
        else:
            messages = post_tool_messages
        
        response = st.session_state.llm.invoke(messages)
        return {"messages": [response]}

    # Router to decide which state to go to
    def route_conversation(state: State) -> Literal["add_tool_message", "gather", "__end__"]:
        """Decide what to do next based on current state"""
        last_msg = state["messages"][-1]
        
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "add_tool_message"  # Requirements collected, transition to generate
        elif not isinstance(last_msg, HumanMessage):
            return "__end__"  # Done
        else:
            return "gather"  # Keep gathering requirements
    
    # Build workflow
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("gather", gather_requirements)
    workflow.add_node("add_tool_message", add_tool_message)
    workflow.add_node("generate", generate_prompt)
    
    # Add edges
    workflow.add_edge(START, "gather")
    workflow.add_conditional_edges(
        "gather",
        route_conversation,
        {
            "add_tool_message": "add_tool_message",
            "gather": "gather",
            "__end__": END
        }
    )
    workflow.add_edge("add_tool_message", "generate")
    workflow.add_edge("generate", END)
    
    # Compile and save
    st.session_state.prompt_generator = workflow.compile()


# =============================================================================
# GREETING
# =============================================================================

if not st.session_state.messages:
    greeting = "Hi! My name is Assistant. What can I do for you today?"
    st.session_state.messages.append({"role": "assistant", "content": greeting, "is_greeting": True})

# =============================================================================
# CHAT HISTORY
# =============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =============================================================================
# USER INPUT
# =============================================================================

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant" and not msg.get("is_greeting"):
                    messages.append(AIMessage(content=msg["content"]))            

            result = st.session_state.prompt_generator.invoke({"messages": messages})
            response = result["messages"][-1].content

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
