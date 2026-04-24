"""
Chat with your Data - RAG with PDF documents.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import streamlit as st
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangGraph: For building agentic workflows
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Literal


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Chat with your Data",
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

st.markdown('<div class="page-title">RAG — Chat with your Data</div>', unsafe_allow_html=True)
st.markdown('<div class="page-caption">Upload PDF documents and ask questions about their content.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_llm" not in st.session_state:
    st.session_state.rag_llm = None

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None  # Store the agentic RAG workflow


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
    <div class="info-banner">OpenAI key loaded — upload a PDF below to get started.</div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📚 Chat with your Data (Agentic RAG)")
    st.caption("AI agent that intelligently searches and answers from your documents")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.rag_messages = []
        st.rerun()
    if st.button("Clear documents"):
        st.session_state.vector_store = None
        st.session_state.rag_llm = None
        st.session_state.rag_messages = []
        st.session_state.processed_files = []
        st.session_state.rag_agent = None
        st.rerun()
    if st.button("Home"):
        st.switch_page("Home.py")


# =============================================================================
# PDF UPLOAD AND PROCESSING
# =============================================================================

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    current_files = [f.name for f in uploaded_files]

    if st.session_state.processed_files != current_files:
        with st.spinner("Processing documents..."):
            documents = []
            os.makedirs("tmp", exist_ok=True)

            for file in uploaded_files:
                file_path = os.path.join("tmp", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(api_key=openai_key)
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

            st.session_state.rag_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=openai_key
            )

            st.session_state.rag_messages = []
            st.session_state.processed_files = current_files

        st.success(f"Processed {len(uploaded_files)} document(s). You can now ask questions below.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =============================================================================
# CREATE AGENTIC RAG WORKFLOW
# =============================================================================

if st.session_state.vector_store and not st.session_state.rag_agent:
    
    # Define state structure for the workflow
    class AgentState(TypedDict):
        question: str  # User's question
        documents: list  # Retrieved documents
        generation: str  # Generated answer
        steps: list  # Track what the agent does
        rewrite_count: int  # Guard against infinite rewrite loops

    # Node 1: Retrieve documents
    def retrieve_documents(state: AgentState):
        """Search documents for relevant information"""
        question = state["question"]
        retriever = st.session_state.vector_store.as_retriever()
        docs = retriever.invoke(question)
        
        return {
            "documents": docs,
            "steps": state.get("steps", []) + ["📚 Retrieved documents"]
        }

    # Node 2: Grade document relevance
    def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
            """Check if retrieved documents are actually relevant"""
            question = state["question"]
            docs = state["documents"]

            if not docs:
                return "generate"

            if state.get("rewrite_count", 0) >= 3:
                return "generate"

            # Simple relevance check using LLM
            prompt = f"""Are these documents relevant to the question: "{question}"?

    Documents: {docs[0].page_content[:500]}

    Answer with just 'yes' or 'no'."""

            response = st.session_state.rag_llm.invoke(prompt)
            is_relevant = "yes" in response.content.lower()

            return "generate" if is_relevant else "rewrite"

    # Node 3: Rewrite question
    def rewrite_question(state: AgentState):
        """Rewrite question for better search results"""
        question = state["question"]

        rewrite_prompt = f"Rewrite this question to be more specific and searchable: {question}"
        new_question = st.session_state.rag_llm.invoke(rewrite_prompt).content

        return {
            "question": new_question,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "steps": state["steps"] + [f"🔄 Rewrote question: {new_question}"]
        }

    # Node 4: Generate answer
    def generate_answer(state: AgentState):
        """Generate final answer from documents"""
        question = state["question"]
        docs = state["documents"]
        
        if not docs:
            return {
                "generation": "I couldn't find relevant information in the documents.",
                "steps": state["steps"] + ["❌ No relevant documents found"]
            }
        
        # Combine documents into context
        context = "\n\n---\n\n".join(doc.page_content for doc in docs[:5])
        
        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using ONLY the provided context. Be concise and accurate. \
            If you cannot generate an answer based on the context, say you don't know."),
            ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
        ])
        
        response = st.session_state.rag_llm.invoke(
            prompt.format_messages(question=question, context=context)
        )
        
        return {
            "generation": response.content,
            "steps": state["steps"] + ["💬 Generated answer"]
        }

    # Build the workflow graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("generate", generate_answer)

    # Define the flow
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("rewrite", "retrieve")  # After rewrite, retrieve again
    workflow.add_edge("generate", END)

    # Compile and save
    st.session_state.rag_agent = workflow.compile()

# =============================================================================
# CHAT INTERFACE
# =============================================================================

if st.session_state.vector_store:

    if not st.session_state.rag_messages:
        greeting = "Hi! My name is Assistant. I've read your documents and I'm ready to answer questions about them. What would you like to know?"
        st.session_state.rag_messages.append({"role": "assistant", "content": greeting})

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        st.session_state.rag_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        # Generate response using agentic workflow
        with st.chat_message("assistant"):
            with st.spinner("Agent is working..."):
                
                # Run the agentic workflow
                result = st.session_state.rag_agent.invoke({
                    "question": user_input,
                    "documents": [],
                    "generation": "",
                    "steps": [],
                    "rewrite_count": 0
                })
                
                # Show agent's reasoning process
                with st.expander("🤖 View Agent Process", expanded=False):
                    st.markdown("### What the agent did:")
                    for step in result["steps"]:
                        st.markdown(f"- {step}")
                
                # Display final answer
                st.write(result["generation"])
                
                # Save to history
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": result["generation"]
                })

else:
    st.info("Upload one or more PDF documents above to start chatting.")
