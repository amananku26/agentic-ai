import streamlit as st
import os
import time
from typing import TypedDict, List
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper 
from langgraph.graph import StateGraph, END

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IFC Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stChatMessage { background-color: transparent; }
    .stStatus { border-radius: 10px; border: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIG & TOOLS ---
VECTOR_STORE_PATH = "faiss_index"
PROJECT_ID = "gd-gcp-gridu-genai"
LOCATION = "us-central1"
MAX_ITERATIONS = 3

@tool
def document_search(query: str):
    """Searches the IFC Annual Report for financial details."""
    try:
        if not os.path.exists(VECTOR_STORE_PATH): return "Error: Index missing."
        embedding_model = VertexAIEmbeddings(model_name="text-embedding-004", project=PROJECT_ID, location=LOCATION)
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs]) if docs else "No info found."
    except Exception as e: return f"Error: {e}"

@tool
def web_search(query: str):
    """Searches the web for general knowledge."""
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=3) 
        return wrapper.run(query) or "No web results found."
    except Exception as e: return f"Error: {e}"

# --- 3. AGENT STATE ---
class AgentState(TypedDict):
    user_query: str
    plan: str
    tool_input: str  # Stores the specific search term
    evidence: List[str]
    draft_answer: str
    critique_feedback: str
    iteration_count: int

# --- 4. BUILD GRAPH (Cached) ---
@st.cache_resource
def build_agent_graph():
    llm = VertexAI(model_name="gemini-2.0-flash-001", project=PROJECT_ID, location=LOCATION, temperature=0)

    def planner_node(state: AgentState):
        # Decide Tool AND Rewrite Query
        if state.get("critique_feedback"):
            prompt = f"""
            Context: User Query: "{state['user_query']}"
            Failed Attempt Feedback: "{state['critique_feedback']}"
            
            Task: Select the best tool to fix the error and write a search query.
            Available Tools: document_search, web_search.
            
            CRITICAL INSTRUCTION: Output ONLY the Tool Name and Query. Do not explain.
            Format: TOOL_NAME: SEARCH_QUERY
            """
        else:
            prompt = f"""
            Context: User Query: "{state['user_query']}"
            
            Task: Select the best tool for the first missing part and write a query.
            Available Tools: document_search, web_search.
            
            CRITICAL INSTRUCTION: Output ONLY the Tool Name and Query. Do not explain.
            Format: TOOL_NAME: SEARCH_QUERY
            """
            
        response = llm.invoke(prompt).strip()
        
        # --- ROBUST PARSING LOGIC ---
        # This handles cases where the AI adds extra text like "Plan: ..."
        if "document_search" in response:
            tool_name = "document_search"
            # Split by the tool name and take everything after it
            parts = response.split("document_search", 1)
            tool_query = parts[1].lstrip(": ").strip()
        elif "web_search" in response:
            tool_name = "web_search"
            parts = response.split("web_search", 1)
            tool_query = parts[1].lstrip(": ").strip()
        else:
            # Fallback default
            tool_name = "web_search"
            tool_query = state['user_query']
            
        return {"plan": tool_name, "tool_input": tool_query}

    def executor_node(state: AgentState):
        tool_name = state['plan']
        tool_query = state['tool_input'] 
        
        if "document" in tool_name:
            result = f"[Source: PDF] {document_search.invoke(tool_query)}"
        else:
            result = f"[Source: Web] {web_search.invoke(tool_query)}"
            
        return {"evidence": state.get('evidence', []) + [result]}

    def synthesizer_node(state: AgentState):
        evidence_text = "\n".join(state['evidence'])
        prompt = f"""
        User Query: {state['user_query']}
        
        Collected Evidence: 
        {evidence_text}
        
        Task: Draft a complete answer. 
        If you have info for one part but miss the other, write what you have so the Critic can see the gap.
        """
        draft = llm.invoke(prompt)
        return {"draft_answer": draft}

    def critique_node(state: AgentState):
        draft = state['draft_answer']
        prompt = f"""
        User Query: "{state['user_query']}"
        Current Draft Answer: "{draft}"
        
        Step 1: Does the draft answer ALL parts of the user query?
        Step 2: Are there any "missing info" errors?
        
        If COMPLETE: return "SATISFIED".
        If INCOMPLETE: Return a short instruction on what to search for next.
        """
        feedback = llm.invoke(prompt).strip()
        return {"critique_feedback": feedback, "iteration_count": state['iteration_count'] + 1}

    def should_continue(state: AgentState):
        if "SATISFIED" in state['critique_feedback'] or state['iteration_count'] >= MAX_ITERATIONS:
            return "end"
        return "continue"

    # --- Build Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("critique", critique_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", "critique")
    workflow.add_conditional_edges("critique", should_continue, {"continue": "planner", "end": END})
    
    return workflow.compile()

# Initialize Graph
try:
    agent_app = build_agent_graph()
except Exception as e:
    st.error(f"Error building graph: {e}")
    st.stop()

# --- 5. SIDEBAR UI ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/IFC_Logo.svg/1200px-IFC_Logo.svg.png", width=200)
    st.title("Autonomous Agent")
    st.info("This agent uses a Planner-Executor-Critique loop to refine answers autonomously.")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 6. MAIN CHAT INTERFACE ---
st.title("🧠 IFC Autonomous Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a complex question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # THE "THINKING" CONTAINER
        with st.status("🤖 Agent is thinking...", expanded=True) as status:
            inputs = {
                "user_query": prompt, 
                "plan": "",
                "tool_input": "",
                "evidence": [], 
                "draft_answer": "",
                "iteration_count": 0, 
                "critique_feedback": ""
            }
            final_answer = ""
            
            # Stream events from the graph to update the UI live
            try:
                for event in agent_app.stream(inputs):
                    for key, value in event.items():
                        if key == "planner":
                            st.write(f"🧠 **Plan:** Searching `{value['tool_input']}` via `{value['plan']}`...")
                        elif key == "executor":
                            st.write("🕵️ **Executing Search...** Found data.")
                        elif key == "critique":
                            feedback = value['critique_feedback']
                            if "SATISFIED" in feedback:
                                st.write("✅ **Critique:** Answer looks good!")
                            else:
                                st.warning(f"⚖️ **Critique:** {feedback} (Retrying...)")
                        elif key == "synthesizer":
                            final_answer = value.get("draft_answer", "")
                
                status.update(label="✅ Response Generated", state="complete", expanded=False)
                
                # Display Final Answer
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                st.error(f"Agent Error: {e}")
                status.update(label="❌ Error", state="error", expanded=True)