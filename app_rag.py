from chatbot_backend_db_tools_rag import (
    MODEL_NAME,
    MODEL_PROVIDER,
    chatbot,
    create_chatbot,
    get_all_threads,
)
from chunking import create_retriever
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid 
import json
import os
import tempfile
from urllib.parse import urlparse


def urls_in(value):
    urls = set()
    if isinstance(value, dict):
        for item in value.values():
            urls.update(urls_in(item))
    elif isinstance(value, list):
        for item in value:
            urls.update(urls_in(item))
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        if urlparse(value).netloc:
            urls.add(value)
    return urls


def query_metadata(events, pdf_name=None):
    metadata = {"tools": [], "sources": [], "input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
                "model": MODEL_NAME, "provider": MODEL_PROVIDER,
                "pdf_name": pdf_name, "rag": []}
    for message in events:
        if isinstance(message, ToolMessage):
            if message.name and message.name not in metadata["tools"]:
                metadata["tools"].append(message.name)
            content = message.content
            try:
                content = json.loads(content) if isinstance(content, str) else content
            except (TypeError, json.JSONDecodeError):
                pass
            metadata["sources"].extend(urls_in(content))
            if isinstance(content, dict) and content.get("type") == "rag_retrieval":
                metadata["rag"].append({
                    "query": content.get("query", ""),
                    "document_name": content.get("document_name", pdf_name),
                    "retrieval": content.get("retrieval", {}),
                    "documents": [
                        {
                            "rank": document.get("rank"),
                            "page": document.get("page"),
                            "chunk_characters": document.get("chunk_characters", 0),
                        }
                        for document in content.get("retrieved_documents", [])
                    ],
                })
        if isinstance(message, AIMessage):
            usage = message.usage_metadata or {}
            response_metadata = message.response_metadata or {}
            fallback = response_metadata.get("token_usage", {})
            detected_model = (
                response_metadata.get("model_name")
                or response_metadata.get("model")
            )
            if detected_model:
                metadata["model"] = detected_model
            metadata["input_tokens"] = max(metadata["input_tokens"], usage.get("input_tokens", fallback.get("prompt_tokens", 0)) or 0)
            metadata["output_tokens"] = max(metadata["output_tokens"], usage.get("output_tokens", fallback.get("completion_tokens", 0)) or 0)
            metadata["total_tokens"] = max(metadata["total_tokens"], usage.get("total_tokens", fallback.get("total_tokens", 0)) or 0)
    metadata["sources"] = sorted(metadata["sources"])
    if not metadata["total_tokens"]:
        metadata["total_tokens"] = metadata["input_tokens"] + metadata["output_tokens"]
    return metadata


def show_metadata(metadata):
    tools = ", ".join(metadata.get("tools", [])) or "None"
    tokens = f"{metadata.get('total_tokens', 0):,} total ({metadata.get('input_tokens', 0):,} input, {metadata.get('output_tokens', 0):,} output)"
    with st.expander("Query details", expanded=False):
        st.caption(f"Model: {metadata.get('model', MODEL_NAME)} ({metadata.get('provider', MODEL_PROVIDER)})")
        st.caption(f"Tools: {tools}")
        st.caption(f"Tokens: {tokens}")
        if metadata.get("pdf_name"):
            st.caption(f"PDF: {metadata['pdf_name']}")
        for retrieval in metadata.get("rag", []):
            retrieval_config = retrieval.get("retrieval", {})
            documents = retrieval.get("documents", [])
            st.caption(
                f"RAG retrieval: {retrieval_config.get('strategy', 'similarity')} "
                f"({len(documents)} chunks returned)"
            )
            for document in documents:
                st.caption(
                    f"Chunk {document.get('rank', '?')} | "
                    f"Page {document.get('page', '?')} | "
                    f"{document.get('chunk_characters', 0):,} characters"
                )
        for source in metadata.get("sources", []):
            st.markdown(f"- [{source}]({source})")

# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())



# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

    

# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])



# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = st.session_state["chatbot"].get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])



# Display the main application title
st.title("Agentic Chatbot with LangGraph")


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = chatbot

if "pdf_signature" not in st.session_state:
    st.session_state["pdf_signature"] = None


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()



# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


# ========================= Sidebar threading feature =========================

# Display the sidebar title
st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()




# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    # Create one sidebar button for every conversation
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []


        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue


            # Convert the LangChain message into a dictionary
            temp_messages.append({
                "role": role,
                "content": message.content,
                "metadata": {}
            })


        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()



# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])
        if message["role"] == "assistant" and message.get("metadata"):
            show_metadata(message["metadata"])



# Create the chat input box. PDFs are attached to the individual message.
submission = st.chat_input(
    "Ask a question or attach a PDF",
    accept_file=True,
    file_type=["pdf"],
)


# Run this block after the user submits a message
if submission:
    user_input = (getattr(submission, "text", "") or "").strip()
    uploaded_files = getattr(submission, "files", []) or []

    if uploaded_files:
        uploaded_pdf = uploaded_files[0]
        signature = (uploaded_pdf.name, uploaded_pdf.size)
        if signature != st.session_state["pdf_signature"]:
            try:
                with st.spinner(f"Indexing {uploaded_pdf.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf:
                        pdf.write(uploaded_pdf.getvalue())
                        pdf_path = pdf.name
                    retriever = create_retriever(pdf_path)
                    st.session_state["chatbot"] = create_chatbot(
                        retriever, document_name=uploaded_pdf.name
                    )
                    st.session_state["pdf_signature"] = signature
                    st.session_state["pdf_name"] = uploaded_pdf.name
            except Exception as error:
                st.error(f"Could not index PDF: {error}")
                st.stop()

    if not user_input:
        user_input = f"I uploaded `{uploaded_files[0].name}`. What can you tell me about it?" if uploaded_files else ""

    if not user_input:
        st.stop()

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })


    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)
        for uploaded_file in uploaded_files:
            st.caption(f"Attached: {uploaded_file.name}")


    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }



    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}
        events = []

        def ai_only_stream():
            for message_chunk, metadata in st.session_state["chatbot"].stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                events.append(message_chunk)
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, str) and content:
                        yield content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("text"):
                                yield block["text"]

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as error:
            ai_message = f"I couldn't complete that request: {error}"
            st.error(ai_message)

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )


    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message,
        "metadata": query_metadata(
            events, pdf_name=st.session_state.get("pdf_name")
        )
    })
    show_metadata(st.session_state["message_history"][-1]["metadata"])
