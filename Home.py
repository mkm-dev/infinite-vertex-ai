import streamlit as st
import time

st.set_page_config(page_title="Financial Chatbot Experiment", layout="wide")

# Default welcome message from bot
default_message = {
    "role": "assistant",
    "content": "How can I help you?",
    "key": time.time(),
    "type": "text",
}

# To store user queries
st.session_state.setdefault("queries", [])

# To store llm output
st.session_state.setdefault("replies", [])


def bot_process():
    return "Hello there!"


def process_file():
    # Handle file
    pass


def on_input_change():
    user_input = st.session_state.user_input

    st.session_state.queries.append(
        {
            "role": "user",
            "content": user_input,
            "key": time.time(),
            "type": "text"
        }
    )

    response = bot_process()

    st.session_state.replies.append(
        {
            "role": "assistant",
            "content": response,
            "key": time.time(),
            "type": "text"
        })


with st.sidebar:
    # llm_api_key = st.text_input(
    #    "LLM API Key", key="chatbot_api_key", type="password")

    uploaded_file = st.file_uploader("Upload Document")
    if uploaded_file is not None:
        process_file()

st.write("### Financial ChatBot Experiment")

c1 = st.container()

with c1:
    # Show Default Message
    st.chat_message(default_message["role"]).write(
        default_message["content"])

    # Iterate over history
    for idx, msg in enumerate(st.session_state.queries):
        query = st.session_state.queries[idx]
        reply = st.session_state.replies[idx]

        st.chat_message(query["role"]).write(query["content"])
        st.chat_message(reply["role"]).write(reply["content"])

# User Input
prompt = st.chat_input("Ask me something ...",
                       key="user_input", on_submit=on_input_change)
