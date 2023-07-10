import streamlit as st
import time

import langchain
import os

# Vertex AI
from google.cloud import aiplatform
# from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI

# Vector Store
from langchain.vectorstores import Chroma

# Q&A
from langchain.chains import RetrievalQA

print(f"LangChain version: {langchain.__version__}")
print(f"Vertex AI SDK version: {aiplatform.__version__}")

st.set_page_config(
    page_title="Infinite AI: Financial Chatbot Experiment", layout="wide")

demo_data = [
    {"title": "Apple 10-K", "link": "#"}
]
# Change it to your project id
aiplatform.init(project="vertex-ai-try")

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

# Embedding
embeddings = VertexAIEmbeddings()

# Chroma DB
# If db is already there load from files else error no db files present
if os.path.exists(".chromadb/chroma-embeddings.parquet"):
    db = Chroma(persist_directory=".chromadb/",
                embedding_function=embeddings)
    print("Loading db from files")
else:
    print("No DB Files Present, create embeddings first")

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

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

st.session_state.setdefault("usedb", True)

st.session_state.setdefault("query_type", "Chat")


def ask_bot(query):
    clean_query = query

    if st.session_state["query_type"] == "Chat":
        response = llm(clean_query)
    else:
        output = qa({"query": clean_query})
        response = output["result"]
    return response


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

    response = ask_bot(user_input)

    st.session_state.replies.append(
        {
            "role": "assistant",
            "content": response,
            "key": time.time(),
            "type": "text"
        })


with st.sidebar:
    st.header("Infinite AI")
    st.subheader("Financial Chatbot Experiment")

    st.info(
        """In this demo you can experience the Generative AI features of Vertex AI.""")

    with st.container():
        """
        We are using Vertex AI to perform some Q&A on Apple 10-K PDF using the text bison LLM.
        """

        # Dropdown to select task
        st.selectbox(
            ' ',
            ('Chat', 'Q&A'),
            key="query_type"
        )

# st.write("### Infinite AI: Financial Chatbot Experiment")

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
