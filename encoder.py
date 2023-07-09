import time
from typing import List

import langchain
import os
import sys

# Vertex AI
from google.cloud import aiplatform
# from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI

# Ingest PDF files
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Store
from langchain.vectorstores import Chroma


print(f"LangChain version: {langchain.__version__}")
print(f"Vertex AI SDK version: {aiplatform.__version__}")

demo_data = [
    {"title": "Alphabet Q1 2023 10-Q", "path": "./data/20230426-alphabet-10q.pdf"}
]


llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)


# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=0)

file = demo_data[0]

loader = PyPDFLoader(file["path"])
documents = loader.load()

docs = text_splitter.split_documents(documents)
print(f"# of documents = {len(docs)}")
print(docs[0])

# Embedding
embeddings = VertexAIEmbeddings()

# Test embeddings
sample_text = "Hi! Let's get started"
text_embedding = embeddings.embed_query(sample_text)
print(f"Embedding length: {len(text_embedding)}")
print(f"Looks like this: {text_embedding[:5]}...")

# chroma

# db = Chroma.from_documents(docs, embeddings)
