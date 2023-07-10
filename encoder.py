import time
from typing import List
from pydantic import BaseModel

import langchain
import os

# Vertex AI
from google.cloud import aiplatform
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI

# PDF files
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Store
from langchain.vectorstores import Chroma

# Q&A
from langchain.chains import RetrievalQA


print(f"LangChain version: {langchain.__version__}")
print(f"Vertex AI SDK version: {aiplatform.__version__}")

demo_data = [
    {"title": "Alphabet Q1 2023 10-Q", "path": "./data/20230426-alphabet-10q.pdf"}
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

use_custom_embeddings = True


# Custom Embedding with rate limiting
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Rate limit")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch:],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]


# Embeddings
if use_custom_embeddings:
    embeddings = CustomVertexAIEmbeddings(
        requests_per_minute=60,
        num_instances_per_batch=5,
    )
else:
    embeddings = VertexAIEmbeddings()

"""
# Test embeddings
sample_text = "Hi! Let's get started"
text_embedding = embeddings.embed_query(sample_text)
print(f"Embedding length: {len(text_embedding)}")
print(f"Looks like this: {text_embedding[:5]}...")
"""

# Chroma DB
CHROMADB_PATH = ".chromadb/"

# If db is already there load from files else create from docs
if os.path.exists(CHROMADB_PATH + "chroma-embeddings.parquet"):
    db = Chroma(persist_directory=CHROMADB_PATH,
                embedding_function=embeddings)
    print("Loading db from files")
else:
    print("Creating db from docs: This may take a while ...")

    loader = PyPDFLoader(demo_data[0]["path"])
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=0)

    docs = text_splitter.split_documents(documents)
    # print(f"# of documents = {len(docs)}")
    # print(docs[0])

    db = Chroma.from_documents(
        docs[:30], embeddings, persist_directory=CHROMADB_PATH)
    db.persist()

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Sample Query
query = "What is the address of Alphabet Inc?"
result = qa({"query": query})
print(result["result"])
