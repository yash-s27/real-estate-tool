from uuid import uuid4
import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()
GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
VECTORSTORE_DIR = Path(__file__).parent/ "resources/vectorstore"
COLLECTION_NAME = "real_estate"
llm = None
vector_store = None

def initialize_components():
    """
    Initialize the LLM and Chroma vector store only once.
    These objects are expensive to create, so we cache them
    for the lifetime of the Streamlit process.
    """
    
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500,
            api_key= GROQ_API_KEY
        )

    if vector_store is None:

        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "trust_remote_code": True
            }
        )

        VECTORSTORE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls): #this function will scrap the data from url and stores in the vector databse
    yield "Initializing Components....."
    initialize_components()
    try:
        vector_store.reset_collection()
    except Exception:
        pass
    yield "Loading the Data..."
    def load_url(urls):
        headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/138.0.0.0 Safari/537.36'
            )
        }

        # If user passes a single URL
        if isinstance(urls, str):
            urls = [urls]

        docs = []

        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()

                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.get_text(" ", strip=True)

                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": url}
                    )
                )

            except Exception as e:
                print(f"Could not fetch {url}: {e}")

        return docs

    data = load_url(urls)
    yield "Splitting the Text...."
    text_splitter = RecursiveCharacterTextSplitter(
        separators =["\n\n","\n","."," "],
        chunk_size = CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Adding documents to VectorDB..."

    uuids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(
        documents=docs,
        ids=uuids
    )

    yield "Done adding docs to vector databases..."

def create_qa_chain():
    initialize_components()

    prompt = ChatPromptTemplate.from_messages([
            ("system",
    """
    You are a helpful AI assistant.

    Answer ONLY using the provided context.

    If the answer cannot be found in the context,
    reply with:

    "I couldn't find the answer in the provided documents."

    Context:
    {context}
    """),
            ("human", "{input}"),
        ])

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return retrieval_chain
def generate_answer(query, retrieval_chain):

    if retrieval_chain is None:
        raise RuntimeError("Please process the URL(s) first.")

    result = retrieval_chain.invoke({
            "input": query
        })

    answer = result["answer"]

    sources = list({
            doc.metadata.get("source", "Unknown")
            for doc in result["context"]
        })

    return answer, sources



if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    process_urls(urls)
    ans , sources = generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
    print(f"{ans}")
    print(f"Sources : {sources}")