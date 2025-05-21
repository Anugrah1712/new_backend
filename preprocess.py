# preprocess.py

from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from fastapi import UploadFile
from dotenv import load_dotenv
from io import BytesIO
import os
import numpy as np
import faiss
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import sys
import sqlite3

# Fix for FAISS SQLite dependency in some environments
sys.modules["sqlite3"] = sqlite3

load_dotenv()

# Preprocess uploaded files + scraped data into text chunks
async def preprocess_text(files: list[UploadFile], size, overlap, scraped_data):
    paragraphs = []

    for file in files:
        contents = await file.read()
        file_object = BytesIO(contents)

        if file.filename.endswith(".pdf"):
            reader = PdfReader(file_object)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    paragraphs.append(' '.join(text.split()))

        elif file.filename.endswith(".docx"):
            docx = DocxDocument(file_object)
            full_text = "\n\n".join([para.text.strip() for para in docx.paragraphs if para.text.strip()])
            paragraphs.append(full_text)

    # Include scraped data if any
    if scraped_data:
        if isinstance(scraped_data, str):
            paragraphs.extend(scraped_data.split("\n\n"))
        elif isinstance(scraped_data, list):
            for item in scraped_data:
                if isinstance(item, dict) and 'full_text' in item:
                    chunks = [chunk.strip() for chunk in item['full_text'].split("\n\n") if chunk.strip()]
                    paragraphs.extend(chunks)

    # Clean and convert to Langchain documents
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    docs = [LangchainDocument(page_content=para) for para in paragraphs]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return text_splitter.split_documents(docs)

# Main entrypoint to support multiple vector DBs
async def preprocess_vectordbs(
    doc_files, embedding_model_name, chunk_size, chunk_overlap, scraped_data,
    selected_vectordb, persist_directory=None
):
    texts = await preprocess_text(doc_files, chunk_size, chunk_overlap, scraped_data)

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    if selected_vectordb == "FAISS":
        # Build FAISS index
        vectorstore = FAISS.from_documents(texts, embedding_model)

        if persist_directory:
            vectorstore.save_local(persist_directory)

        retriever = vectorstore.as_retriever()
        return (
            vectorstore.index,
            vectorstore.docstore,
            vectorstore.index_to_docstore_id,
            vectorstore,
            retriever,
            embedding_model,
            None,  # pinecone_index_name
            None,  # vs
            None   # qdrant_client
        )

    raise ValueError(f"Unsupported vector DB selected: {selected_vectordb}")
