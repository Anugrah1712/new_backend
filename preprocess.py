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
    print("[INFO] Starting text preprocessing...")
    paragraphs = []

    for file in files:
        print(f"[DEBUG] Reading file: {file.filename}")
        contents = await file.read()
        file_object = BytesIO(contents)

        if file.filename.endswith(".pdf"):
            print(f"[DEBUG] Detected PDF file: {file.filename}")
            reader = PdfReader(file_object)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    clean_text = ' '.join(text.split())
                    paragraphs.append(clean_text)
                    print(f"[DEBUG] Extracted text from PDF page {i}")

        elif file.filename.endswith(".docx"):
            print(f"[DEBUG] Detected DOCX file: {file.filename}")
            docx = DocxDocument(file_object)
            full_text = "\n\n".join([para.text.strip() for para in docx.paragraphs if para.text.strip()])
            paragraphs.append(full_text)
            print(f"[DEBUG] Extracted text from DOCX file: {file.filename}")

    if scraped_data:
        print("[INFO] Including scraped data...")
        if isinstance(scraped_data, str):
            scraped_chunks = scraped_data.split("\n\n")
            paragraphs.extend(scraped_chunks)
            print(f"[DEBUG] Added {len(scraped_chunks)} chunks from scraped string data")
        elif isinstance(scraped_data, list):
            for item in scraped_data:
                if isinstance(item, dict) and 'full_text' in item:
                    chunks = [chunk.strip() for chunk in item['full_text'].split("\n\n") if chunk.strip()]
                    paragraphs.extend(chunks)
                    print(f"[DEBUG] Added {len(chunks)} chunks from scraped list item")

    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    print(f"[INFO] Total cleaned paragraphs: {len(paragraphs)}")

    docs = [LangchainDocument(page_content=para) for para in paragraphs]
    print(f"[INFO] Converted to Langchain documents: {len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    split_docs = text_splitter.split_documents(docs)
    print(f"[INFO] Split into {len(split_docs)} text chunks with size={size}, overlap={overlap}")

    # [DEBUG] Print preview of split documents
    for i, d in enumerate(split_docs[:5]):
        print(f"[DEBUG] Sample chunk {i+1}: {d.page_content[:300]}...\n")

    return split_docs

# Main entrypoint to support multiple vector DBs
async def preprocess_vectordbs(
    doc_files, embedding_model_name, chunk_size, chunk_overlap, scraped_data,
    selected_vectordb, persist_directory=None
):
    print(f"[INFO] Preprocessing for vector DB: {selected_vectordb}")
    texts = await preprocess_text(doc_files, chunk_size, chunk_overlap, scraped_data)

    print(f"[INFO] Initializing embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    if selected_vectordb == "FAISS":
        print("[INFO] Building FAISS vectorstore...")
        vectorstore = FAISS.from_documents(texts, embedding_model)

        if persist_directory:
            print(f"[INFO] Saving FAISS index to: {persist_directory}")
            vectorstore.save_local(persist_directory)

        # [DEBUG] Validate FAISS docstore contents
        print("[DEBUG] Verifying FAISS docstore contents:")
        empty_count = 0
        for k, v in vectorstore.docstore._dict.items():
            if not v or not v.page_content.strip():
                print(f"[WARN] Empty or None docstore entry for key: {k}")
                empty_count += 1
        if empty_count == 0:
            print("[DEBUG] All docstore entries look valid.")
        else:
            print(f"[WARN] Found {empty_count} empty/invalid docstore entries.")

        # [DEBUG] Print dimension of a sample embedding
        sample_text = texts[0].page_content
        sample_vector = embedding_model.embed_query(sample_text)
        print(f"[DEBUG] Sample embedding dimension: {len(sample_vector)}")

        retriever = vectorstore.as_retriever()
        print("[INFO] FAISS vectorstore ready.")
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

    raise ValueError(f"[ERROR] Unsupported vector DB selected: {selected_vectordb}")
