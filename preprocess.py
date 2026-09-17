#preprocess.py

from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from fastapi import UploadFile
from dotenv import load_dotenv
from io import BytesIO
import os
import pickle
import hashlib
import numpy as np
from typing import Union
import sys
import sqlite3
import faiss
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


# Fix for FAISS SQLite dependency in some environments
sys.modules["sqlite3"] = sqlite3

load_dotenv()


def _normalize_for_hash(text):
    """Collapses whitespace/case so near-identical chunks (same sidebar,
    slightly different trailing spaces/newlines) hash the same."""
    return " ".join(text.split()).lower()


def _is_link_list_chunk(text, max_line_len=40, min_lines=4, threshold=0.7):
    """Heuristic for nav/sidebar 'Related Links' style blocks: mostly short
    lines with no sentence-ending punctuation. These carry navigation value,
    not article content, and — because the same sidebar repeats verbatim
    across every page in a category (all loan pages, all card pages, etc.)
    — they flood the vector store with near-duplicate, keyword-dense chunks
    that out-rank the one real content chunk on any topical query.

    This is deliberately per-page-category robust: unlike the crawler's
    cross-page frequency stripper (which only catches lines repeated on
    >50% of ALL pages site-wide), this catches link-list *shape* regardless
    of what fraction of pages it appears on.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < min_lines:
        return False
    short_no_punct = sum(
        1 for l in lines
        if len(l) <= max_line_len and not l.rstrip().endswith((".", "!", "?", ":"))
    )
    return (short_no_punct / len(lines)) >= threshold


# Preprocess uploaded files + scraped data into text chunks
async def preprocess_text(files: list[Union[str, 'UploadFile']], size, overlap, scraped_data=None):
    docs = []

    for file in files:
        # Handle FastAPI UploadFile
        if hasattr(file, "filename") and hasattr(file, "read"):
            filename = file.filename
            contents = await file.read()
            file_object = BytesIO(contents)

        # Handle file path as str
        elif isinstance(file, str):
            filename = os.path.basename(file)
            with open(file, "rb") as f:
                contents = f.read()
            file_object = BytesIO(contents)

        else:
            continue  # Invalid file format

        # PDF handling
        if filename.endswith(".pdf"):
            reader = PdfReader(file_object)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = ' '.join(page_text.split())
                    docs.append(LangchainDocument(
                        page_content=cleaned_text,
                        metadata={"source_type": "document", "source_name": filename, "page": page_num}
                    ))

        # DOCX handling
        elif filename.endswith(".docx"):
            docx = DocxDocument(file_object)
            full_text = ""
            for para in docx.paragraphs:
                if para.text.strip():
                    full_text += para.text.strip() + "\n\n"
            if full_text.strip():
                docs.append(LangchainDocument(
                    page_content=full_text,
                    metadata={"source_type": "document", "source_name": filename}
                ))

    print(f"📄 Total document-based chunks before scraped data: {len(docs)}")
    print("🧩 First 5 extracted docs:")
    for i, d in enumerate(docs[:5]):
        print(f"{i+1}. {d.page_content[:100]}...")

    if scraped_data:
        seen_hashes = set()
        dropped_dupes = 0
        dropped_linklists = 0

        if isinstance(scraped_data, str):
            # Legacy path: plain text blob with no URL attached to any of it.
            for chunk in scraped_data.split("\n\n"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if _is_link_list_chunk(chunk):
                    dropped_linklists += 1
                    continue
                chunk_hash = hashlib.sha256(_normalize_for_hash(chunk).encode("utf-8")).hexdigest()
                if chunk_hash in seen_hashes:
                    dropped_dupes += 1
                    continue
                seen_hashes.add(chunk_hash)
                docs.append(LangchainDocument(
                    page_content=chunk,
                    metadata={"source_type": "web", "source_url": None}
                ))
        elif isinstance(scraped_data, list):
            for item in scraped_data:
                if isinstance(item, dict) and 'full_text' in item:
                    url = item.get('url')
                    chunks = [chunk.strip() for chunk in item['full_text'].split("\n\n") if chunk.strip()]
                    for chunk in chunks:
                        # Drop nav/sidebar "Related Links" style blocks — these
                        # repeat near-verbatim across every page in a category
                        # (loan pages, card pages, etc.) and would otherwise
                        # flood the index with keyword-dense duplicates that
                        # out-rank the actual unique content for a topical query.
                        if _is_link_list_chunk(chunk):
                            dropped_linklists += 1
                            continue

                        # Exact/near-duplicate content (the same sidebar or
                        # boilerplate paragraph appearing on many pages) —
                        # keep only the first occurrence.
                        chunk_hash = hashlib.sha256(_normalize_for_hash(chunk).encode("utf-8")).hexdigest()
                        if chunk_hash in seen_hashes:
                            dropped_dupes += 1
                            continue
                        seen_hashes.add(chunk_hash)

                        docs.append(LangchainDocument(
                            page_content=chunk,
                            metadata={"source_type": "web", "source_url": url}
                        ))

        print(f"🧹 [DEDUP] ➤ Dropped {dropped_linklists} link-list-shaped chunks "
              f"and {dropped_dupes} exact/near-duplicate chunks from scraped data.")

    docs = [d for d in docs if d.page_content.strip()]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    # split_documents() carries each Document's metadata over to its split chunks
    # automatically, so source_type/source_url survive chunking.
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks

# Main entrypoint to support multiple vector DBs
async def preprocess_vectordbs(
    doc_files, embedding_model_name, chunk_size, chunk_overlap, scraped_data,
    selected_vectordb, persist_directory=None
):
    print(f"[INFO] Preprocessing for vector DB: {selected_vectordb}")

    # ✅ Check: if both PDF and scraped data are missing, raise error
    if not doc_files and not scraped_data:
        raise ValueError("No documents or scraped content to process.")

    texts = await preprocess_text(doc_files, chunk_size, chunk_overlap, scraped_data)
    print(f"[DEBUG] Number of documents/chunks: {len(texts)}")
    for i, doc in enumerate(texts[:10]):  # Check the first 10 chunks
        print(f"[DEBUG] Chunk {i+1} content preview: {repr(doc.page_content[:100])}")
        if not doc.page_content.strip():
            print(f"[WARNING] Chunk {i+1} is empty or whitespace!")

    print(f"[INFO] Initializing embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    embedding_vector = embedding_model.embed_query("test")
    # print("Embedding dimension:", len(embedding_vector))

    if selected_vectordb == "FAISS":
        print("[INFO] Building FAISS vectorstore...")
        print(f"[INFO] Total document chunks: {len(texts)}")
        for i, doc in enumerate(texts):
            if not doc.page_content.strip():
                print(f"[⚠️] Empty content at chunk {i}")
            else:
                print(f"[✅] Chunk {i} preview: {doc.page_content[:80]}...")

        # ✅ Rebuild from scratch
        vectorstore = FAISS.from_documents(texts, embedding_model)

        if persist_directory:
            print(f"[INFO] Saving FAISS index manually to: {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)

            # Save FAISS index
            faiss.write_index(vectorstore.index, os.path.join(persist_directory, "index.faiss"))

            # Save docstore and mapping
            with open(os.path.join(persist_directory, "index.pkl"), "wb") as f:
                pickle.dump({
                    "docstore": vectorstore.docstore,
                    "index_to_docstore_id": vectorstore.index_to_docstore_id
                }, f)

            # ✅ DEBUG: Confirm docstore validity
            print("✅ Number of documents in docstore:", len(vectorstore.docstore._dict))
            print("✅ Sample docstore entries:")
            for i, (k, v) in enumerate(vectorstore.docstore._dict.items()):
                print(f"  {i+1}. Key: {k} → Content: {v.page_content[:100] if v else 'None'}")
                if i >= 4: break  # print first 5 only

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

    # ❌ Unsupported DB case
    raise ValueError(f"[ERROR] Unsupported vector DB selected: {selected_vectordb}")