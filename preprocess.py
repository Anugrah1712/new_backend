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
import re
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


# NEW: uploaded PDFs/DOCX often contain URLs inline in their text (e.g. a
# "senior citizen FD rates" page copy-pasted into a doc, with a link to the
# full rate sheet at the bottom). We want to cite those the same way the
# chatbot already cites scraped web pages — but the URL has to be extracted
# here, at preprocessing time, since by inference time we only have raw
# chunk text and no clean way to tell "this looks like a URL" from
# "this looks like an answer".
_URL_REGEX = re.compile(r'https?://[^\s<>\[\]{}"\'()]+')


def _extract_urls(text):
    """Finds URLs embedded in document text and returns a deduped list, in
    order of first appearance. Strips trailing punctuation (periods, commas,
    colons) that's often glued onto a URL by PDF text extraction or normal
    sentence punctuation, not part of the URL itself."""
    if not text:
        return []
    cleaned = []
    for url in _URL_REGEX.findall(text):
        url = url.rstrip('.,;:!?')
        if url and url not in cleaned:
            cleaned.append(url)
    return cleaned


def _strip_boilerplate_urls(per_page_urls, min_pages=3, frequency_threshold=0.5):
    """Given one URL list per page of a SINGLE document, drops any URL that
    appears on a large fraction of that document's pages.

    Multi-page PDFs (and long DOCX exports) very often carry the same
    header/footer links — "download our app", a gift-card promo, terms &
    conditions — printed verbatim on every page. _extract_urls() has no way
    to tell those apart from a genuine in-content reference just by looking
    at one page in isolation, so without this filter every chunk of the
    document inherits the SAME footer link(s) as its `source_urls`
    metadata — and the chatbot ends up citing that one irrelevant footer
    page after every single answer, no matter what the answer is actually
    about. This mirrors webscrape.py's _strip_repeated_boilerplate_lines,
    which solves the identical problem for scraped web pages: a URL/line
    that repeats across most pages is boilerplate, not content, regardless
    of what produced it.
    """
    total_pages = len(per_page_urls)
    if total_pages < min_pages:
        return per_page_urls  # not enough pages to tell boilerplate from content

    url_page_counts = {}
    for urls in per_page_urls:
        for url in set(urls):
            url_page_counts[url] = url_page_counts.get(url, 0) + 1

    threshold_count = max(min_pages, int(total_pages * frequency_threshold))
    boilerplate = {url for url, count in url_page_counts.items() if count >= threshold_count}

    if not boilerplate:
        return per_page_urls

    print(f"[URL DEDUP] ➤ Dropped {len(boilerplate)} boilerplate URL(s) repeated on "
          f"≥{int(frequency_threshold * 100)}% of {total_pages} page(s): {sorted(boilerplate)}")

    return [[u for u in urls if u not in boilerplate] for urls in per_page_urls]


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


def _dedupe_urls(urls):
    """Deduplicate URLs while preserving their original order."""
    out = []
    seen = set()
    for url in urls or []:
        url = (url or "").strip().rstrip('.,;:!?')
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _extract_pdf_annotation_urls(page):
    """Extract clickable hyperlink targets from a PDF page when PyPDF2 exposes them.

    This complements URLs visible in extracted text. It is deliberately best-effort:
    malformed annotations are ignored rather than failing preprocessing.
    """
    urls = []
    try:
        annotations = page.get("/Annots") or []
        for annotation_ref in annotations:
            try:
                annotation = annotation_ref.get_object()
                action = annotation.get("/A")
                if action:
                    uri = action.get("/URI")
                    if uri:
                        urls.append(str(uri))
            except Exception:
                continue
    except Exception:
        pass
    return _dedupe_urls(urls)


def _docx_paragraph_urls(paragraph):
    """Return both visible and embedded hyperlink URLs for a DOCX paragraph."""
    urls = _extract_urls(paragraph.text)
    try:
        # python-docx does not include hyperlink targets in paragraph.text.
        # Resolve r:id relationships directly from the paragraph XML.
        rel_ids = paragraph._p.xpath('.//w:hyperlink/@r:id')
        for rel_id in rel_ids:
            rel = paragraph.part.rels.get(rel_id)
            if rel is not None and getattr(rel, "target_ref", None):
                target = str(rel.target_ref)
                if target.startswith(("http://", "https://")):
                    urls.append(target)
    except Exception:
        pass
    return _dedupe_urls(urls)


def _append_document_chunks(output_docs, text_splitter, text, base_metadata, urls=None):
    """Split one logical source section and attach only its own relevant URLs.

    Important: URL metadata is attached AFTER logical-source segmentation, so a
    long document containing thousands of links can never copy all of those links
    onto every FAISS chunk.
    """
    if not text or not text.strip():
        return

    urls = _dedupe_urls(urls)
    for chunk in text_splitter.split_text(text.strip()):
        if not chunk.strip():
            continue

        metadata = dict(base_metadata)

        # Prefer URLs actually present in this chunk. If the logical section has
        # exactly one source URL, it is safe to inherit it for all child chunks.
        # If a section has multiple URLs and this chunk contains none of them, do
        # not guess: leaving the chunk uncited is better than showing a wrong link.
        chunk_urls = _extract_urls(chunk)
        relevant_urls = [u for u in urls if u in chunk_urls]
        if not relevant_urls and len(urls) == 1:
            relevant_urls = urls

        if relevant_urls:
            metadata["source_urls"] = _dedupe_urls(relevant_urls)

        output_docs.append(LangchainDocument(page_content=chunk, metadata=metadata))


def _build_docx_sections(docx):
    """Build logical DOCX sections with local URL ownership.

    A paragraph containing a URL/hyperlink starts a new source section. Text that
    follows belongs to that section until another URL-bearing paragraph appears.
    This works well for exported/crawled documents of the form `URL -> content`,
    while also preventing a normal DOCX with many unrelated links from assigning
    every link to every chunk.

    For a DOCX with only one URL in the whole file, all text is treated as one
    source section and may safely inherit that single URL.
    """
    paragraphs = []
    for para in docx.paragraphs:
        text = para.text.strip()
        urls = _docx_paragraph_urls(para)
        if text or urls:
            paragraphs.append((text, urls))

    all_urls = _dedupe_urls([u for _, urls in paragraphs for u in urls])

    if len(all_urls) <= 1:
        full_text = "\n\n".join(text for text, _ in paragraphs if text).strip()
        return [(full_text, all_urls)] if full_text else []

    sections = []
    current_lines = []
    current_urls = []

    def flush():
        nonlocal current_lines, current_urls
        text = "\n\n".join(x for x in current_lines if x).strip()
        if text:
            sections.append((text, _dedupe_urls(current_urls)))
        current_lines = []
        current_urls = []

    for text, para_urls in paragraphs:
        if para_urls:
            # A new URL-bearing paragraph is a source boundary. This is the key
            # difference from the old implementation, which put ALL document URLs
            # in one metadata list before chunking.
            flush()
            current_urls = para_urls
            if text:
                current_lines.append(text)
        elif text:
            current_lines.append(text)

    flush()
    return sections


# Preprocess uploaded files + scraped data into text chunks
async def preprocess_text(files: list[Union[str, 'UploadFile']], size, overlap, scraped_data=None):
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

    for file in files or []:
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
            continue

        lower_name = filename.lower()

        # ---------------- PDF ----------------
        if lower_name.endswith(".pdf"):
            reader = PdfReader(file_object)
            pdf_pages = []

            # First collect URLs page-by-page so repeated header/footer links can
            # be removed across the document before any metadata is attached.
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                # Keep paragraph/line boundaries for better RecursiveCharacterTextSplitter
                # behavior instead of collapsing the entire page to one line.
                cleaned_text = "\n".join(
                    line.strip() for line in page_text.splitlines() if line.strip()
                )
                visible_urls = _extract_urls(cleaned_text)
                annotation_urls = _extract_pdf_annotation_urls(page)
                page_urls = _dedupe_urls(visible_urls + annotation_urls)
                pdf_pages.append((page_num, cleaned_text, page_urls))

            filtered_url_lists = _strip_boilerplate_urls(
                [urls for _, _, urls in pdf_pages]
            )

            for (page_num, cleaned_text, _), page_urls in zip(pdf_pages, filtered_url_lists):
                if not cleaned_text.strip():
                    continue

                base_metadata = {
                    "source_type": "document",
                    "source_name": filename,
                    "page": page_num,
                }

                # Split first, then attach only locally relevant URLs. A page with
                # one genuine source URL can safely pass it to all its chunks; a
                # page with many links will only cite links actually present in a
                # particular chunk instead of blindly copying every page link.
                _append_document_chunks(
                    docs, text_splitter, cleaned_text, base_metadata, page_urls
                )

        # ---------------- DOCX ----------------
        elif lower_name.endswith(".docx"):
            docx = DocxDocument(file_object)
            sections = _build_docx_sections(docx)

            print(f"[DOCX] ➤ {filename}: built {len(sections)} logical source section(s).")
            for section_num, (section_text, section_urls) in enumerate(sections, start=1):
                base_metadata = {
                    "source_type": "document",
                    "source_name": filename,
                    "section": section_num,
                }
                _append_document_chunks(
                    docs, text_splitter, section_text, base_metadata, section_urls
                )

        else:
            print(f"⚠️ [PREPROCESS] Unsupported document type skipped: {filename}")

    print(f"📄 Total document-based chunks before scraped data: {len(docs)}")
    print("🧩 First 5 extracted docs:")
    for i, d in enumerate(docs[:5]):
        print(f"{i+1}. {d.page_content[:100]}...")
        print(f"   metadata={d.metadata}")

    # ---------------- Scraped web data ----------------
    if scraped_data:
        seen_hashes = set()
        dropped_dupes = 0
        dropped_linklists = 0

        if isinstance(scraped_data, str):
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
                _append_document_chunks(
                    docs,
                    text_splitter,
                    chunk,
                    {"source_type": "web", "source_url": None},
                    [],
                )

        elif isinstance(scraped_data, list):
            for item in scraped_data:
                if not (isinstance(item, dict) and "full_text" in item):
                    continue

                url = item.get("url")
                raw_chunks = [
                    chunk.strip()
                    for chunk in item["full_text"].split("\n\n")
                    if chunk.strip()
                ]

                for chunk in raw_chunks:
                    if _is_link_list_chunk(chunk):
                        dropped_linklists += 1
                        continue

                    chunk_hash = hashlib.sha256(_normalize_for_hash(chunk).encode("utf-8")).hexdigest()
                    if chunk_hash in seen_hashes:
                        dropped_dupes += 1
                        continue
                    seen_hashes.add(chunk_hash)

                    # Scraped pages already have a canonical source_url. Keep that
                    # one-to-one page ownership unchanged.
                    base_metadata = {"source_type": "web", "source_url": url}
                    for split_chunk in text_splitter.split_text(chunk):
                        if split_chunk.strip():
                            docs.append(LangchainDocument(
                                page_content=split_chunk,
                                metadata=dict(base_metadata),
                            ))

        print(f"🧹 [DEDUP] ➤ Dropped {dropped_linklists} link-list-shaped chunks "
              f"and {dropped_dupes} exact/near-duplicate chunks from scraped data.")

    docs = [d for d in docs if d.page_content.strip()]
    return docs

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