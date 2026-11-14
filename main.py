# main.py
from pathlib import Path
from conversation_layer import finalize_answer, conversational_guardrail, is_casual_social
import builtins as _console
import logging as _logging
import contextvars as _contextvars

# Suppress verbose legacy diagnostic prints; request summaries use _console.print.
def print(*args, **kwargs):
    return None

from metadata_router import load_metadata, distinct_products
from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Request, Header
from typing import List, Optional
from preprocess import preprocess_vectordbs
from inference import (
    inference,
    clean_customer_text,
    validate_greeting,
    _aura_stage_timings,
)
from webscrape import scrape_web_data
import uvicorn
import os, json, validators, pickle
import tldextract
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
import threading
import uuid
import re
import hmac
import time
from functools import lru_cache

from metadata_router import (
    load_metadata,
    route_question_llm,
    rank_metadata_urls,
    recommendation_candidates,
    select_urls_with_llm,
    is_generic_savings_question,
    savings_overview_urls,
    normalize_url,
)

from inference import inference_selected_urls, indexed_source_urls

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

print("Theme admin key configured:", bool(os.getenv("THEME_ADMIN_KEY")))

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Initialize SQLite DB
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "chatlog.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS chatlog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT,
    ip TEXT,
    project_name TEXT,
    timestamp TEXT
)
''')
conn.commit()


def save_chat_log(question, answer, ip, project_name):
    """Persist each completed chatbot turn without sharing a global cursor."""
    try:
        with sqlite3.connect(DB_PATH, timeout=15) as db:
            db.execute("""
                INSERT INTO chatlog (question, answer, ip, project_name, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (str(question), str(answer), str(ip), str(project_name),
                  datetime.utcnow().isoformat()))
        _console.print("[SQLITE] Chat saved successfully", flush=True)
    except Exception as exc:
        _console.print(f"[SQLITE ERROR] Failed to save chat: {exc!r}", flush=True)


# Allow frontend CORS origins
origins = [
    os.getenv("AWS_IP"),
    os.getenv("DOMAIN1"),
    os.getenv("DOMAIN2"),
    os.getenv("LOCAL"),
    os.getenv("DATALYSIS"),
    os.getenv("GENAITECHSOL"),
    os.getenv("KUNJWEB"),
    os.getenv("RAHI"),
    os.getenv("LUCKPAY"),
    os.getenv("ANUWEB"),
    os.getenv("BAJAJ"),
    os.getenv("DOMAIN3"),
    os.getenv("AUBANK1"),
    os.getenv("AUBANK2")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", os.path.join(BASE_DIR, "projects"))
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print("🧱 BASE_OUTPUT_DIR =", BASE_OUTPUT_DIR)
print("🧱 Contents of BASE_OUTPUT_DIR:", os.listdir(BASE_OUTPUT_DIR))


session_state = {
    "retriever": None,
    "preprocessing_done": False,
    "index": None,
    "docstore": None,
    "embedding_model_global": None,
    "selected_vectordb": "FAISS",
    "selected_chat_model": None,
    "messages": []
}

conversation_states = {}
conversation_lock = threading.Lock()

# ------------------------ 🔧 Helper to Rebuild FAISS Retriever ------------------------ #
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

import faiss
import pickle
import os

async def rebuild_faiss_retriever(index_path: str):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)

    abs_index_path = os.path.abspath(index_path)
    index_faiss_path = os.path.join(abs_index_path, "index.faiss")
    index_pkl_path = os.path.join(abs_index_path, "index.pkl")

    # ✅ Proper FAISS load with docstore + mapping
    index = faiss.read_index(index_faiss_path)

    with open(index_pkl_path, "rb") as f:
        store_data = pickle.load(f)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=store_data["docstore"],
        index_to_docstore_id=store_data["index_to_docstore_id"]
    )

    # ✅ Optional debug print
    print("✅ Docstore size:", len(vectorstore.docstore._dict))
    print("✅ Index-to-docstore-id mapping keys (sample):", list(vectorstore.index_to_docstore_id.keys())[:5])

    keys = list(vectorstore.docstore._dict.keys())[:5]
    for i, k in enumerate(keys):
        print(f"[✅] Doc {i+1} (ID: {k}) preview: {vectorstore.docstore._dict[k].page_content[:120]}")

    return vectorstore.as_retriever()

@app.post("/preprocess")
async def preprocess(
    request: Request,
    project_name: str = Form(None),
    doc_files: Optional[List[UploadFile]] = File(None),
    metadata_file: Optional[UploadFile] = File(None),
    links: Optional[str] = Form(None),
    embedding_model: str = Form(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...)
):
    try:
        print("\n🛠️ [PREPROCESS] ➤ Started preprocessing request.")

        # ✅ Ensure at least one input is provided
        if not doc_files and not links:
            raise HTTPException(status_code=400, detail="❌ You must provide at least one PDF or a URL.")

                # ✅ Process links if provided
        links_list = []
        if links:
            try:
                links_list = json.loads(links)
                print(f"🌐 [PREPROCESS] ➤ Received {len(links_list)} links for scraping.")
                for link in links_list:
                    print(f"🔗 Validating link: {link}")
                    if not validators.url(link):
                        raise HTTPException(status_code=400, detail=f"❌ Invalid URL: {link}")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="❌ Invalid JSON format for 'links' field.")

        # ✅ Process files if provided
        if doc_files:
            print(f"📎 [PREPROCESS] ➤ Uploaded {len(doc_files)} files:")
            for file in doc_files:
                print(f"   └── {file.filename}")
                if file.filename == "":
                    raise HTTPException(status_code=400, detail="❌ One of the uploaded files is empty!")

        # ✅ Derive domain — use project_name as-is when given; only fall back to
        # tldextract when we have no project_name and must derive one from a URL.
        if project_name:
            domain = project_name.strip()
            print(f"📂 [PREPROCESS] ➤ Using provided project_name as domain: {domain}")
        elif links_list:
            domain_info = tldextract.extract(links_list[0])
            domain = f"{domain_info.subdomain + '.' if domain_info.subdomain else ''}{domain_info.domain}.{domain_info.suffix}"
            print(f"🌍 [PREPROCESS] ➤ Derived domain from links: {domain}")
        else:
            raise ValueError("Either project_name or links_list must be provided.")

        domain_folder = os.path.join(BASE_OUTPUT_DIR, domain)
        os.makedirs(domain_folder, exist_ok=True)
        print(f"📁 [PREPROCESS] ➤ Created/using domain folder: {domain_folder}")
        
        metadata_stats = None
        metadata_temp_path = None

        if metadata_file is not None:
            if not metadata_file.filename.lower().endswith(".json"):
                raise HTTPException(
                    status_code=400,
                    detail="Metadata must be a JSON file."
                )

            raw_metadata = await metadata_file.read()

            if len(raw_metadata) > 25 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="Metadata JSON exceeds the 25 MB limit."
                )

            metadata_temp_path = os.path.join(
                domain_folder, "metadata.uploading.json"
            )

            try:
                with open(metadata_temp_path, "wb") as f:
                    f.write(raw_metadata)

                metadata_records = load_metadata(
                    metadata_temp_path
                )

                metadata_stats = {
                    "total_urls": len(metadata_records),
                    "distinct_products": len(
                        distinct_products(metadata_records)
                    )
                }

                print(
                    "[METADATA] Validated:",
                    metadata_stats
                )

            except (ValueError, json.JSONDecodeError, UnicodeError) as e:
                if os.path.exists(metadata_temp_path):
                    os.remove(metadata_temp_path)

                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid metadata JSON: {e}"
                )


                # ✅ Scrape web data if links provided — but reuse existing scraped
        # data for this project if it was already scraped for these exact
        # links, instead of re-crawling every time. Avoids hammering
        # already-sensitive sites (bot protection, rate limits) on repeat
        # requests for the same project+URLs.
        scraped_data = []
        if links_list:
            cache_path = os.path.join(domain_folder, "scraped_cache.pkl")
            meta_path = os.path.join(domain_folder, "scrape_meta.json")
            sorted_links = sorted(links_list)

            reused_cache = False
            if os.path.exists(cache_path) and os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if meta.get("links") == sorted_links:
                        with open(cache_path, "rb") as f:
                            scraped_data = pickle.load(f)
                        reused_cache = True
                        print(f"♻️  [SCRAPER] ➤ Reusing existing scraped data for "
                              f"'{domain}' — same links already scraped "
                              f"({len(scraped_data)} pages cached).")
                except Exception as e:
                    print(f"⚠️ [SCRAPER] ➤ Couldn't read existing cache/meta, "
                          f"will re-scrape: {e}")

            if not reused_cache:
                try:
                    print("🕸️ [SCRAPER] ➤ Starting web scraping...")
                    for link in links_list:
                        scraped_data.extend(await scrape_web_data(link))
                        print(f"✅ [SCRAPER] ➤ Scraped content from: {link}")
                    with open(cache_path, "wb") as f:
                        pickle.dump(scraped_data, f)
                    with open(meta_path, "w") as f:
                        json.dump({"links": sorted_links}, f)
                    print(f"💾 [SCRAPER] ➤ Scraped data cached at: {domain_folder}")
                except Exception as e:
                    print(f"❌ [SCRAPER] ➤ Web scraping failed: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Web scraping failed: {str(e)}")

        # ✅ Proceed to vector DB preprocessing
        print("🧠 [VECTORDB] ➤ Calling preprocess_vectordbs...")
        try:
            index, docstore, index_to_docstore_id, vectorstore, retriever, embedding_model_global, pinecone_index_name, vs, qdrant_client = await preprocess_vectordbs(
                doc_files or [], embedding_model, chunk_size, chunk_overlap,
                scraped_data, session_state["selected_vectordb"],
                persist_directory=os.path.join(domain_folder, "faiss_index")
            )
            print(f"🧾 [POST-PROCESS] Docstore preview: {list(docstore._dict.keys())[:5]}")
            print(f"🔗 [POST-PROCESS] Index to Docstore ID: {list(index_to_docstore_id.items())[:5]}")
            print("✅ [VECTORDB] ➤ Embedding & indexing complete.")

            session_state.update({
                "retriever": retriever,
                "preprocessing_done": True,
                "index": index,
                "docstore": docstore,
                "embedding_model_global": embedding_model_global,
                "pinecone_index_name": pinecone_index_name,
                "vs": vs,
                "qdrant_client": qdrant_client
            })

            state_to_save = session_state.copy()
            state_to_save["embedding_model_name"] = embedding_model
            state_to_save.pop("retriever", None)
            state_to_save.pop("index", None)
            state_to_save.pop("docstore", None)

            with open(os.path.join(domain_folder, "session_state.pkl"), "wb") as f:
                pickle.dump(state_to_save, f)

            print("💾 [STATE] ➤ Session state saved to:", domain_folder)

            if metadata_temp_path:
                os.replace(
                    metadata_temp_path,
                    os.path.join(domain_folder, "metadata.json")
                )

            return {
                "message": f"Preprocessing completed and saved in {domain_folder}",
                "project_name": domain,
                "metadata_enabled": metadata_stats is not None,
                "metadata": metadata_stats
            }

        except Exception as e:
            print(f"❌ [VECTORDB] ➤ Error in preprocess_vectordbs: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    except Exception as e:
        print(f"❌ [PREPROCESS] ➤ Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

@app.post("/select_vectordb")
async def select_vectordb(vectordb: str = Form(...)):
    session_state["selected_vectordb"] = vectordb
    print(f"✅ Selected Vector DB: {vectordb}")
    return {"message": f"Vector DB set to: {vectordb}"}

@app.post("/select_chat_model")
async def select_chat_model(
    chat_model: str = Form(...),
    custom_prompt: str = Form(None),
    project_name: str = Form(None),
    max_output_tokens: int = Form(1024),  
    initial_message: str = Form("Hello! 👋 How can I help you today?"),
    chatbot_title: str = Form("AI Chat Assistant"),
    top_k: int = Form(8),  
    temperature: float = Form(0.3),
    quick_questions: str = Form(None)  # NEW: JSON-encoded array from the frontend
):
    session_state["selected_chat_model"] = chat_model
    session_state["custom_prompt"] = custom_prompt
    session_state["max_output_tokens"] = max_output_tokens 
    session_state["initial_message"] = initial_message
    session_state["chatbot_title"] = chatbot_title
    session_state["top_k"] = top_k
    session_state["temperature"] = temperature

    # NEW: parse the quick_questions JSON string into an actual list.
    # Falls back to [] on missing/invalid input rather than raising, since
    # this field is optional and we don't want a bad value to 500 the whole save.
    try:
        parsed_quick_questions = json.loads(quick_questions) if quick_questions else []
        if not isinstance(parsed_quick_questions, list):
            parsed_quick_questions = []
    except (json.JSONDecodeError, TypeError):
        parsed_quick_questions = []
    session_state["quick_questions"] = parsed_quick_questions

    print(f"✅ Value of k: {top_k} & Temperature: {temperature}")
    print(f"✅ Chat model set: {chat_model}, Prompt: {custom_prompt}, Max Tokens: {max_output_tokens}")
    print(f"📝 Title: {chatbot_title}, Initial Message: {initial_message}")
    print(f"❓ Quick Questions: {parsed_quick_questions}")

    # Use project_name directly
    domain = project_name or "local_upload"
    domain_folder = os.path.join(BASE_OUTPUT_DIR, domain)

    # Load existing session state pickle (if exists)
    session_file = os.path.join(domain_folder, "session_state.pkl")
    if os.path.exists(session_file):
        try:
            with open(session_file, "rb") as f:
                loaded_session = pickle.load(f)
        except:
            loaded_session = {}
    else:
        loaded_session = {}

    # Update loaded session
    loaded_session["selected_chat_model"] = chat_model
    loaded_session["custom_prompt"] = custom_prompt
    loaded_session["max_output_tokens"] = max_output_tokens  # NEW
    loaded_session["initial_message"] = initial_message
    loaded_session["chatbot_title"] = chatbot_title
    loaded_session["top_k"] = top_k
    loaded_session["temperature"] = temperature
    loaded_session["quick_questions"] = parsed_quick_questions  # NEW


    # Save back to file
    os.makedirs(domain_folder, exist_ok=True)
    with open(session_file, "wb") as f:
        pickle.dump(loaded_session, f)

    print(f"💾 [STATE] ➤ Chat model saved in session state at: {session_file}")

    return {"message": "Chat model and max_output_tokens set."}


# ---------------------------- 💬 CHAT Endpoint ---------------------------- #
class ChatRequest(BaseModel):
    query: str
    project_name: str
    conversation_id: Optional[str] = None

from fastapi import FastAPI, Request, HTTPException
from langdetect import detect
from datetime import datetime


from functools import lru_cache


@lru_cache(maxsize=4)
def _load_metadata_index(index_dir, model_name, faiss_mtime, pickle_mtime):
    import faiss
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index.pkl"), "rb") as f:
        stored = pickle.load(f)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=stored["docstore"],
        index_to_docstore_id=stored["index_to_docstore_id"]
    )
    return embeddings, vectorstore



# Dedicated authoritative sources. The uploaded PDF is bundled alongside the backend.
AU_LEA_PDF_URL = "https://www.au.bank.in/pdf/contact-us/point-of-contacts-for-law-enforcement-and-intelligence-agencie.pdf"
AU_LEA_COMPLAINT_URL = "https://www.au.bank.in/support/contact-us/raise-complaint"
AU_ABOUT_URL = "https://www.au.bank.in/about-us"

def _is_lea_query(question):
    q = str(question).lower()
    return bool(re.search(r"\b(lea|leas|law enforcement|intelligence agenc(?:y|ies)|ncrp|cyber police|police notice|police requisition)\b", q))

def _is_branch_count_query(question):
    q = str(question).lower()
    return bool(re.search(r"\b(branch|branches)\b", q) and
                re.search(r"\b(how many|number|total|count|network)\b", q))

@lru_cache(maxsize=1)
def _lea_pdf_text():
    path = os.path.join(BASE_DIR, "lea_source.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError("Bundled LEA PDF extraction missing: " + path)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()

def _source_grounded_special_answer(question, loaded_session, response_language, faiss_index_dir):
    from inference import run_chat_model, append_web_sources
    if _is_lea_query(question):
        evidence = _lea_pdf_text()
        if not evidence.strip():
            return "The AU Bank law-enforcement contact document could not be read."
        instructions = ("Answer ONLY from the official AU Bank law-enforcement contact PDF. "
                        "Give only the requested names, email addresses, phone numbers or escalation contacts. "
                        "Do not invent any details or discuss unrelated services. "
                        + answer_language_instruction(response_language, question))
        result = run_chat_model(loaded_session.get("selected_chat_model"), evidence[:12500],
                                question, [], custom_instructions=instructions,
                                max_output_tokens=1100, temperature=0)
        return append_web_sources(clean_customer_text(result), [AU_LEA_COMPLAINT_URL], question=question)
    if _is_branch_count_query(question):
        # About Us was checked in the supplied DOCX. Its dynamic counters
        # appear as literal zero, NOT an actual branch count.
        return ("The scraped AU Bank About Us page does not provide a verifiable "
                "numeric branch count: its dynamic counters were captured as 0. "
                "Please check the official About Us page for the latest figure.\n\n"
                "Related search results:\n" + AU_ABOUT_URL)
    return None

def strip_social_links(reply):
    """Defense in depth: social replies must never expose citations or URLs."""
    if not isinstance(reply, str):
        return reply
    # Remove an entire related-results/sources section, if generated.
    reply = re.split(r"(?im)^\s*(?:Related search results|Sources)\s*:\s*$", reply, maxsplit=1)[0]
    reply = re.sub(r"\[[^\]\n]*\]\(https?://[^)]+\)", "", reply)
    reply = re.sub(r"https?://[^\s<>]+", "", reply)
    return clean_customer_text(reply).strip()


def conversational_social_reply(question, chat_model, response_language):
    """LLM-based conversational gate: no fixed greeting phrase list.

    Returns None for substantive banking requests and unrelated questions.
    """
    from inference import run_chat_model, get_current_datetime
    now_ist = get_current_datetime()
    rules = (
        "You are a narrow conversational intent classifier for AU Bank's chatbot. "
        "Classify the current message into SOCIAL, BANKING, or OTHER. "
        "SOCIAL includes greetings, checking how the assistant is, thanks, farewells, "
        "and short courteous small talk that does not request unrelated factual content. "
        "BANKING includes any question or request about banking, AU Bank, products, "
        "accounts, loans, deposits, cards, and their follow-ups, EVEN IF it begins with a greeting. "
        "OTHER includes requests about unrelated subjects, such as movies, sports, recipes, or coding. "
        "Return ONLY one of these exact words: SOCIAL, BANKING, OTHER. "
        "Do not answer the message."
    )
    classification = run_chat_model(
        chat_model, "", question, [], custom_instructions=rules,
        max_output_tokens=160, temperature=0)
    label = str(classification).strip().upper()
    if label != "SOCIAL":
        return None
    language_rule = answer_language_instruction(response_language, question)
    reply_rules = (
        "You are AUra+, AU Bank's friendly, professional banking assistant. "
        "Respond naturally, with a little personality, to the user's SOCIAL message, in 1-3 short sentences. "
        "React to what the user actually said; a casual how-are-you deserves a friendly answer "
        "and a conversational follow-up, not a generic product menu. "
        "Acknowledge celebrations and difficult moments. For insults, remain calm, set polite boundaries, and offer help. "
        "Do not answer unrelated factual questions or pretend to have human experiences. "
        "Gently invite a banking question, but do not repeat a robotic scope disclaimer. "
        "Use the current Indian time to validate time-of-day greetings. "
        f"The current time in Asia/Kolkata is {now_ist}. "
        "Morning is 05:00-11:59, afternoon 12:00-16:59, evening 17:00-20:59, "
        "and late night 21:00-04:59. If the user's time-of-day greeting is wrong, "
        "politely use the correct greeting without lecturing them. "
        "If the greeting is correct, reciprocate it. "
        "For late-night conversational openings, a simple Hello is appropriate; "
        "respond naturally to Good night as a farewell if that is the user's intent. "
        "Do not include URLs, sources, markdown, or financial claims. " + language_rule
    )
    # Reasoning-capable models may spend the entire output budget on reasoning.
    # Never return an empty assistant message to the frontend.
    try:
        reply = run_chat_model(
            chat_model, "", question, [], custom_instructions=reply_rules,
            max_output_tokens=450, temperature=0.25)
        reply = strip_social_links(reply or "")
        if reply:
            return reply
    except Exception as exc:
        print("[SOCIAL] Generation failed:", repr(exc))

    # Safe, non-factual fallback; no source retrieval is needed for small talk.
    if response_language in ("hi", "hi-Latn"):
        return ("नमस्कार! AU Bank से जुड़ी किसी जानकारी में मैं आपकी कैसे मदद कर सकता हूँ?"
                if response_language == "hi" else
                "Namaskar! AU Bank se judi kisi jaankari mein main aapki kaise madad kar sakta hoon?")
    return "Hello! I'm here to help with your AU Bank questions. How can I assist you?"


def run_scope_reply(question, chat_model, language_instruction):
    from inference import run_chat_model
    return run_chat_model(
        chat_model=chat_model, context="", question=question, chat_history=[],
        custom_instructions=(
            "This question is outside the AU Bank chatbot's scope. "
            "Reply briefly and politely that you can help with AU Bank products "
            "and services, such as accounts, deposits, loans, and cards. "
            "Do not answer the unrelated question. End with a friendly invitation "
            "such as: Would you like to know about any of them? "
            "Do not ask which specific product the user means. " + language_instruction),
        max_output_tokens=120, temperature=0)


def retrieval_query_in_english(question, chat_model):
    """Semantic query rewrite only; never replace the customer's original wording."""
    from inference import run_chat_model
    # A single selected-model call for translation and retrieval normalization.
    try:
        result = run_chat_model(
            chat_model=chat_model, context="", question=question, chat_history=[],
            custom_instructions=(
                "TASK: Translate/rewrite the USER QUESTION into a short, precise English "
                "banking search query. Preserve all product names, conditions, customer "
                "types, amounts, tenure, and the exact intent. Translate Gujarati, Hindi, "
                "Hinglish and any other language. Output ONLY the English search query; "
                "no explanations, no answers, no quotes, no markdown."
            ), max_output_tokens=180, temperature=0)
        result = str(result).strip().strip('\"')
        if result and len(result) < 600 and not result.startswith('{'):
            return result
    except Exception as exc:
        print('[MULTILINGUAL] Query rewrite failed; using original:', exc)
    return question


# Deterministic script-first language selection. langdetect is unreliable for
# short banking inputs (e.g. "I want PL" can be mislabeled as Swahili).
_SCRIPT_LANGUAGES = (
    (r"[\u0A80-\u0AFF]", "gu"), (r"[\u0900-\u097F]", "hi"),
    (r"[\u0A00-\u0A7F]", "pa"), (r"[\u0980-\u09FF]", "bn"),
    (r"[\u0B80-\u0BFF]", "ta"), (r"[\u0C00-\u0C7F]", "te"),
    (r"[\u0C80-\u0CFF]", "kn"), (r"[\u0D00-\u0D7F]", "ml"),
    (r"[\u0600-\u06FF]", "ur"),
)
_ROMANIZED_HINDI = re.compile(
    r"\b(?:mujhe|mera|meri|mere|kaise|kya|kitna|kitni|chahiye|batao|"
    r"bataiye|ke\s+liye|mein|nahi|hai|hain|karna|karo|sabse|zyada|"
    r"khata|kholna|chahata|chahta)\b", re.I)

def detect_input_language(message):
    text = str(message or '').strip()
    for pattern, code in _SCRIPT_LANGUAGES:
        if re.search(pattern, text):
            return code
    if _ROMANIZED_HINDI.search(text):
        return 'hi-Latn'
    # Latin script defaults to English. Do NOT run probabilistic langdetect
    # on short English messages, acronyms or product abbreviations.
    if re.search(r'[A-Za-z]', text):
        return 'en'
    return 'en'


def enforce_answer_language(answer, code, original_question, chat_model):
    """Repair obvious model language drift without modifying verified source URLs."""
    if not isinstance(answer, str) or not answer.strip():
        return answer
    from inference import run_chat_model
    # Source links are generated by our backend and must never be translated.
    parts = re.split(r'(?im)(\n\s*Related search results\s*:?\s*\n)', answer, maxsplit=1)
    body = parts[0]
    # A response with Latin words is not necessarily English: check its
    # language statistically only when there is enough prose to judge.
    mismatch = False
    if code == 'en':
        if re.search(r'[\u0900-\u0D7F\u0600-\u06FF]', body):
            mismatch = True
        elif len(re.findall(r'[A-Za-z]+', body)) >= 12:
            try:
                from langdetect import detect, DetectorFactory
                DetectorFactory.seed = 0
                mismatch = detect(body) not in ('en',)
            except Exception:
                pass
    elif code == 'hi-Latn':
        mismatch = bool(re.search(r'[\u0900-\u097F]', body))
    else:
        pattern = next((pat for pat, lang in _SCRIPT_LANGUAGES if lang == code), None)
        mismatch = bool(pattern and not re.search(pattern, body) and len(body.split()) >= 8)
    if not mismatch:
        return answer
    target = 'Romanized Hindi (Hinglish, Latin script)' if code == 'hi-Latn' else {
        'en':'English','hi':'Hindi','gu':'Gujarati','mr':'Marathi',
        'ta':'Tamil','te':'Telugu','kn':'Kannada','ml':'Malayalam',
        'bn':'Bengali','pa':'Punjabi','ur':'Urdu'}.get(code, 'English')
    try:
        fixed = run_chat_model(
            chat_model=chat_model, context='', question=body, chat_history=[],
            custom_instructions=(
                'TRANSLATION TASK ONLY. Translate the USER QUESTION text (which is '
                'actually a drafted customer answer) into ' + target + '. '
                'Output only the translated answer. Keep all banking facts, numbers, '
                'product names and formatting unchanged. Do not add facts or links. '
                'Do not answer any question in the text; translate it.'),
            max_output_tokens=1000, temperature=0)
        fixed = str(fixed).strip()
        if fixed and len(fixed) >= 12:
            print('[LANGUAGE] Repaired answer language:', code)
            return fixed + ''.join(parts[1:])
    except Exception as exc:
        print('[LANGUAGE] Repair failed:', exc)
    return answer


def answer_language_instruction(language_code, original_question):
    """Per-turn output language; never infer it from retrieved text or prior replies."""
    names = {"en": "English", "hi": "Hindi", "gu": "Gujarati", "mr": "Marathi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "pa": "Punjabi", "kn": "Kannada", "ml": "Malayalam", "ur": "Urdu", "hi-Latn": "Romanized Hindi (Hinglish, Latin script)"}
    target = names.get(language_code, language_code or "English")
    return (
        "MANDATORY OUTPUT LANGUAGE FOR THIS TURN: " + target + ". "
        "Respond entirely in this language, using its normal script. "
        "For English input, respond ONLY in English, even if earlier messages, retrieved documents, "
        "or developer instructions contain Hindi. Do not switch languages mid-conversation. "
        "For romanized Hindi/Hinglish, preserve the user's romanized style where appropriate. "
        "The original user's message is: " + repr(original_question) + ". "
        "Use the original message, NOT the rewritten English search query, to decide output language. "
        "Give all evidence-supported details directly. When relevant information is partial and related "
        "source links will be provided, finish with 'Please refer to the related search results below for more information.' "
        "Translate that closing sentence into the required output language. "
        "Do not add generic statements that other information is unavailable, unless an essential "
        "specifically requested fact is truly unsupported. Never invent missing details."
    )


@lru_cache(maxsize=8)
def _load_metadata_cached(path, mtime_ns, size):
    """Reuse parsed metadata until the file changes on disk."""
    return load_metadata(path)




def _fd_question_intent(question):
    q = str(question or "").lower()
    is_fd = bool(re.search(r"\b(fd|fixed[ -]?deposits?|term[ -]?deposits?)\b", q))
    return {
        "fd": is_fd,
        "calculator": is_fd and bool(re.search(r"\b(calculat(?:e|or|ion)|estimate|maturity amount|returns?)\b", q)),
        "manage": bool(re.search(r"\b(manage|update|modify|change|close|renew|download|certificate|existing|track|status|premature|withdraw)\b", q)),
        "apply": bool(re.search(r"\b(open|book|apply|start|create|invest|new fd)\b", q)),
        "eligibility": bool(re.search(r"\b(eligib|qualif|who can)\b", q)),
        "documents": bool(re.search(r"\b(documents?|paperwork|kyc|proof)\b", q)),
        "green": bool(re.search(r"\bgreen[ -]?(?:fd|fixed[ -]?deposit)\b", q)),
        "bulk": bool(re.search(r"\bbulk[ -]?(?:fd|fixed[ -]?deposit)\b", q)),
        "variants": bool(re.search(r"\b(types?|variants?|compare|comparison|versus|vs)\b", q)),
    }


def _intent_profile(question):
    q = str(question or "").lower()
    return {
        "calculator": bool(re.search(r"\b(?:calculator|calculate|calculation)\b", q)),
        "documents": bool(re.search(r"\b(?:documents?|documentation|paperwork|kyc|proofs?|salary slip)\b", q)),
        "eligibility": bool(re.search(r"\b(?:eligib\w*|qualif\w*|who can)\b", q)),
        "activation": bool(re.search(r"\b(?:enable|activat\w*|register|registration|set\s*up|first.time)\b", q)),
        "opening": bool(re.search(r"\b(?:open|opening|apply|application|book|start)\b", q)),
        "servicing": bool(re.search(r"\b(?:manage|update|modify|close|renew|download|certificate|existing|track|status|premature|withdraw)\b", q)),
        "rates": bool(re.search(r"\b(?:rates?|interest|yield)\b", q)),
        "specialized": bool(re.search(r"\b(?:self.employed|freelanc\w*|without\s+(?:a\s+)?salary|nri|nre|green fd|bulk fd)\b", q)),
    }


def _page_relevance(question, record, product):
    """Conservative intent-aware score; does not treat incidental keyword mentions as evidence."""
    from urllib.parse import urlsplit
    q = str(question or "").lower()
    intent = _intent_profile(q)
    path = urlsplit(record["url"]).path.lower()
    summary = record.get("summary", "").lower()
    category = str(record.get("category", "")).lower()
    p = str(product or "").lower()
    score = 0.0
    if category == "main": score += 18
    if "/blogs/" in path: score -= 25
    if "/exclusive-news" in path: score -= 90
    if "/support/" in path: score -= 4
    if path in ("/loans", "/personal-banking/loans", "/interest-rate", "/interest-rates"):
        score -= 35
    if "crowdfunding" in path or "commercial-and-small-finance" in path: score -= 75
    if intent["documents"]:
        if re.search(r"document|kyc|salary.slip|paperwork|proof", path): score += 35
        if re.search(r"document|kyc|proof", summary): score += 10
        if "personal-loan" in path and "/blogs/" not in path: score += 48
        if not intent["specialized"] and re.search(r"without.salary|self.employed|freelanc", path): score -= 95
    if intent["eligibility"]:
        if re.search(r"eligib|qualif|criteria|requirement", path): score += 65
        if re.search(r"eligib|qualif|criteria", summary): score += 12
    if intent["activation"]:
        if re.search(r"activat|enable|registr|first.time|set.up|login|netbanking", path): score += 45
        if re.search(r"activat|enable|registr|set up", summary): score += 16
        if "digital-transactions-limit" in path: score -= 85
        if "/blogs/" in path: score -= 45
    if intent["opening"]:
        if re.search(r"open|apply|application|book", path): score += 50
        if re.search(r"open|apply|application|video kyc", summary): score += 12
        if "savings-account" in path and "/support/" not in path: score += 50
    if intent["calculator"]:
        if "calculator" in path: score += 110
        else: score -= 18
    if intent["rates"] and "interest-rate" in path: score += 38
    if not intent["servicing"] and re.search(r"interest.certificate|download|view.or.update|update.your.pan|error", path): score -= 75
    if not intent["specialized"] and re.search(r"nri.or.nre|nr.customer|green.fd|bulk.fd", path): score -= 65
    if "msme" in p:
        if re.search(r"msme|micro.small|small.business|business.loan|secured.business", path): score += 35
        if not re.search(r"msme|business.loan|secured.business|small.enterprise|micro.enterprise", path + " " + summary): score -= 65
    if "netbanking" in p or "net_banking" in p:
        if re.search(r"netbanking|net.banking|internet.banking", path): score += 55
        if not re.search(r"netbanking|net banking|internet banking", path + " " + summary): score -= 60
    return score


def _select_intent_sources(question, product, records, available_urls, candidates, limit=4):
    """Use indexed, product-matching pages; refuse irrelevant fallbacks."""
    from metadata_router import canonical_product
    from urllib.parse import urlsplit
    available = {normalize_url(u) for u in available_urls}
    target = canonical_product(product)
    aliases = {target}
    if target == "msme_loan":
        aliases.add("msme_business_loan")
    if target in {"netbanking", "au_net_banking", "net_banking", "au_0101_netbanking"}:
        # Metadata uses au_net_banking; LLM/canonicalizer may return netbanking.
        # Match both catalog identifiers without changing the routed product.
        aliases.update({"netbanking", "au_net_banking", "net_banking", "au_0101_netbanking"})
    pool = [r for r in records if normalize_url(r["url"]) in available
            and any(canonical_product(p) in aliases for p in r.get("product", []))]
    base = {normalize_url(r["url"]): float(r.get("score", 0)) for r in candidates}
    # Include pages beyond top-12 semantic candidates, but do not invent any source.
    ranked = sorted(pool, key=lambda r: (_page_relevance(question, r, product)
                     + min(15, base.get(normalize_url(r["url"]), 0)*12)), reverse=True)
    intent = _intent_profile(question)
    if intent["activation"] and "netbanking" in str(product).lower():
        ranked = [r for r in ranked if (
                      re.search(r"netbanking|net.banking|internet.banking|registration|activation", r["url"], re.I)
                      or re.search(r"(?:enable|activate|register|registration|set up).{0,75}(?:netbanking|net banking|internet banking)", r["summary"], re.I))
                  and not re.search(r"/blogs/|digital-transactions-limit", r["url"], re.I)]
    if intent["eligibility"] and "msme" in str(product).lower():
        ranked = [r for r in ranked if re.search(r"msme|business.loan|secured.business|small.enterprise|micro.enterprise", r["url"] + " " + r["summary"], re.I)
                  and not re.search(r"/blogs/(?:what-is-crowdfunding|difference-between-commercial)|exclusive-news", r["url"], re.I)]
        ranked.sort(key=lambda r: (int("msme-business-loan-documents-and-eligibility" in r["url"]),
                                   _page_relevance(question, r, product)), reverse=True)
    if intent["documents"] and "personal_loan" in str(product):
        # Documentation questions require pages about DOCUMENTS, not how to
        # click Apply in an app. Keep the primary product page and indexed
        # document/KYC-specific resources; don't pad with generic pages.
        def document_evidence_page(r):
            url = r["url"].lower()
            summary = r.get("summary", "").lower()
            if re.search(r"/savings-account/|company-secretaries|/apply-now/|/au-0101-app|/business/loans/apply-for-loan/", url):
                return False
            if re.search(r"without.salary|self.employed|freelanc", url) and not intent["specialized"]:
                return False
            if url.rstrip("/").endswith("/personal-banking/personal-loan"):
                return True
            # Strong product + documentation signals must coexist.
            is_personal = bool(re.search(r"personal[-_ ]loan", url + " " + summary))
            is_docs = bool(re.search(r"document|documentation|kyc|proof|salary.slip|bank.statement|paperwork", url + " " + summary))
            return is_personal and is_docs
        ranked = [r for r in ranked if document_evidence_page(r)]
        ranked.sort(key=lambda r: (
            int(bool(re.search(r"personal.loan.{0,55}(?:documents?|documentation|kyc)|(?:documents?|documentation|kyc).{0,55}personal.loan", r["url"].lower()))),
            int(r["url"].lower().rstrip("/").endswith("/personal-banking/personal-loan")),
            _page_relevance(question, r, product)
        ), reverse=True)
    if intent["documents"] and "personal_loan" in str(product):
        ranked = [r for r in ranked if not re.search(r"/savings-account/|company-secretaries|/au-0101-app|/apply-for-personal-loan-on-au-0101", r["url"], re.I)]
        ranked.sort(key=lambda r: (int(r["url"].endswith("/personal-banking/personal-loan")),
                                   _page_relevance(question, r, product)), reverse=True)
    if intent["opening"] and "savings_account" in str(product):
        ranked = [r for r in ranked if not re.search(r"/blogs/(?:open-2-savings|open-child-first|open-minor|open-kids)", r["url"], re.I)]
    if intent["documents"] and not intent["specialized"] and "personal_loan" in str(product):
        ranked = [r for r in ranked if not re.search(r"without.salary|self.employed|freelanc", r["url"], re.I)]
    result, seen = [], set()
    for r in ranked:
        path = urlsplit(r["url"]).path.lower().rstrip("/")
        key = "interest-rate-directory" if path in ("/interest-rate", "/interest-rates") else path
        if key in seen: continue
        seen.add(key)
        result.append(r["url"])
        if len(result) >= limit: break
    return result


def _related_metadata_urls(question, products, ranked_by_product,
                           selected_by_product, indexed_urls, records, max_related=5):
    """Only show intent-relevant indexed resources; fewer than five is fine."""
    from metadata_router import canonical_product
    from urllib.parse import urlsplit
    available = {normalize_url(u) for u in indexed_urls}
    product_ids = {canonical_product(p) for p in products}
    if "msme_loan" in product_ids: product_ids.add("msme_business_loan")
    if product_ids.intersection({"au_net_banking", "netbanking", "net_banking", "au_0101_netbanking"}): product_ids.update({"au_net_banking", "netbanking", "net_banking", "au_0101_netbanking"})
    candidates = [r for r in records if normalize_url(r["url"]) in available
                  and any(canonical_product(p) in product_ids for p in r.get("product", []))]
    if product_ids.intersection({"au_net_banking", "netbanking", "net_banking", "au_0101_netbanking"}):
        # Metadata tags may differ from the router's chosen canonical name.
        # Restrict the fallback to verified indexed NetBanking paths.
        existing = {normalize_url(r["url"]) for r in candidates}
        for r in records:
            url = normalize_url(r["url"])
            path = urlsplit(url).path.lower()
            if (url in available and url not in existing and
                    ("netbanking" in path or "net-banking" in path)):
                candidates.append(r)
                existing.add(url)
    base = {normalize_url(c["url"]): c.get("score", 0)
            for ranked in ranked_by_product.values() for c in ranked}
    ranked = sorted(candidates, key=lambda r: max(_page_relevance(question, r, p) for p in products)
                    + min(15, base.get(normalize_url(r["url"]), 0)*12), reverse=True)
    result, seen = [], set()
    intent = _intent_profile(question)
    for r in ranked:
        path = urlsplit(r["url"]).path.lower().rstrip("/")
        key = "interest-rate-directory" if path in ("/interest-rate", "/interest-rates") else path
        score = max(_page_relevance(question, r, p) for p in products)
        if score < 0 or key in seen: continue
        if intent["opening"] and "savings_account" in product_ids and re.search(r"/blogs/(?:open-2-savings|open-child-first|open-minor|open-kids)", path): continue
        if intent["documents"] and "au_personal_loan" in product_ids:
            if re.search(r"/savings-account/|company-secretaries|/au-0101-app|/apply-now/|/business/loans/apply-for-loan/", path): continue
            if not (path == "/personal-banking/personal-loan" or (
                re.search(r"personal.loan", path) and re.search(r"document|documentation|kyc|proof|salary.slip", path + " " + r.get("summary", "").lower())
            )): continue
        if not intent["servicing"] and re.search(r"interest.certificate|download|view.or.update|update.your.pan|error", path): continue
        if not intent["specialized"] and re.search(r"nri.or.nre|nr.customer|green.fd|bulk.fd", path): continue
        if intent["activation"] and re.search(r"/blogs/|digital-transactions-limit", path): continue
        if intent["eligibility"] and "msme" in " ".join(products).lower() and re.search(r"/blogs/(?:what-is-crowdfunding|difference-between-commercial)|exclusive-news", path): continue
        seen.add(key); result.append(r["url"])
        if len(result) >= max_related: break
    return result


def _contextual_ctas(question, related_urls, max_ctas=2):
    """Action buttons must target the exact official product/service page.

    Never turn an informational blog or cross-sold account into an application CTA.
    """
    from urllib.parse import urlsplit
    intent = _intent_profile(question)
    q = str(question or "").lower()
    is_savings = bool(re.search(r"\bsavings?\s+accounts?\b", q))
    is_personal_loan = bool(re.search(r"\bpersonal\s+loans?\b", q))
    is_fd = bool(re.search(r"\b(?:fd|fixed\s+deposits?)\b", q))
    is_netbanking = bool(re.search(r"\b(?:net\s*banking|internet\s+banking)\b", q))
    actions = []
    for url in related_urls:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in ("au.bank.in", "www.au.bank.in"):
            continue
        path = parts.path.lower().rstrip("/")
        if "/blogs/" in path or "/support/" in path:
            continue
        entry = None
        if is_savings and intent["opening"] and path == "/personal-banking/savings-account":
            entry = (100, "Explore Savings Accounts", "explore_savings_accounts")
        elif is_personal_loan and path == "/personal-banking/personal-loan":
            if intent["documents"]:
                entry = (100, "View Personal Loan Details", "view_personal_loan")
            elif intent["opening"]:
                entry = (90, "Explore Personal Loan", "explore_personal_loan")
        elif is_fd and path.endswith("/fd-calculator"):
            # Rate queries should offer the calculator as a useful next step too.
            entry = (100 if intent["calculator"] else 85,
                     "Calculate FD Returns", "calculate_fd_returns")
        elif is_fd and path == "/interest-rates/fixed-deposit-interest-rates":
            entry = (95 if not intent["calculator"] else 70,
                     "View FD Interest Rates", "view_fd_interest_rates")
        elif is_netbanking and intent["activation"] and path in (
            "/personal-banking/digital-banking/au-0101/au-0101-netbanking",
            "/personal-banking/digital-banking/online-banking/au-mobile/steps-to-register-au-mobile",
        ):
            entry = (90, "Set Up NetBanking", "setup_netbanking")
        if entry:
            priority, label, action = entry
            actions.append((priority, {"label": label, "action": action, "url": url}))
    result, used = [], set()
    for _, item in sorted(actions, key=lambda x: -x[0]):
        if item["action"] not in used:
            used.add(item["action"])
            result.append(item)
        if len(result) >= max_ctas: break
    return result


def _replace_related_results(answer, urls):
    """Keep one clean sidebar section; never alter the generated answer body."""
    if not isinstance(answer, str):
        return answer
    body = re.split(r"(?im)^\s*Related search results\s*:\s*$", answer, maxsplit=1)[0].rstrip()
    if not urls or not body or body.lower().startswith((
        "an error occurred", "this service is temporarily unavailable",
        "i couldn't find relevant source content")):
        return body
    return body + "\n\nRelated search results:\n" + "\n".join(urls)

def metadata_chat(
    prompt,
    domain,
    conversation_id,
    loaded_session,
    faiss_index_dir,
    metadata_path,
    response_language="en",
    original_question=None
):
    route_started = time.perf_counter()
    _stage_data = _aura_stage_timings.get()
    metadata_stat = os.stat(metadata_path)
    records = _load_metadata_cached(
        metadata_path, metadata_stat.st_mtime_ns, metadata_stat.st_size
    )
    answer_instructions = (loaded_session.get("custom_prompt") or "") + "\n" + answer_language_instruction(response_language, original_question or prompt)
    answer_instructions += ("\nAU BANK EVIDENCE RULE: Answer only about AU Bank. "
        "For account-opening questions, distinguish an initial opening deposit "
        "from minimum balance and average monthly balance (AMB). "
        "If a retrieved AU Bank page states an AMB but not a one-time opening deposit, "
        "give the verified AMB with its exact account/branch qualification, "
        "and explicitly say the initial deposit is not specified. "
        "Never imply zero AMB proves zero initial deposit. "
        "Never invent financial amounts or cite irrelevant pages. "
        "If the user asks for DAILY transfer limits, do not substitute "
        "a temporary newly-added-beneficiary restriction for the daily limit. "
        "If the exact daily limit is not supported, state that it cannot be verified. ")

    answer_instructions += ("\nCONVERSATIONAL INTELLIGENCE: Be warm, engaging and context-aware. "
        "Acknowledge good news, setbacks or frustration naturally before addressing banking intent. "
        "Understand informal speech and short forms. Maintain relevant follow-up context. "
        "Be respectful with hostile language and politely ask for civility when appropriate. "
        "Do not fabricate product details or use emojis mechanically. "
        "Never let conversational tone replace verified evidence. ")

    # Payout mode is determined by the current user request, not by the
    # presence of other rate tables in the retrieved article.
    fd_rate_request = bool(re.search(
        r"\b(fd|fds|fixed\s+deposits?|term\s+deposits?)\b", prompt, re.I
    ) and re.search(r"\b(interest|rates?|yield|returns?)\b", prompt, re.I))
    monthly_requested = bool(re.search(
        r"\b(monthly|every\s+month|month[-\s]?wise)\b", prompt, re.I
    ))
    quarterly_requested = bool(re.search(
        r"\b(quarterly|every\s+quarter)\b", prompt, re.I
    ))
    compare_payouts = bool(re.search(
        r"\b(compare|comparison|difference|versus|vs\.?|both|all\s+payouts?)\b",
        prompt, re.I
    ))
    default_fd_payout = (fd_rate_request and not (
        monthly_requested or quarterly_requested or compare_payouts
    ))
    if default_fd_payout:
        answer_instructions += ("\nMANDATORY REGULAR FD RATE TABLE POLICY: "
            "The user has not requested monthly or quarterly payout rates. "
            "Use the standard/regular FD interest-rate table, with interest payable "
            "at maturity, as the default. For senior-citizen questions use the "
            "eligible senior-citizen column of that regular table. "
            "Do not display monthly-payout or quarterly-payout columns or "
            "their figures, and do not mix different payout tables. "
            "Do not substitute annualized yield for the quoted interest rate. "
            "Preserve verified tenures, rates, eligibility, and effective dates; "
            "if the requested table cannot be verified, say so rather than inventing it.")
    elif fd_rate_request and (monthly_requested or quarterly_requested):
        answer_instructions += ("\nFD PAYOUT TABLE POLICY: The user explicitly "
            "requested a payout frequency. Use that frequency's verified FD rate "
            "table. Do not substitute regular maturity rates for it unless the "
            "user explicitly asks for a comparison. ")

    state_key = (domain, conversation_id)

    with conversation_lock:
        state = dict(conversation_states.get(state_key, {}))

    route = route_question_llm(
        question=prompt, records=records, conversation_state=state,
        chat_model=loaded_session.get("selected_chat_model")
    )

    old_products = state.get("products") or ([state["product"]] if state.get("product") else [])
    new_products = route.get("products", [])
    if route["status"] in ("ready", "recommend") and old_products and set(old_products) != set(new_products):
        route["state"] = dict(route["state"])
        # Keep history for the router, but avoid injecting unrelated products
        # into the evidence-based final answer on a fresh topic.
        route["state"]["history"] = state.get("history", [])[-12:]
        print("[ROUTER] Products switched:", old_products, "->", new_products)
    print("[ROUTER] Status:",route["status"],"Products:",new_products)
    if _stage_data is not None:
        _stage_data["routing"] = time.perf_counter() - route_started
    _stage_clock = time.perf_counter()

    with conversation_lock:
        conversation_states[state_key] = route["state"]

    if route["status"] == "out_of_scope":
        language_instruction = ("Respond in the language and script of the user's message. "
                                "If Hinglish, use natural Hinglish.")
        return run_scope_reply(prompt, loaded_session.get("selected_chat_model"), language_instruction)

    if route["status"] == "general":
        from inference import inference_faiss
        model_name = loaded_session.get("embedding_model_name", "all-MiniLM-L6-v2")
        faiss_path = os.path.join(faiss_index_dir, "index.faiss")
        pickle_path = os.path.join(faiss_index_dir, "index.pkl")
        embeddings, vectorstore = _load_metadata_index(
            faiss_index_dir, model_name, os.stat(faiss_path).st_mtime_ns,
            os.stat(pickle_path).st_mtime_ns)
        answer = inference_faiss(
            chat_model=loaded_session.get("selected_chat_model"), question=prompt,
            embedding_model_global=embeddings, index=vectorstore.index,
            docstore=vectorstore.docstore,
            index_to_docstore_id=vectorstore.index_to_docstore_id,
            chat_history=route["state"].get("history", []),
            custom_instructions=answer_instructions,
            max_output_tokens=loaded_session.get("max_output_tokens", 1024),
            top_k=loaded_session.get("top_k", 8),
            temperature=loaded_session.get("temperature", 0.3))
        with conversation_lock:
            latest = conversation_states.get(state_key, {})
            latest.setdefault("history", [])
            latest["history"].extend([{"role":"user","content":prompt},
                                      {"role":"assistant","content":answer}])
            latest["history"] = latest["history"][-12:]
            conversation_states[state_key] = latest
        return answer

    if route["status"] == "recommend":
        from inference import inference_recommendation
        model_name = loaded_session.get("embedding_model_name", "all-MiniLM-L6-v2")
        faiss_path = os.path.join(faiss_index_dir, "index.faiss")
        pickle_path = os.path.join(faiss_index_dir, "index.pkl")
        embeddings, vectorstore = _load_metadata_index(
            faiss_index_dir, model_name,
            os.stat(faiss_path).st_mtime_ns, os.stat(pickle_path).st_mtime_ns)
        indexed = indexed_source_urls(vectorstore.docstore)
        candidates = recommendation_candidates(prompt, route["family"], records, indexed_urls=indexed)
        print("[RECOMMEND] Candidates:", [(x['url'], x['product']) for x in candidates[:12]])
        answer = inference_recommendation(
            chat_model=loaded_session.get("selected_chat_model"),
            question=prompt, candidates=candidates,
            embedding_model=embeddings, docstore=vectorstore.docstore,
            index=vectorstore.index,
            index_to_docstore_id=vectorstore.index_to_docstore_id,
            custom_instructions=answer_instructions,
            max_output_tokens=loaded_session.get("max_output_tokens", 1024),
            top_k=loaded_session.get("top_k", 8),
            temperature=loaded_session.get("temperature", 0.3))
        with conversation_lock:
            latest = conversation_states.get(state_key, {})
            latest.setdefault("history", [])
            latest["history"].extend([{"role":"user","content":prompt},
                                       {"role":"assistant","content":answer}])
            latest["history"] = latest["history"][-12:]
            conversation_states[state_key] = latest
        return answer

    if route["status"] != "ready":
        return route["response"]

    resolved_question = route["question"]
    # A product-only switch must keep the user's substantive question for
    # retrieval, while the response remains in the language of the current turn.
    # Keep original wording for response language; use English only for searching.
    search_question = (resolved_question if response_language == "en" else
        retrieval_query_in_english(resolved_question, loaded_session.get("selected_chat_model")))
    print("[MULTILINGUAL] Retrieval query:", search_question)
    from metadata_router import deduplicate_product_aliases
    products = deduplicate_product_aliases(
        route.get("products") or [route["product"]], distinct_products(records)
    )
    if len(products) != len(route.get("products") or [route["product"]]):
        print("[ROUTER] Deduplicated product aliases:", products)
    # Guard against a generic family replacing an explicitly named product.
    # Only resolve against actual metadata IDs, never invent a product.
    if len(products) == 1 and products[0] in {"loan", "account", "deposit", "card"}:
        from metadata_router import _explicitly_named_products, canonical_product
        explicit_ids = _explicitly_named_products(prompt, distinct_products(records))
        specific_ids = [p for p in distinct_products(records)
                        if canonical_product(p) in explicit_ids and p not in
                        {"loan", "account", "deposit", "card"}]
        if len(specific_ids) == 1:
            print("[ROUTER] Recovered explicit product:", products[0], "->", specific_ids[0])
            products = specific_ids


    model_name = loaded_session.get("embedding_model_name", "all-MiniLM-L6-v2")
    faiss_path = os.path.join(faiss_index_dir, "index.faiss")
    pickle_path = os.path.join(faiss_index_dir, "index.pkl")
    embeddings, vectorstore = _load_metadata_index(
        faiss_index_dir, model_name,
        os.stat(faiss_path).st_mtime_ns,
        os.stat(pickle_path).st_mtime_ns
    )
    available_urls = indexed_source_urls(vectorstore.docstore)

    # Specific, verified answer pages for a cross-product service question.
    # The metadata tags these pages under UPI/IMPS/NEFT/NetBanking rather
    # than a generic fund_transfer product. Normal product-only filtering
    # would therefore exclude the pages containing the actual limits.
    daily_limits_request = bool(re.search(
        r"\b(daily|per\s*day|day)\b", search_question, re.I
    ) and re.search(
        r"\b(fund\s*transfers?|money\s*transfers?|transfer\s*limits?)\b",
        search_question, re.I
    ) and re.search(r"\blimits?\b", search_question, re.I))
    daily_limit_sources = [
        "https://www.au.bank.in/personal-banking/digital-banking/payments/digital-transactions-limit",
        "https://www.au.bank.in/support/contact-us/retail/payments/fund-transfers/what-are-the-daily-limits-for-online-fund-transfers",
    ]
    indexed_daily_sources = [u for u in daily_limit_sources
                             if normalize_url(u) in {normalize_url(v) for v in available_urls}]
    if daily_limits_request and indexed_daily_sources:
        # Use a single source group; never combine with an unrelated
        # beneficiary setup article or a previous product's context.
        products = [products[0]] if products else ["fund_transfer"]
        selected_by_product = {products[0]: indexed_daily_sources}
        print("[ROUTER] Verified daily-limit source URLs:", indexed_daily_sources)
        for url in indexed_daily_sources:
            record = next((r for r in records if normalize_url(r["url"]) == normalize_url(url)), None)
            if record:
                print("[URL SUMMARY]", url, "|", record["summary"])
    else:
        selected_by_product = {}
    related_candidates_by_product = {}
    for product in ([] if daily_limits_request and indexed_daily_sources else products):
        candidates = rank_metadata_urls(
            question=search_question, product=product, records=records,
            embedding_model=embeddings, limit=12, indexed_urls=available_urls
        )
        if not candidates:
            # Metadata semantic ranking can miss aliases even when indexed
            # catalog records exist. Let the intent selector inspect those records.
            print("[ROUTER] No semantic candidates; checking indexed product aliases:", product)
        related_candidates_by_product[product] = candidates
        # Keep the question-aware ranking. Do not re-sort by category: a
        # documentation FAQ/blog can be more relevant than a generic main page.
        # Include up to 3 indexed sources for complementary evidence, without
        # introducing another LLM call or permitting unindexed URLs.
        urls = _select_intent_sources(search_question, product, records, available_urls, candidates, limit=4)
        print("[INTENT RANKING]", product, urls)
        # Calculator instructions need the calculator page, not unrelated FD variants.
        fd_intent = _fd_question_intent(search_question)
        if fd_intent["calculator"] and product == "fixed_deposit":
            calculator_url = next((
                r["url"] for r in records
                if "fd-calculator" in normalize_url(r["url"]).lower()
                and normalize_url(r["url"]) in available_urls
            ), None)
            if calculator_url:
                urls = [calculator_url]
                print("[FD CALCULATOR] Primary calculator source selected:", calculator_url)
        # For ordinary FD rate questions, prioritize the indexed primary
        # retail FD rate sheet; exclude Green/Bulk FD and generic landing pages
        # from answer evidence unless the user explicitly asks for a variant.
        if default_fd_payout and product == "fixed_deposit":
            primary_path = "/interest-rates/fixed-deposit-interest-rates"
            primary = next((
                r["url"] for r in records
                if normalize_url(r["url"]).lower().endswith(primary_path)
                and normalize_url(r["url"]) in available_urls
            ), None)
            if primary:
                urls = [primary]
                print("[FD POLICY] Primary regular FD rate sheet selected:", primary)
            else:
                # No invented URL or unsupported primary source.
                urls = [u for u in urls if not re.search(
                    r"green[-_/]?fd|green[-_/]?fixed|bulk[-_/]?fd|bulk[-_/]?fixed",
                    u, re.I
                )]
                print("[FD POLICY] Primary FD rate sheet not indexed; using ranked non-variant evidence")
        # Prefer the verified AU Savings Account overview for generic requests.
        # Do not allow a blog fallback to obscure an indexed official overview.
        if is_generic_savings_question(search_question) and product == "savings_account":
            overview = savings_overview_urls(records, available_urls)
            if overview:
                urls = overview[:1] + [u for u in urls if u not in overview][:2]
        if not urls:
            # A model rejection is NOT permission to cite an irrelevant page.
            print("[ROUTER] No sufficiently relevant URL selected:", product)
            continue
        selected_by_product[product] = urls
        print("[ROUTER] Product:", product, "Selected URLs:", urls)
        summary_by_url = {r["url"]: r["summary"] for r in records}
        for selected_url in urls:
            print("[URL SUMMARY]", selected_url, "|", summary_by_url.get(selected_url, "NOT FOUND"))

    if len(selected_by_product) != len(products):
        missing = [p for p in products if p not in selected_by_product]
        return "I couldn't find indexed source information for: " + ", ".join(missing)

    if _stage_data is not None:
        _stage_data["source_selection"] = time.perf_counter() - _stage_clock
    history = route["state"].get("history", [])
    if old_products and set(old_products) != set(products) and not state.get("pending_question"):
        history = []  # new topic: do not contaminate RAG answer with old topic
    if len(products) > 1:
        from inference import inference_multiple_products
        answer = inference_multiple_products(
            chat_model=loaded_session.get("selected_chat_model"),
            question=resolved_question, embedding_model=embeddings,
            docstore=vectorstore.docstore, selected_by_product=selected_by_product,
            index=vectorstore.index,
            index_to_docstore_id=vectorstore.index_to_docstore_id,
            chat_history=history, custom_instructions=answer_instructions,
            max_output_tokens=loaded_session.get("max_output_tokens", 1024),
            top_k=loaded_session.get("top_k", 8),
            temperature=loaded_session.get("temperature", 0.3)
        )
    else:
        answer = inference_selected_urls(
            chat_model=loaded_session.get("selected_chat_model"),
            question=resolved_question, retrieval_question=search_question,
            embedding_model=embeddings,
            docstore=vectorstore.docstore,
            selected_urls=selected_by_product[products[0]],
            index=vectorstore.index,
            index_to_docstore_id=vectorstore.index_to_docstore_id,
            chat_history=history, custom_instructions=answer_instructions,
            max_output_tokens=loaded_session.get("max_output_tokens", 1024),
            top_k=loaded_session.get("top_k", 8),
            temperature=loaded_session.get("temperature", 0.3)
        )

    # Issue 4: related discovery is independent of answer evidence. These
    # URLs are NOT passed to inference_selected_urls / the answer model.
    # Use the already ranked metadata candidates: no extra embedding/LLM call.
    related_urls = _related_metadata_urls(
        search_question, products, related_candidates_by_product,
        selected_by_product, available_urls, records, max_related=5
    )
    if "savings_account" in products and _intent_profile(search_question)["opening"]:
        overview = savings_overview_urls(records, available_urls)
        related_urls = list(dict.fromkeys(overview[:1] + related_urls))[:5]
    if "au_personal_loan" in products:
        primary = "https://www.au.bank.in/personal-banking/personal-loan"
        if normalize_url(primary) in {normalize_url(u) for u in available_urls}:
            related_urls = list(dict.fromkeys([primary] + related_urls))[:5]
    answer = _replace_related_results(answer, related_urls)
    print("[RELATED PAGES]", related_urls)

    with conversation_lock:
        latest = conversation_states.get(state_key, {})
        latest.setdefault("history", [])
        latest["history"].extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ])
        latest["history"] = latest["history"][-12:]
        conversation_states[state_key] = latest

    return answer


# Lightweight, metadata-grounded typeahead (no model call required).
@app.get("/suggestions")
def get_suggestions(q: str, project_name: str):
    import difflib
    q = q.strip().lower()[:160]
    if len(q) < 2 or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", project_name):
        return {"suggestions": []}
    path = os.path.join(BASE_OUTPUT_DIR, project_name, "metadata.json")
    if not os.path.isfile(path):
        return {"suggestions": []}
    try:
        records = load_metadata(path)
        candidates = []
        for record in records:
            for product in record.get("product", []):
                label = product.replace("_", " ").strip().title()
                if label:
                    candidates.append(f"Tell me about AU Bank {label}")
                    candidates.append(f"What are the eligibility criteria for {label}?")
        candidates = list(dict.fromkeys(candidates))
        tokens = q.split()
        scored = [(sum(2 for token in tokens if token in item.lower()) + difflib.SequenceMatcher(None,q,item.lower()).ratio(), item) for item in candidates]
        return {"suggestions": [item for score,item in sorted(scored,reverse=True) if score >= 0.65][:5]}
    except Exception:
        return {"suggestions": []}

# Fetch verified HTML meta descriptions from the same bank host as a related URL.
@lru_cache(maxsize=2048)
def _related_page_description(url: str):
    from urllib.parse import urlsplit
    from html.parser import HTMLParser
    import requests
    class MetaParser(HTMLParser):
        description = ""
        title = ""
        in_title = False
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "meta" and (a.get("name", "").lower() == "description" or a.get("property", "").lower() == "og:description") and not self.description:
                self.description = a.get("content", "").strip()
            if tag == "title": self.in_title = True
        def handle_endtag(self, tag):
            if tag == "title": self.in_title = False
        def handle_data(self, data):
            if self.in_title: self.title += data
    try:
        host = (urlsplit(url).hostname or "").lower()
        # No arbitrary hosts, private-network destinations or redirects.
        if not (host == "aubank.in" or host.endswith(".aubank.in") or host == "au.bank.in" or host.endswith(".au.bank.in")):
            return {"url":url,"title":"","description":""}
        response = requests.get(url,timeout=3,allow_redirects=False,headers={"User-Agent":"AUra-Metadata/1.0"})
        if response.status_code != 200 or "text/html" not in response.headers.get("content-type", ""):
            return {"url":url,"title":"","description":""}
        parser = MetaParser(); parser.feed(response.text[:180000])
        return {"url":url,"title":parser.title.strip()[:140],"description":parser.description[:350]}
    except Exception:
        return {"url":url,"title":"","description":""}

@app.post("/chat")
async def chat_with_bot(payload: ChatRequest, request: Request):
    _request_started = time.perf_counter()
    _stage_data = {}
    _stage_token = _aura_stage_timings.set(_stage_data)
    _console.print("\n[QUESTION] " + str(payload.query)[:500], flush=True)
    prompt = payload.query
    domain = payload.project_name
    def _record_chat_response(result):
        # All successful /chat response branches pass through this helper.
        save_chat_log(
            question=prompt,
            answer=result.get("response", ""),
            ip=request.client.host if request.client else "unknown",
            project_name=domain,
        )
        return result

    print(f"🧾 [CHAT] ➤ Domain: {domain}, Prompt: {prompt}")

    # 🌐 Script-first detection: English acronyms and short inputs stay English.
    detected_lang = detect_input_language(prompt)

    _console.print(f"[LANGUAGE] {detected_lang}", flush=True)

    domain_folder = os.path.join(BASE_OUTPUT_DIR, domain)
    session_file = os.path.join(domain_folder, "session_state.pkl")

    if not os.path.exists(session_file):
        print("⚠️ [CHAT] ➤ Session file not found.")
        raise HTTPException(status_code=400, detail="❌ Session not found. Please preprocess data first.")

    try:
        with open(session_file, "rb") as f:
            loaded_session = pickle.load(f)
        print("✅ [CHAT] ➤ Session loaded.")
    except Exception as e:
        print(f"❌ [CHAT] ➤ Failed to load session state: {e}")
        raise HTTPException(status_code=500, detail="❌ Failed to load session state.")

    # Specialized factual routes avoid the 3-5 serial LLM calls of generic RAG.
    if os.path.isfile(os.path.join(domain_folder, "metadata.json")) and (
            _is_lea_query(prompt) or _is_branch_count_query(prompt)):
        t0 = time.perf_counter()
        special = _source_grounded_special_answer(prompt, loaded_session, detected_lang,
                    os.path.join(domain_folder, "faiss_index"))
        print(f"[PERFORMANCE] Special-source answer: {time.perf_counter()-t0:.2f}s")
        return _record_chat_response({"response": clean_customer_text(finalize_answer(special, prompt, detected_lang)), "language": detected_lang,
                "conversation_id": payload.conversation_id or str(uuid.uuid4())})

    # Fast, context-aware conversational safety gate. This runs BEFORE scope routing.
    # It does not block a legitimate banking question containing an expletive.
    guarded = conversational_guardrail(prompt, detected_lang)
    if guarded is not None:
        return _record_chat_response({"response": guarded, "language": detected_lang,
                "conversation_id": payload.conversation_id or str(uuid.uuid4())})

    # Natural social exchanges bypass retrieval and its generic banking fallback.
    if is_casual_social(prompt):
        social_reply = conversational_social_reply(
            prompt, loaded_session.get("selected_chat_model"), detected_lang)
        if social_reply:
            return _record_chat_response({"response": strip_social_links(social_reply), "language": detected_lang,
                    "conversation_id": payload.conversation_id or str(uuid.uuid4())})

    # Social conversation is handled by the selected LLM, before RAG.
    # This intentionally does not depend on a hardcoded greeting list.
    # Only short messages need this extra classification call.
    if len(prompt.split()) <= 18 and not re.search(
            r"\b(bank|fd|fixed deposit|loan|account|branch|deposit|transfer|rate|interest|card|lea|police|ncrp|complaint|balance|kyc|atm|m[ -]?pin|pin|password|otp|reset|forgot|forgotten|upi|imps|neft|rtgs|netbanking)\b",
            prompt, re.I):
        greeting = validate_greeting(prompt) if not is_casual_social(prompt) else None
        if greeting:
            return _record_chat_response({"response": greeting, "language": detected_lang,
                    "conversation_id": payload.conversation_id or str(uuid.uuid4())})
        social_reply = conversational_social_reply(
            prompt, loaded_session.get("selected_chat_model"), detected_lang)
        if social_reply is not None:
            return _record_chat_response({
                "response": strip_social_links(social_reply),
                "language": detected_lang,
                "conversation_id": payload.conversation_id or str(uuid.uuid4())
            })

    metadata_path = os.path.join(
        domain_folder, "metadata.json"
    )

    if os.path.isfile(metadata_path):
        conversation_id = (
            payload.conversation_id or str(uuid.uuid4())
        )

        faiss_index_dir = os.path.join(
            domain_folder, "faiss_index"
        )

        try:
            _metadata_started = time.perf_counter()
            response = metadata_chat(
                prompt=prompt,
                domain=domain,
                conversation_id=conversation_id,
                loaded_session=loaded_session,
                faiss_index_dir=faiss_index_dir,
                metadata_path=metadata_path,
                response_language=detected_lang,
                original_question=prompt
            )

            _stage_data["metadata_pipeline"] = time.perf_counter() - _metadata_started
            _language_started = time.perf_counter()
            response = enforce_answer_language(
                response, detected_lang, prompt, loaded_session.get("selected_chat_model"))

            _stage_data["language_check"] = time.perf_counter() - _language_started
            # CTAs are separate from the response text, and are grounded in
            # the same verified Related Pages URLs the UI already displays.
            related_section = re.split(r"(?im)^\s*Related search results\s*:\s*$", response, maxsplit=1)
            related_urls = (re.findall(r"https?://[^\s<>]+", related_section[1])
                            if len(related_section) > 1 else [])
            ctas = _contextual_ctas(prompt, related_urls)
            _final_response = clean_customer_text(finalize_answer(response, prompt, detected_lang))
            _console.print("[TIMINGS] " + " | ".join(
                f"{key}={seconds:.3f}s" for key, seconds in _stage_data.items()
                if key != "metadata_pipeline"
            ) + f" | total={time.perf_counter() - _request_started:.3f}s", flush=True)
            _console.print("[RESPONSE] " + re.split(r"(?im)^\s*Related search results\s*:\s*$", _final_response, maxsplit=1)[0][:4000], flush=True)
            _aura_stage_timings.reset(_stage_token)
            return _record_chat_response({
                "response": _final_response,
                "language": detected_lang,
                "conversation_id": conversation_id,
                "ctas": ctas,
                "related_pages": [_related_page_description(url) for url in list(dict.fromkeys(related_urls))[:5]]
            })
        

        except Exception as e:
            _console.print("[ERROR] Metadata chat:", str(e), flush=True)
            raise HTTPException(
                status_code=500,
                detail="Metadata chat processing failed."
            )

    messages = loaded_session.get("messages", [])
    embedding_model = loaded_session.get("embedding_model_global", None)
    selected_vectordb = loaded_session.get("selected_vectordb", "FAISS")
    selected_chat_model = loaded_session.get("selected_chat_model", None)
    custom_prompt = (loaded_session.get("custom_prompt") or "") + "\n" + answer_language_instruction(detected_lang, prompt)
    top_k = loaded_session.get("top_k", 8)
    temperature = loaded_session.get("temperature", 0.3)

    faiss_index_dir = os.path.join(domain_folder, "faiss_index")
    retriever = await rebuild_faiss_retriever(faiss_index_dir)
    print("📦 [CHAT] ➤ Retriever rebuilt.")

    if not retriever:
        print("❌ [CHAT] ➤ Retriever not found.")
        raise HTTPException(status_code=500, detail="Retriever not available. Please preprocess data again.")

    try:
        docs = retriever.get_relevant_documents(prompt)
        print(f"🔍 [DEBUG] ➤ Retrieved {len(docs)} documents for query.")
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:200].replace('\n', ' ')
            print(f"📄 Doc {i+1}: {snippet}...")
    except Exception as e:
        print(f"❌ [DEBUG] ➤ Document retrieval failed: {e}")

    messages.append({"role": "user", "content": prompt})

    try:
        print("🧠 [INFERENCE] ➤ Calling inference engine...")
        response = inference(
            selected_vectordb,
            selected_chat_model,
            prompt,
            embedding_model,
            messages,
            custom_instructions=custom_prompt,
            faiss_index_dir=faiss_index_dir,
            max_output_tokens=loaded_session.get("max_output_tokens", 1024),
            top_k=top_k,
            temperature=temperature
        )
        print("✅ [INFERENCE] ➤ Response generated.")
        messages.append({"role": "assistant", "content": response})

        return _record_chat_response({
            "response": clean_customer_text(finalize_answer(response, prompt, detected_lang)),
            "language": detected_lang
        })

    except Exception as e:
        print(f"❌ [INFERENCE] ➤ Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Shared theme is stored per project, not in a browser.
THEME_FIELDS = (
    "primary", "secondary", "background", "surface", "input",
    "userBubble", "botBubble", "text", "mutedText", "link"
)


def _safe_project_folder(project_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,150}", project_name or "") or project_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid project name")
    return os.path.join(BASE_OUTPUT_DIR, project_name)


class ThemeUpdate(BaseModel):
    project_name: str
    theme: dict


@app.put("/project_theme")
async def update_project_theme(
    payload: ThemeUpdate,
    x_theme_admin_key: Optional[str] = Header(default=None),
):
    expected_key = os.getenv("THEME_ADMIN_KEY")
    if not expected_key:
        raise HTTPException(status_code=503, detail="THEME_ADMIN_KEY is not configured")
    if not x_theme_admin_key or not hmac.compare_digest(x_theme_admin_key, expected_key):
        raise HTTPException(status_code=403, detail="Unauthorized theme update")

    project_dir = _safe_project_folder(payload.project_name)
    if not os.path.isdir(project_dir):
        raise HTTPException(status_code=404, detail="Project does not exist")

    theme = payload.theme
    if set(theme) != set(THEME_FIELDS) or any(
        not isinstance(theme[key], str) or
        not re.fullmatch(r"#[0-9a-fA-F]{6}", theme[key])
        for key in THEME_FIELDS
    ):
        raise HTTPException(status_code=422, detail="Invalid theme colors")

    destination = os.path.join(project_dir, "theme.json")
    temporary = os.path.join(project_dir, f".theme-{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(theme, file)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"status": "success", "theme": theme}


@app.get("/get_config")
async def get_config(project_name: str = "local_upload"):
    import os
    import pickle

    domain_folder = _safe_project_folder(project_name)
    session_file = os.path.join(domain_folder, "session_state.pkl")

    if not os.path.exists(session_file):
        return JSONResponse(
            status_code=404,
            content={"error": "Config not found for this project_name"},
        )

    try:
        with open(session_file, "rb") as f:
            session_data = pickle.load(f)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to read session state: {str(e)}"},
        )

    saved_theme = None
    theme_path = os.path.join(domain_folder, "theme.json")
    if os.path.isfile(theme_path):
        try:
            with open(theme_path, "r", encoding="utf-8") as file:
                saved_theme = json.load(file)
        except (OSError, ValueError):
            saved_theme = None

    return {
        "theme": saved_theme,
        "chatbot_title": session_data.get("chatbot_title", "AI Chat Assistant"),
        "initial_message": session_data.get("initial_message", "Hello! 👋 How can I help you today?"),
        "custom_prompt": session_data.get("custom_prompt", ""),
        "chat_model": session_data.get("selected_chat_model", ""),
        "max_output_tokens": session_data.get("max_output_tokens", 1024),
        "top_k":session_data.get("top_k"),
        "temperature":session_data.get("temperature"),
        "quick_questions": session_data.get("quick_questions", [])  # NEW
    }


@app.post("/reset")
async def reset_chat():
    session_state.update({
        "retriever": None,
        "preprocessing_done": False,
        "index": None,
        "docstore": None,
        "embedding_model_global": None,
        "messages": []
    })
    print("🔄 [RESET] ➤ Chat session reset.")
    return {"message": "Session reset successfully."}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
