# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Request
from typing import List, Optional
from preprocess import preprocess_vectordbs
from inference import inference
from webscrape import scrape_web_data
import uvicorn
import os, json, validators, pickle
import tldextract
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from datetime import datetime
import sqlite3

load_dotenv()

app = FastAPI()

# Initialize SQLite DB
# conn = sqlite3.connect('/data/chatlog.db', check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS chatlog (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     question TEXT,
#     answer TEXT,
#     ip TEXT,
#     project_name TEXT,
#     timestamp TEXT
# )
# ''')
# conn.commit()

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
    os.getenv("DOMAIN3")
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

        # ✅ Scrape web data if links provided
        scraped_data = []
        if links_list:
            try:
                print("🕸️ [SCRAPER] ➤ Starting web scraping...")
                for link in links_list:
                    scraped_data.extend(await scrape_web_data(link))
                    print(f"✅ [SCRAPER] ➤ Scraped content from: {link}")
                with open(os.path.join(domain_folder, "scraped_cache.pkl"), "wb") as f:
                    pickle.dump(scraped_data, f)
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
            state_to_save.pop("retriever", None)
            state_to_save.pop("index", None)
            state_to_save.pop("docstore", None)

            with open(os.path.join(domain_folder, "session_state.pkl"), "wb") as f:
                pickle.dump(state_to_save, f)

            print("💾 [STATE] ➤ Session state saved to:", domain_folder)
            return {"message": f"Preprocessing completed and saved in {domain_folder}"}

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
    temperature: float = Form(0.3)
):
    session_state["selected_chat_model"] = chat_model
    session_state["custom_prompt"] = custom_prompt
    session_state["max_output_tokens"] = max_output_tokens 
    session_state["initial_message"] = initial_message
    session_state["chatbot_title"] = chatbot_title
    session_state["top_k"] = top_k
    session_state["temperature"] = temperature


    print(f"✅ Value of k: {top_k} & Temperature: {temperature}")
    print(f"✅ Chat model set: {chat_model}, Prompt: {custom_prompt}, Max Tokens: {max_output_tokens}")
    print(f"📝 Title: {chatbot_title}, Initial Message: {initial_message}")

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

from fastapi import FastAPI, Request, HTTPException
from langdetect import detect
from datetime import datetime

@app.post("/chat")
async def chat_with_bot(payload: ChatRequest, request: Request):
    print("\n💬 [CHAT] ➤ New chat query received.")
    prompt = payload.query
    domain = payload.project_name
    print(f"🧾 [CHAT] ➤ Domain: {domain}, Prompt: {prompt}")

    # 🌐 Detect language
    try:
        detected_lang = detect(prompt)
        if detected_lang not in ['en', 'hi']:
            detected_lang = 'en'
    except Exception:
        print("⚠️ [LANGUAGE] ➤ Language detection failed. Defaulting to English.")
        detected_lang = 'en'

    print(f"🌐 [LANGUAGE] ➤ Detected language: {detected_lang}")

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

    messages = loaded_session.get("messages", [])
    embedding_model = loaded_session.get("embedding_model_global", None)
    selected_vectordb = loaded_session.get("selected_vectordb", "FAISS")
    selected_chat_model = loaded_session.get("selected_chat_model", None)
    custom_prompt = loaded_session.get("custom_prompt", None)
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

        # ✅ Log question + answer to SQLite
        try:
            user_ip = request.client.host
            cursor.execute('''
                INSERT INTO chatlog (question, ip, project_name, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (f"Q: {prompt} | A: {response}", user_ip, domain, datetime.utcnow().isoformat()))
            conn.commit()
            print("🗃️ [SQLITE] ➤ Chat log (with answer) inserted.")
        except Exception as e:
            print(f"⚠️ [SQLITE] ➤ Failed to log chat: {e}")

        return {
            "response": response,
            "language": detected_lang
        }

    except Exception as e:
        print(f"❌ [INFERENCE] ➤ Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/get_config")
async def get_config(project_name: str = "local_upload"):
    import os
    import pickle

    domain_folder = os.path.join(BASE_OUTPUT_DIR, project_name)
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

    return {
        "chatbot_title": session_data.get("chatbot_title", "AI Chat Assistant"),
        "initial_message": session_data.get("initial_message", "Hello! 👋 How can I help you today?"),
        "custom_prompt": session_data.get("custom_prompt", ""),
        "chat_model": session_data.get("selected_chat_model", ""),
        "max_output_tokens": session_data.get("max_output_tokens", 1024),
        "top_k":session_data.get("top_k"),
        "temperature":session_data.get("temperature")
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
