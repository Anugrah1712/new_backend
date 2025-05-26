# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Request
from typing import List
from preprocess import preprocess_vectordbs
from inference import inference
from webscrape import scrape_web_data
import validators
import uvicorn
import json
import asyncio
import os
import pickle
import tldextract
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Allow frontend CORS origins
origins = [
    "https://rag-chatbot-frontend-three.vercel.app",
    "http://13.60.34.232:8000",
    "https://rag-chatbot-web.shop",
    "http://rag-chatbot-web.shop",
    "http://localhost:3000",
    "https://datalysis-website.vercel.app",
    # "https://marketing.rag-chatbot-web.shop",
    "https://datalysis.rag-chatbot-web.shop",
    "https://gptbot-rosy.vercel.app",
    "https://anugrah-web.vercel.app",
    "https://kunjeshweb.vercel.app"
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

# ----------------------------- 🔧 Helper to Extract Domain ----------------------------- #
def extract_domain_from_request(request: Request):
    referer = request.headers.get("referer") or request.headers.get("origin")
    if referer:
        domain_info = tldextract.extract(referer)
        domain = f"{domain_info.subdomain + '.' if domain_info.subdomain else ''}{domain_info.domain}.{domain_info.suffix}"
        print(f"🔍 Domain from referer: {referer} → Parsed domain: {domain}")
        return domain
    return None

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
    doc_files: List[UploadFile] = File(...),
    links: str = Form(...),
    embedding_model: str = Form(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...)
):
    try:
        print("\n🛠️ [PREPROCESS] ➤ Started preprocessing request.")

        # Extract domain from the request origin (frontend domain)
        domain = extract_domain_from_request(request)
        if not domain:
            domain = "local_upload"

        domain_folder = os.path.join(BASE_OUTPUT_DIR, domain)
        os.makedirs(domain_folder, exist_ok=True)
        print(f"📁 [PREPROCESS] ➤ Created/using domain folder: {domain_folder}")

        links_list = json.loads(links)
        print(f"🌐 [PREPROCESS] ➤ Received {len(links_list)} links for scraping.")
        for link in links_list:
            print(f"🔗 Validating link: {link}")
            if not validators.url(link):
                raise HTTPException(status_code=400, detail=f"❌ Invalid URL: {link}")

        print(f"📎 [PREPROCESS] ➤ Uploaded {len(doc_files)} files:")
        for file in doc_files:
            print(f"   └── {file.filename}")
            if file.filename == "":
                raise HTTPException(status_code=400, detail="❌ One of the uploaded files is empty!")

        # Extract domain from links if any
        if links_list:
            domain_info = tldextract.extract(links_list[0])
            domain = f"{domain_info.subdomain + '.' if domain_info.subdomain else ''}{domain_info.domain}.{domain_info.suffix}"
            print(f"🌍 [PREPROCESS] ➤ Derived domain from links: {domain}")
            domain_folder = os.path.join(BASE_OUTPUT_DIR, domain)
            os.makedirs(domain_folder, exist_ok=True)

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

        print("🧠 [VECTORDB] ➤ Calling preprocess_vectordbs...")
        try:
            index, docstore, index_to_docstore_id, vectorstore, retriever, embedding_model_global, pinecone_index_name, vs, qdrant_client = await preprocess_vectordbs(
                doc_files, embedding_model, chunk_size, chunk_overlap, scraped_data, session_state["selected_vectordb"],
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
async def select_chat_model(chat_model: str = Form(...), custom_prompt: str = Form(None), request: Request = None):
    session_state["selected_chat_model"] = chat_model
    session_state["custom_prompt"] = custom_prompt
    print(f"✅ Chat model set: {chat_model}, Prompt: {custom_prompt}")

    # Extract domain from referer/origin header like in preprocess
    domain = None
    if request:
        domain = extract_domain_from_request(request)
    if not domain:
        domain = "local_upload"
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

    # Update loaded session with the new chat model and prompt
    loaded_session["selected_chat_model"] = chat_model
    loaded_session["custom_prompt"] = custom_prompt

    # Save back to file
    with open(session_file, "wb") as f:
        pickle.dump(loaded_session, f)

    print(f"💾 [STATE] ➤ Chat model saved in session state at: {session_file}")

    return {"message": f"Chat model set."}

# ---------------------------- 💬 CHAT Endpoint ---------------------------- #
class ChatRequest(BaseModel):
    query: str
    project_name: str

@app.post("/chat")
async def chat_with_bot(payload: ChatRequest, request: Request):
    print("\n💬 [CHAT] ➤ New chat query received.")
    prompt = payload.query
    domain = payload.project_name
    print(f"🧾 [CHAT] ➤ Domain: {domain}, Prompt: {prompt}")

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

    # ✅ REBUILD and USE the FAISS retriever
    faiss_index_dir = os.path.join(domain_folder, "faiss_index")
    retriever = await rebuild_faiss_retriever(faiss_index_dir)
    print("📦 [CHAT] ➤ Retriever rebuilt.")

    if not retriever:
        print("❌ [CHAT] ➤ Retriever not found.")
        raise HTTPException(status_code=500, detail="Retriever not available. Please preprocess data again.")

    # ✅ Debug: test retrieval BEFORE inference
    try:
        docs = retriever.get_relevant_documents(prompt)
        print(f"🔍 [DEBUG] ➤ Retrieved {len(docs)} documents for query.")
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:200].replace('\n', ' ')
            print(f"📄 Doc {i+1}: {snippet}...")
    except Exception as e:
        print(f"❌ [DEBUG] ➤ Document retrieval failed: {e}")

    # Append user message
    messages.append({"role": "user", "content": prompt})

    # 🔄 Inference call
    try:
        print("🧠 [INFERENCE] ➤ Calling inference engine...")
        response = inference(
            selected_vectordb,
            selected_chat_model,
            prompt,
            embedding_model,
            messages,
            custom_instructions=custom_prompt,
            faiss_index_dir=faiss_index_dir
        )
        print("✅ [INFERENCE] ➤ Response generated.")
        messages.append({"role": "assistant", "content": response})
        return {"response": response}
    except Exception as e:
        print(f"❌ [INFERENCE] ➤ Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

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
