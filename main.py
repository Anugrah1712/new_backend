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
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Allow frontend CORS origins
origins = [
    "https://rag-chatbot-frontend-three.vercel.app",
    "https://rag-chatbot-frontend-anugrah-mishra-s-projects.vercel.app",
    "https://rag-chatbot-frontend-git-main-anugrah-mishra-s-projects.vercel.app",
    "http://13.60.34.232:8000",
    "https://rag-chatbot-web.shop",
    "http://rag-chatbot-web.shop",
    "http://localhost:3000",
    "https://datalysis-website.vercel.app",
    "https://marketing.rag-chatbot-web.shop",
    "https://datalysis.rag-chatbot-web.shop",
    "http://18.205.19.63:8000",
    "https://gptbot-rosy.vercel.app"
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
    "selected_chat_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
async def rebuild_faiss_retriever(index_path: str):
    embeddings = HuggingFaceEmbeddings()
    abs_index_path = os.path.abspath(index_path)
    print("📌 Absolute FAISS index path:", abs_index_path)
    vectorstore = FAISS.load_local(abs_index_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever, vectorstore.index, vectorstore.docstore, vectorstore

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
        print("\n🔍 Preprocessing Started...")

        # Extract domain from the request origin (frontend domain)
        domain = extract_domain_from_request(request)
        if not domain:
            domain = "local_upload"

        domain_folder = os.path.join(BASE_OUTPUT_DIR, domain)
        os.makedirs(domain_folder, exist_ok=True)

        links_list = json.loads(links)
        for link in links_list:
            if not validators.url(link):
                raise HTTPException(status_code=400, detail=f"❌ Invalid URL: {link}")

        if not doc_files and not links_list:
            raise HTTPException(status_code=400, detail="❌ No documents or links provided for preprocessing!")

        for file in doc_files:
            if file.filename == "":
                raise HTTPException(status_code=400, detail="❌ One of the uploaded files is empty!")

        # Extract domain
        domain = "local_upload"
        if links_list:
            domain_info = tldextract.extract(links_list[0])
            domain = f"{domain_info.subdomain + '.' if domain_info.subdomain else ''}{domain_info.domain}.{domain_info.suffix}"

        domain_folder = os.path.join(BASE_OUTPUT_DIR, domain)
        os.makedirs(domain_folder, exist_ok=True)

        scraped_data = []
        if links_list:
            try:
                print("🌐 Scraping web data...")
                for link in links_list:
                    scraped_data.extend(await scrape_web_data(link))
                with open(os.path.join(domain_folder, "scraped_cache.pkl"), "wb") as f:
                    pickle.dump(scraped_data, f)
                print("✅ Scraped data saved to:", domain_folder)
            except Exception as e:
                print(f"❌ Web scraping failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Web scraping failed: {str(e)}")

        try:
            index, docstore, index_to_docstore_id, vectorstore, retriever, embedding_model_global, pinecone_index_name, vs, qdrant_client = await preprocess_vectordbs(
                doc_files, embedding_model, chunk_size, chunk_overlap, scraped_data, session_state["selected_vectordb"],
                persist_directory=os.path.join(domain_folder, "faiss_index")
            )

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

            print("💾 Session state saved to:", domain_folder)
            return {"message": f"Preprocessing completed and saved in {domain_folder}"}
        except Exception as e:
            print(f"❌ Error in preprocess_vectordbs: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

@app.post("/select_vectordb")
async def select_vectordb(vectordb: str = Form(...)):
    session_state["selected_vectordb"] = vectordb
    print(f"✅ Selected Vector DB: {vectordb}")
    return {"message": f"Vector DB set to: {vectordb}"}

@app.post("/select_chat_model")
async def select_chat_model(chat_model: str = Form(...), custom_prompt: str = Form(None)):
    session_state["selected_chat_model"] = chat_model
    session_state["custom_prompt"] = custom_prompt
    print(f"✅ Chat model set: {chat_model}, Prompt: {custom_prompt}")
    return {"message": f"Chat model set."}

class ChatRequest(BaseModel):
    prompt: str

# ---------------------------- 💬 CHAT Endpoint ---------------------------- #
@app.post("/chat")
async def chat_with_bot(request: Request, prompt: str = Form(...), custom_prompt: str = Form(None)):
    domain = extract_domain_from_request(request)
    if not domain:
        raise HTTPException(status_code=400, detail="❌ Cannot determine domain from request headers.")

    domain_folder = os.path.join(BASE_OUTPUT_DIR, domain)
    session_file = os.path.join(domain_folder, "session_state.pkl")

    if not os.path.exists(session_file):
        raise HTTPException(status_code=400, detail="❌ Session not found. Please preprocess data first.")

    with open(session_file, "rb") as f:
        loaded_session = pickle.load(f)

    vector_db_dir = os.path.join(domain_folder, "faiss_index")
    retriever, index, docstore, vs = await rebuild_faiss_retriever(vector_db_dir)

    messages = loaded_session.get("messages", [])
    embedding_model = loaded_session.get("embedding_model_global", None)
    selected_vectordb = loaded_session.get("selected_vectordb", "FAISS")
    selected_chat_model = loaded_session.get("selected_chat_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

    messages.append({"role": "user", "content": prompt})
    if not custom_prompt:
        custom_prompt = loaded_session.get("custom_prompt", None)

    try:
        faiss_index_dir = os.path.join(BASE_OUTPUT_DIR, domain, "faiss_index")
        response = inference(
            selected_vectordb,
            selected_chat_model,
            prompt,
            embedding_model,
            messages,
            custom_instructions=custom_prompt,
            faiss_index_dir=faiss_index_dir
        )
        messages.append({"role": "assistant", "content": response})
        return {"response": response}
    except Exception as e:
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
    print("🔄 Chat session reset.")
    return {"message": "Session reset successfully."}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
