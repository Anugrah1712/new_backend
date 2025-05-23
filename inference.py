# inference.py
import os
import google.generativeai as genai
import openai
import pytz
import numpy as np
from datetime import datetime
from langchain_together import ChatTogether
from dotenv import load_dotenv

load_dotenv()

# --- API Configuration ---
genai.configure(api_key=("AIzaSyD364sF7FOZgaW4ktkIcITe_7miCqjhs4k"))
openai.api_key = ("OPENAI_API_KEY")

# --- Prompt Builder ---
def build_rag_prompt(context, history, question, current_datetime, custom_instructions=None):
    default_instructions = f"""

{context}

### CHAT HISTORY
{history}

### USER QUESTION
{question}

### SYSTEM INSTRUCTIONS

1. Do not hallucinate & do not repeat greetings after once  
2. You are a helpful AI assistant.Answer only using the exact content from the provided context & for general questions like "how are you?" repond naturally.
3. Do not mention : "Based on the provided text".
4. If the user asks for time or date, respond using {current_datetime}. Otherwise, do not mention the time.
5. Limit your answers to 50 words. Be factual and literal.
6. Never disclose technical details like your architecture or language. Politely decline and say: "Please contact gptbot@ai."
7. If the answer is not word-for-word in the context, respond with: "I am not sure about it."
"""
    if custom_instructions:
        full_prompt = default_instructions + "\n" + custom_instructions
    else:
        full_prompt = default_instructions
    return full_prompt

# --- Greeting Detection ---
def validate_greeting(user_input):
    user_input_lower = user_input.lower().strip()
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    hour = now.hour

    if 5 <= hour <= 11:
        correct_greeting = "good morning"
    elif 12 <= hour <= 16:
        correct_greeting = "good afternoon"
    elif 17 <= hour <= 20:
        correct_greeting = "good evening"
    else:
        correct_greeting = "good night"

    greetings = ["good morning", "good afternoon", "good evening", "good night"]
    if user_input_lower in greetings:
        return f"{correct_greeting.capitalize()}. How can I help you?"
    if user_input_lower in ["hello", "hi", "hey", "greetings"]:
        return f"{correct_greeting.capitalize()}. How can I help you?"
    return None

# --- Time Utility ---
def get_current_datetime():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# --- Unified Chat Model Handler ---
def run_chat_model(chat_model, context, question, chat_history, custom_instructions=None):
    current_datetime = get_current_datetime()
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    prompt = build_rag_prompt(context, history_context, question, current_datetime, custom_instructions)

    if "gemini" in chat_model.lower():
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(
            [prompt],
            generation_config={"temperature": 0.2},
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE"
            }
        )
        print("Gemini Respone :",response)
        return response.text


    elif "gpt" in chat_model.lower():
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        response = openai.ChatCompletion.create(
            model=chat_model,
            messages=messages,
            temperature=0.4,
        )
        return response["choices"][0]["message"]["content"]

    else:
        model = ChatTogether(
            together_api_key="tgp_v1_l11DnqQV2U4ZhRlsIrcok2tRTI2Kx_o7hwnqaXkF_Ks",
            model=chat_model
        )
        return model.predict(prompt)

# --- FAISS Inference Only ---
def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history, custom_instructions=None):
    print("[FAISS] Performing FAISS search...")
    try:
        query_embedding = embedding_model_global.embed_query(question)
        k = 5
        D, I = index.search(np.array([query_embedding]), k=k)

        contexts = []
        for i, idx in enumerate(I[0]):
            if idx != -1:
                doc = docstore.search(idx)
                if hasattr(doc, "page_content"):
                    contexts.append(doc.page_content)

        if not contexts:
            return "No relevant context found in the documents."

        context = "\n\n---\n\n".join(contexts)
        return run_chat_model(chat_model, context, question, chat_history, custom_instructions)

    except Exception as e:
        print(f"Error during FAISS inference: {str(e)}")
        return "An error occurred while processing your question."

# --- Dispatcher (Only FAISS retained) ---
def inference(vectordb_name, chat_model, question, embedding_model_global, chat_history, custom_instructions=None, faiss_index_dir=None):
    print(f"[Dispatcher] Routing to {vectordb_name} inference...")

    custom_greeting_response = validate_greeting(question)
    if custom_greeting_response:
        return custom_greeting_response

    if vectordb_name == "FAISS":
        from langchain.vectorstores import FAISS

        if faiss_index_dir is None:
            return "❌ FAISS index directory not provided."
        print("🔍 Looking for FAISS index in:", faiss_index_dir)
        try:
            print("📂 Contents:", os.listdir(faiss_index_dir))
        except Exception as e:
            print("⚠️ Error reading directory:", e)

        faiss_index_path = os.path.join(faiss_index_dir, "index.faiss")
        if os.path.exists(faiss_index_path):
            faiss_store = FAISS.load_local(faiss_index_dir, embedding_model_global, allow_dangerous_deserialization=True)
            return inference_faiss(
                chat_model, question, embedding_model_global,
                faiss_store.index, faiss_store.docstore, chat_history, custom_instructions
            )
        else:
            return f"❌ FAISS index file not found at {faiss_index_path}. Please run preprocessing first."
    else:
        return "❌ Invalid vector database selection! Only FAISS is supported."
