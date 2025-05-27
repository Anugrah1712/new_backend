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
    print("[Prompt Builder] Building prompt with:")
    print("- Context length:", len(context))
    print("- Chat history:", history)
    print("- Question:", question)
    print("- Current datetime:", current_datetime)
    print("- Custom instructions:", custom_instructions is not None)

    default_instructions = f"""

{context}

### CHAT HISTORY
{history}

### USER QUESTION
{question}

### SYSTEM INSTRUCTIONS

1. Do not hallucinate, infer, or fabricate any information. Answer only from the provided context.
2. Never greet more than once or repeat niceties. Avoid filler phrases.
3. You are a concise, helpful AI assistant. Respond clearly, briefly, and only using the given context.
4. For general or personal queries like "how are you?", reply naturally in one sentence only.
5. Do not include phrases like "Based on the provided text" or "According to the context."
6. If asked for the current date/time, respond using {current_datetime}. Do not mention time otherwise.
7. You are a multilingual assistant. Detect the language of the user query and respond in that language (e.g., Hindi, Hinglish, Gujarati, etc.).
8. Do not mention internal processes, model capabilities, or system design. If asked, reply with: "Please contact gptbot@ai."
9. If the question is off-topic or unrelated to the provided context, politely indicate that it is outside your scope.
10. When answering in a language (e.g., Hindi), do not mix with English unless the user uses Hinglish.
11. Do not generate or engage in responses involving hate, violence, illegal activity, or medical/legal advice.

"""
    if custom_instructions:
        full_prompt = default_instructions + "\n" + custom_instructions
    else:
        full_prompt = default_instructions

    print("[Prompt Builder] Final prompt constructed.")
    return full_prompt

# --- Greeting Detection ---
def validate_greeting(user_input):
    print("[Greeting Validator] Checking for greeting in input:", user_input)
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
    if user_input_lower in greetings or user_input_lower in ["hello", "hi", "hey", "greetings"]:
        response = f"{correct_greeting.capitalize()}. How can I help you?"
        print("[Greeting Validator] Matched greeting. Responding with:", response)
        return response
    print("[Greeting Validator] No greeting match found.")
    return None

# --- Time Utility ---
def get_current_datetime():
    ist = pytz.timezone("Asia/Kolkata")
    now_str = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    print("[Time Utility] Current datetime:", now_str)
    return now_str

# --- Unified Chat Model Handler ---
def run_chat_model(chat_model, context, question, chat_history, custom_instructions=None):
    print(f"[Model Handler] Running chat model: {chat_model}")
    current_datetime = get_current_datetime()
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    prompt = build_rag_prompt(context, history_context, question, current_datetime, custom_instructions)

    if "gemini" in chat_model.lower():
        print("[Model Handler] Using Gemini model...")
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
        print("[Gemini Response]", response)
        return response.text

    elif "gpt" in chat_model.lower():
        print("[Model Handler] Using OpenAI GPT model...")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        response = openai.ChatCompletion.create(
            model=chat_model,
            messages=messages,
            temperature=0.4,
        )
        print("[OpenAI GPT Response]", response["choices"][0]["message"]["content"])
        return response["choices"][0]["message"]["content"]

    else:
        print("[Model Handler] Using Together AI model...")
        model = ChatTogether(
            together_api_key="tgp_v1_8ogC_n1TfSj61WucxNlEKKmue3U2uLjKxlcA6WR-fBM",
            model=chat_model
        )
        output = model.predict(prompt)
        print("[Together AI Response]", output)
        return output

# --- FAISS Inference Only ---
def inference_faiss(chat_model, question, embedding_model_global, index, docstore, index_to_docstore_id, chat_history, custom_instructions=None):
    print("[FAISS] Performing FAISS search...")
    try:
        query_embedding = embedding_model_global.embed_query(question)
        k = 5
        D, I = index.search(np.array([query_embedding]), k=k)
        print(f"[FAISS] Top {k} indices: {I[0]}")

        contexts = []
        for faiss_idx in I[0]:
            if faiss_idx != -1:
                docstore_id = index_to_docstore_id.get(faiss_idx)
                if docstore_id:
                    doc = docstore.search(docstore_id)
                    if hasattr(doc, "page_content"):
                        contexts.append(doc.page_content)

        if not contexts:
            print("[FAISS] No documents found in retrieved indices.")
            return "No relevant context found in the documents."
        
        print("[FAISS] Retrieved documents:")
        for doc in contexts:
            print(" -", doc[:200], "...")  # Truncate to avoid log flooding

        context = "\n\n---\n\n".join(contexts)
        return run_chat_model(chat_model, context, question, chat_history, custom_instructions)

    except Exception as e:
        print(f"[FAISS ERROR] {str(e)}")
        return "An error occurred while processing your question."

# --- Dispatcher (Only FAISS retained) ---
def inference(vectordb_name, chat_model, question, embedding_model_global, chat_history, custom_instructions=None, faiss_index_dir=None):
    print(f"[Dispatcher] Routing to {vectordb_name} inference...")
    print(f" - Chat model: {chat_model}")
    print(f" - Question: {question}")
    print(f" - Custom instructions provided: {custom_instructions is not None}")

    custom_greeting_response = validate_greeting(question)
    if custom_greeting_response:
        return custom_greeting_response

    if vectordb_name == "FAISS":
        from langchain_community.vectorstores import FAISS

        if faiss_index_dir is None:
            print("[Dispatcher] ❌ FAISS index directory not provided.")
            return "❌ FAISS index directory not provided."

        faiss_index_path = os.path.join(faiss_index_dir, "index.faiss")
        if os.path.exists(faiss_index_path):
            import faiss
            import pickle

            index = faiss.read_index(faiss_index_path)
            with open(os.path.join(faiss_index_dir, "index.pkl"), "rb") as f:
                store_data = pickle.load(f)

            faiss_store = FAISS(
                embedding_function=embedding_model_global,
                index=index,
                docstore=store_data["docstore"],
                index_to_docstore_id=store_data["index_to_docstore_id"]
            )

            return inference_faiss(
                chat_model, question, embedding_model_global,
                faiss_store.index, faiss_store.docstore, faiss_store.index_to_docstore_id,
                chat_history, custom_instructions
            )
        else:
            return f"❌ FAISS index file not found at {faiss_index_path}. Please run preprocessing first."
    else:
        print("[Dispatcher] ❌ Invalid vector DB:", vectordb_name)
        return "❌ Invalid vector database selection! Only FAISS is supported."
