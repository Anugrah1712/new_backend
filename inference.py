# inference.py
import os
import google.generativeai as genai
import openai
import pytz
import numpy as np
from datetime import datetime
from langchain_together import ChatTogether
from dotenv import load_dotenv
from groq import Groq

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

You are a concise, multilingual AI assistant that responds strictly using the given context. Follow these instructions:

1. Do not hallucinate, infer, or fabricate any information. Only answer using the provided context.
2. Never greet more than once or repeat niceties. Avoid all filler phrases.
3. Respond clearly and briefly. Do not say “Based on the provided text,” “According to the context,” or similar.
4. For general or personal queries like “how are you?”, respond naturally in one short sentence only.
5. If asked for the current date/time, respond strictly with: {current_datetime} (ISO format: YYYY-MM-DDTHH:MM:SSZ).
6. If asked questions like "what is my job experience?", extract the starting year from the context and subtract it from the current year derived from {current_datetime}. Return the number of years as the experience.
7. Detect the user’s language (e.g., Hindi, Hinglish, Gujarati) and reply in that language. Do not mix with English unless the user uses Hinglish.
8. If the query is off-topic or unrelated to the given context, respond: "Sorry, I can only answer questions based on the provided content."
9. Do not mention internal processes, model capabilities, or system details. If asked, respond: “Please contact gptbot@ai.”

Stay concise. Use only the context. No extra explanations.


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
    else :
        correct_greeting = "good evening"

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
            generation_config={"temperature": 0.2 , "max_output_tokens": 1024},
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
            max_tokens=1024
        )
        print("[OpenAI GPT Response]", response["choices"][0]["message"]["content"])
        return response["choices"][0]["message"]["content"]

    elif chat_model in ["llama3-8b-8192", "llama3-70b-8192"]:
        print("[Model Handler] Using Groq model...")
        client = Groq(api_key="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")  
        response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.4,
            max_tokens=1024
        )
        print("[Groq Response]", response.choices[0].message.content)
        return response.choices[0].message.content

    else:
        print("[Model Handler] Using Together AI model...")
        model = ChatTogether(
            together_api_key="tgp_v1_8ogC_n1TfSj61WucxNlEKKmue3U2uLjKxlcA6WR-fBM",
            model=chat_model
        )
        output = model.predict(prompt, max_tokens=1024)
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
