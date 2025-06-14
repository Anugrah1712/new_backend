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
import random

load_dotenv()

# --- API Configuration ---
genai.configure(api_key=("AIzaSyD364sF7FOZgaW4ktkIcITe_7miCqjhs4k"))
openai.api_key = ("OPENAI_API_KEY")

# --- Prompt Builder ---
def build_rag_prompt(context, history, question, current_datetime, custom_instructions=None, max_output_tokens=None):
    print("[Prompt Builder] Building prompt with:")
    print("- Context length:", len(context))
    print("- Chat history:", history)
    print("- Question:", question)
    print("- Current datetime:", current_datetime)
    print("- Custom instructions:", custom_instructions is not None)

    # ⚠️ Remove duplicate question if it's the last in history
    if history.strip().endswith(f"User: {question.strip()}"):
        history = "\n".join(history.strip().split("\n")[:-1])

    combined_context = f"""Below is a conversation and relevant information.

### CHAT HISTORY
{history}

### USER QUESTION
{question}

### RETRIEVED DOCUMENT CONTEXT
{context}
"""

    default_instructions = f"""
{combined_context}

### SYSTEM INSTRUCTIONS

You are a concise, reliable AI assistant designed to provide accurate and relevant answers strictly based on the information available in the chat history and uploaded documents. Follow these instructions carefully:

1. Do not repeat or rephrase the user’s question under any circumstance.
2. Use only the provided context (chat history + documents) to generate your response. If the information is not present, do not attempt to guess or fabricate an answer.
3. When relevant, incorporate precise details from the context. Always remain grounded in the available data.
4. Avoid greetings, pleasantries, or filler statements. Never greet more than once in a session.
5. Keep all responses direct, clear, and as concise as possible. Prioritize clarity and factual correctness over verbosity.
6. The current date and time is: {current_datetime}. Use this when interpreting time-related queries.
7. If the user asks “What is my job experience?”, calculate the experience duration using the starting year mentioned in the chat or document context.
8. If the query is not related to the provided information, respond with:  
   **"Sorry, I can only answer based on the provided content."**
9. Do not disclose, speculate on, or discuss internal model functionality, training data, or behavior.
10. Do not engage in small talk, hypothetical scenarios, or personal opinions unless explicitly stated in the source content.
11. Maintain a professional, neutral tone at all times. Avoid emotive or opinionated language.
12. Ensure consistency across responses—use the same terminology and phrasing as found in the context when available.
13. If multiple interpretations are possible, prefer the most directly supported one from the documents.
14. Do not include disclaimers, unless explicitly instructed to do so by the context.

Your goal is to assist by delivering accurate, efficient, and contextually relevant answers with zero deviation from the source material. Stay focused and disciplined in your responses.
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
        response = f"Hey! {correct_greeting.capitalize()}. How can I help you?"
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
def run_chat_model(chat_model, context, question, chat_history, custom_instructions=None, max_output_tokens=1024, temperature=0.3):
    print(f"[Model Handler] Running chat model: {chat_model}")
    print(f"[Model Handler] max_output_tokens received: {max_output_tokens}")

    # Guard clause for missing chat_model
    if not chat_model or not isinstance(chat_model, str):
        raise ValueError("[Model Handler] ❌ 'chat_model' is None or invalid. Please provide a valid model name.")

    current_datetime = get_current_datetime()
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    prompt = build_rag_prompt(context, history_context, question, current_datetime, custom_instructions, max_output_tokens=max_output_tokens)

    try:
        chat_model_lower = chat_model.lower()

        if "gemini" in chat_model_lower:
            print("[Model Handler] Using Gemini model...")
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(
                [prompt],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens
                },
                safety_settings={
                    "HARASSMENT": "BLOCK_NONE",
                    "HATE": "BLOCK_NONE",
                    "SEXUAL": "BLOCK_NONE",
                    "DANGEROUS": "BLOCK_NONE"
                }
            )
            print("[Gemini Response]", response)
            return response.text

        elif "gpt" in chat_model_lower:
            print("[Model Handler] Using OpenAI GPT model...")
            messages = [
                {"role": "system", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
                model=chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens
            )
            print("[OpenAI GPT Response]", response["choices"][0]["message"]["content"])
            return response["choices"][0]["message"]["content"]

        elif chat_model_lower in ["llama3-8b-8192", "llama3-70b-8192"]:
            print("[Model Handler] Using Groq model with randomized API key rotation...")
            groq_keys = [
                "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",
                "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",
                "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",
                "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",
                "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",
                "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"
            ]

            # Shuffle keys before trying
            random.shuffle(groq_keys)

            for key in groq_keys:
                try:
                    client = Groq(api_key=key)
                    response = client.chat.completions.create(
                        model=chat_model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": question}
                        ],
                        temperature=temperature,
                        max_tokens=max_output_tokens
                    )
                    print(f"[Groq Response with key ending {key[-4:]}] {response.choices[0].message.content}")
                    return response.choices[0].message.content
                
                except Exception as e:
                    print(f"[Groq API Key {key[-4:]} Failed] ➤ {e}")
                    error_message = str(e).lower()
                    if not any(k in error_message for k in ["quota", "exceeded", "limit", "invalid api key", "permission", "unauthorized"]):
                        return f"An error occurred while generating response: {str(e)}"

            return "This service is temporarily unavailable due to exhausted API usage."


        else:
            print("[Model Handler] Using Together AI model...")
            model = ChatTogether(
                together_api_key="94d32cd3eedfd9911a6b1c281bc14d278cd0e4f3e52272b3f7cbbed13e698511",
                model=chat_model
            )
            output = model.predict(prompt, max_tokens=max_output_tokens,temperature=temperature)
            print("[Together AI Response]", output)
            print("Temperature ->>>>>>>>>>>>>>>>>", temperature)
            return output

    except Exception as e:
        print(f"[Model Handler Error] {e}")
        error_message = str(e).lower()
        if any(keyword in error_message for keyword in ["quota", "exceeded", "limit", "exhausted", "invalid api key", "permission denied", "unauthorized"]):
            return "This service is temporarily unavailable due to exhausted API usage."
        return f"An error occurred while generating response: {str(e)}"
    
# --- FAISS Inference Only ---
def inference_faiss(chat_model, question, embedding_model_global, index, docstore, index_to_docstore_id, chat_history, custom_instructions=None, max_output_tokens=1024, top_k=3, temperature=0.3):
    print("[FAISS] Performing FAISS search...")
    try:
        query_embedding = embedding_model_global.embed_query(question)
        k = top_k
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
        return run_chat_model(chat_model, context, question, chat_history, custom_instructions, max_output_tokens=max_output_tokens,temperature=temperature)
    except Exception as e:
        print(f"[FAISS ERROR] {str(e)}")
        return "An error occurred while processing your question."
    
# --- Simplified function for Optuna tuning ---
def get_response(question, top_k=3, temperature=0.3, max_output_tokens=1024, custom_docs=None):
    print("[get_response] Called from Optuna tuning")
    if not custom_docs:
        return "❌ No documents provided."

    context = "\n\n---\n\n".join([doc.page_content for doc in custom_docs if hasattr(doc, "page_content")])
    return run_chat_model(
        chat_model="llama3-8b-8192", 
        context=context,
        question=question,
        chat_history=[],
        custom_instructions=None,
        max_output_tokens=max_output_tokens
    )


# --- Dispatcher (Only FAISS retained) ---
def inference(vectordb_name, chat_model, question, embedding_model_global, chat_history, custom_instructions=None, faiss_index_dir=None, max_output_tokens=1024, top_k=8, temperature=0.3):
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
                chat_history, custom_instructions , max_output_tokens=max_output_tokens,top_k=top_k,temperature=temperature
            )
        else:
            return f"❌ FAISS index file not found at {faiss_index_path}. Please run preprocessing first."
    else:
        print("[Dispatcher] ❌ Invalid vector DB:", vectordb_name)
        return "❌ Invalid vector database selection! Only FAISS is supported."
