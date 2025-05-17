#inference.py
import os
import google.generativeai as genai
import openai
import pytz
from datetime import datetime
from langchain_together import ChatTogether
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- API Configuration ---
genai.configure(api_key = ("AIzaSyD364sF7FOZgaW4ktkIcITe_7miCqjhs4k"))
openai.api_key = ("OPENAI_API_KEY")

# --- Prompt Builder ---
def build_rag_prompt(context, history, question, current_datetime, custom_instructions=None):
    # print("[Debug] Received custom_instructions:", custom_instructions)
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

    # If custom instructions are provided, concatenate them to the default instructions
    if custom_instructions:
        print("[Prompt] Custom instructions detected:")
        # print(custom_instructions)
        full_prompt = default_instructions + "\n" + custom_instructions
    else:
        print("[Prompt] No custom instructions provided.")
        full_prompt = default_instructions

    print("[Prompt] Final RAG prompt built:")
    # print(full_prompt)  # print only first 1000 characters for brevity
    return full_prompt

import pytz

def validate_greeting(user_input):
    user_input_lower = user_input.lower().strip()
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    hour = now.hour

    # Determine correct greeting based on IST time
    if 5 <= hour <= 11:
        correct_greeting = "good morning"
    elif 12 <= hour <= 16:
        correct_greeting = "good afternoon"
    elif 17 <= hour <= 20:
        correct_greeting = "good evening"
    else:
        correct_greeting = "good night"

    # Recognized greetings
    greetings = ["good morning", "good afternoon", "good evening", "good night"]

    if user_input_lower in greetings:
        if user_input_lower == correct_greeting:
            return f"{correct_greeting.capitalize()}. How can I help you?"
        else:
            return f"{correct_greeting.capitalize()}. How can I help you?"

    # Informal greetings like "hello"
    informal = ["hello", "hi", "hey", "greetings"]
    if user_input_lower in informal:
        return f"{correct_greeting.capitalize()}. How can I help you?"

    return None

# --- Time Utility ---
def get_current_datetime():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[Time] Current datetime: {now}")
    return now

# --- Unified Chat Model Handler ---
def run_chat_model(chat_model, context, question, chat_history, custom_instructions=None):
    current_datetime = get_current_datetime()
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    
    # Final prompt with default + custom instructions
    prompt = build_rag_prompt(context, history_context, question, current_datetime, custom_instructions)

    if "gemini" in chat_model.lower():
        # Gemini expects a single prompt
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(
            [prompt],  # make prompt include SYSTEM INSTRUCTIONS at bottom
            generation_config={"temperature": 0.2},
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE"
            }
        )
        return response.text

    elif "gpt" in chat_model.lower():
        # GPT expects messages - so wrap full prompt in a single system message
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
        # Together also uses prompt as string
        model = ChatTogether(
            together_api_key="tgp_v1_l11DnqQV2U4ZhRlsIrcok2tRTI2Kx_o7hwnqaXkF_Ks",
            model=chat_model
        )
        response = model.predict(prompt)
        return response



# --- FAISS Inference ---
def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history, custom_instructions=None):
    print("[FAISS] Performing FAISS search...")
    try:
        query_embedding = embedding_model_global.embed_query(question)
        k = 3
        D, I = index.search(np.array([query_embedding]), k=k)

        contexts = []
        for i, idx in enumerate(I[0]):
            if idx != -1:
                doc = docstore.search(idx)
                if hasattr(doc, "page_content"):
                    contexts.append(doc.page_content)

        if not contexts:
            return "No relevant context found in the documents."

        # Print the documents retrieved by FAISS for debugging purposes
        print("[FAISS] Retrieved documents:")
        for doc in contexts:
            print(doc)

        context = "\n\n---\n\n".join(contexts)
        return run_chat_model(chat_model, context, question, chat_history, custom_instructions)

    except Exception as e:
        print(f"Error during FAISS inference: {str(e)}")
        return "An error occurred while processing your question."


# --- Chroma Inference ---
def inference_chroma(chat_model, question, retriever, chat_history,custom_instructions=None):
    print("[Chroma] Retrieving documents...")
    docs = retriever.get_relevant_documents(question)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs]) if docs else ""
    return run_chat_model(chat_model, context, question, chat_history , custom_instructions)

# --- Weaviate Inference ---
def inference_weaviate(chat_model, question, vs, chat_history,custom_instructions=None):
    print("[Weaviate] Retrieving documents...")
    retriever = vs.as_retriever()
    docs = retriever.get_relevant_documents(question)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs]) if docs else ""
    return run_chat_model(chat_model, context, question, chat_history ,custom_instructions)

# --- Pinecone Inference ---
def inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history,custom_instructions=None):
    print("[Pinecone] Querying Pinecone index...")
    query_embedding = embedding_model_global.embed_query(question)
    results = pinecone_index.query(vector=query_embedding, top_k=4, include_metadata=True)
    context = "\n\n---\n\n".join([match['metadata']['text'] for match in results['matches']])
    return run_chat_model(chat_model, context, question, chat_history, custom_instructions)

# --- Qdrant Inference ---
def inference_qdrant(chat_model, question, embedding_model_global, qdrant_client, chat_history,custom_instructions=None):
    print("[Qdrant] Querying Qdrant collection...")
    query_embedding = embedding_model_global.embed_query(question)
    results = qdrant_client.search(collection_name="text_vectors", query_vector=query_embedding, limit=4)
    context = "\n\n---\n\n".join([hit.payload['page_content'] for hit in results])
    return run_chat_model(chat_model, context, question, chat_history,custom_instructions)

# --- General Inference Dispatcher ---
def inference(vectordb_name, chat_model, question, embedding_model_global, chat_history, pinecone_index_name=None, vs=None, qdrant_client=None ,custom_instructions=None):
    print(f"[Dispatcher] Routing to {vectordb_name} inference...")

        # --- Greeting validation shortcut ---
    custom_greeting_response = validate_greeting(question)
    if custom_greeting_response:
        return custom_greeting_response  # Bypass Gemini/GPT

    if vectordb_name == "Chroma":
        from langchain.vectorstores import Chroma
        retriever = Chroma(persist_directory='db', embedding_function=embedding_model_global).as_retriever()
        return inference_chroma(chat_model, question, retriever, chat_history ,custom_instructions=custom_instructions)

    elif vectordb_name == "FAISS":
        from langchain.vectorstores import FAISS
        faiss_index_path = "faiss_index/index.faiss"
        if os.path.exists(faiss_index_path):
            faiss_store = FAISS.load_local("faiss_index", embedding_model_global, allow_dangerous_deserialization=True)
            return inference_faiss(chat_model, question, embedding_model_global, faiss_store.index, faiss_store.docstore, chat_history,custom_instructions=custom_instructions)
        else:
            return "❌ FAISS index file not found. Please run preprocessing first."

    elif vectordb_name == "Pinecone":
        from pinecone import Pinecone
        if pinecone_index_name:
            pinecone = Pinecone(api_key="pcsk_42Yw14_EaKdaMLiAJfWub3s2sEJYPW3jyXXjdCYkH8Mh8rD8wWJ3pS6oCCC9PGqBNuDTuf", environment=os.getenv("PINECONE_ENV", "us-east-1"))
            pinecone_index = pinecone.Index(pinecone_index_name)
            return inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history,custom_instructions=custom_instructions
)
        else:
            return "❌ Pinecone index not found. Please run preprocessing first."

    elif vectordb_name == "Weaviate":
        if vs:
            return inference_weaviate(chat_model, question, vs, chat_history,custom_instructions=custom_instructions

)
        else:
            return "❌ Weaviate vector store not found. Please run preprocessing first."

    elif vectordb_name == "Qdrant":
        if qdrant_client:
            return inference_qdrant(chat_model, question, embedding_model_global, qdrant_client, chat_history,custom_instructions=custom_instructions

)
        else:
            return "❌ Qdrant vector store not found. Please run preprocessing first."

    else:
        return "❌ Invalid vector database selection!"
