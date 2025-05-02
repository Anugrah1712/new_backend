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
genai.configure(api_key = ("AIzaSyBe-eQo1uquGgPRsolgHTKsnJEBwfqyUhg"))
openai.api_key = ("OPENAI_API_KEY")


# --- Prompt Builder ---
def build_rag_prompt(context, history, question, current_datetime, custom_instructions=None):
    # Default instructions for greeting and preventing hallucinations
    print("[Debug] Received custom_instructions:", custom_instructions)
    default_instructions = f"""[Current Date and Time: {current_datetime}]

    Context: {context}

    Chat History: {history}

    Question: {question}

    Answer to general conversation texts like hello, bye, etc.

    *Strict Instructions to Avoid Hallucination:*
    0. Do not mention the current date or time unless:
        - The user explicitly asks for the time/date.
        
    *Greeting Handling Instructions:*
        - Extract current hour from: "{current_datetime}"
        - Based on the current time (24-hour format), validate user greetings:

        - "Good morning":
            - Valid if current hour is between 5 and 11
            - If current hour is ≥ 12 → respond: "Good Afternoon."
            - If current hour < 5 → respond: "Good Night."

        - "Good afternoon":
            - Valid if current hour is between 12 and 16
            - If current hour < 12 → respond: "Good Morning."
            - If current hour ≥ 17 → respond: "Good Evening."

        - "Good evening":
            - Valid if current hour is between 17 and 20
            - If current hour < 17 → respond: "Actually, it's still afternoon/morning."
            - If current hour ≥ 21 → respond: "It's quite late, you might say good night."

        - "Good night":
            - Valid if current hour is ≥ 21 or < 5
            - If current hour is between 5 and 20 → respond: "It's not night yet. You might want to say good morning/afternoon/evening instead."
            
        - If the greeting is appropriate, respond politely without repeating the same greeting unless the user explicitly asks.
        - If the user is incorrect, politely correct them and provide the correct time-based greeting.
        - If the user asks for the time, provide the current time based on the timestamp.
        - Avoid repeating greetings for each message unless above conditions are met.
        - Do not agree with incorrect greetings.

    1. Only answer using the provided context.
    2. Do not assume or generate information beyond what is explicitly mentioned in the context.
    3. Be contextually aware of the current time of day using the timestamp.
    *Response:* 
    """

    # If custom instructions are provided, concatenate them to the default instructions
    if custom_instructions:
        print("[Prompt] Custom instructions detected:")
        print(custom_instructions)
        full_prompt = default_instructions + "\n" + custom_instructions
    else:
        print("[Prompt] No custom instructions provided.")
        full_prompt = default_instructions

    print("[Prompt] Final RAG prompt built:")
    print(full_prompt)  # print only first 1000 characters for brevity
    return full_prompt

# --- Time Utility ---
def get_current_datetime():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Time] Current datetime: {now}")
    return now

# --- Unified Chat Model Handler ---
def run_chat_model(chat_model, context, question, chat_history, custom_instructions=None):
    current_datetime = get_current_datetime()
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    
    # Final prompt with default + custom instructions
    prompt = build_rag_prompt(context, history_context, question, current_datetime, custom_instructions)

    if "gemini" in chat_model.lower():
        # Gemini expects a single prompt
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            prompt,
            generation_config={"temperature": 0.2}
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
            together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
            model=chat_model
        )
        response = model.predict(prompt)
        return response


# --- FAISS Inference ---
def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history ,custom_instructions=None):
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

        context = "\n\n---\n\n".join(contexts)
        return run_chat_model(chat_model, context, question, chat_history , custom_instructions)

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
            pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV", "us-east-1"))
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
