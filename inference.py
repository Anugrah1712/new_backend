#Inference.py 

from playwright.sync_api import sync_playwright
from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def inference_chroma(chat_model, question, retriever, chat_history):
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert financial advisor. Use the context and the appended chat history in the question to answer accurately and concisely.\n\n"
            "Context: {context}\n\n"
            "{question}\n\n"
            "Answer (be specific and avoid hallucinations):"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    llm_response = qa_chain(question_with_history)
    print("Chroma Response")
    print(llm_response['result'])
    return llm_response['result']

def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history):
    from langchain.chains import LLMChain
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    import numpy as np
    print("Starting inference...")  # Debug point 11
    
    try:
        # Test retrieval
        query_embedding = embedding_model_global.embed_query(question)
        print("Created query embedding")  # Debug point 12
        
        k = 3
        D, I = index.search(np.array([query_embedding]), k=k)
        print(f"Retrieved {len(I[0])} documents")  # Debug point 13
        
        # Collect contexts
        contexts = []
        for i, idx in enumerate(I[0]):
            if idx != -1:
                doc = docstore.search(idx)
                if hasattr(doc, "page_content"):
                    contexts.append(doc.page_content)
                    print(f"Context {i+1}: {doc.page_content[:100]}...")  # Debug point 14
        
        if not contexts:
            return "No relevant context found in the documents."
            
        # Combine contexts
        context = "\n\n---\n\n".join(contexts)
        
        # Create chat completion
        chat_model = ChatTogether(
            together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
            model=chat_model,
        )
        
        prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template="""You are a financial advisor specializing in Bajaj Finance Fixed Deposits. Use the following context to answer questions accurately:

            Context: {context}

            Chat History: {history}

            Question: {question}

            Answer to general conversation texts like hello,bye,etc
            **Strict Instructions to Avoid Hallucination:**
            1. Only answer using the provided context.
            2. Do not assume or generate information beyond what is explicitly mentioned in the context.
            3. Always quote numerical values, interest rates, and tenure periods exactly as found in the context.
            4. If multiple interest rates exist, specify whether they apply to general citizens or senior citizens.
            5. For yield-related questions, provide both the FD rate and yield percentage.
            6. If the question requires a numerical calculation (e.g., FD maturity, tax deduction), perform the necessary calculation.
            7. Use the compound interest formula where required:
               [A = P * r * t]
               where:  
               - P = Principal amount  
               - r = Interest rate (in decimal)  
               - n = Compounding frequency per year (1 for annual, 12 for monthly)  
               - t = Time in years  
            8. For tax-related queries, apply TDS deduction rules:
               - If FD interest exceeds ₹40,000 (₹50,000 for seniors), deduct 10% TDS.
               - If PAN is missing, apply 20% TDS.
            9. When discussing senior citizen rates: 
               - Senior citizens are individuals aged 60 years and above
               - General citizens are individuals below 60 years of age
            10. If the question asks for an interest rate for a specific tenure (e.g., 37 months), but the provided information only contains range-based tenures, find the correct range and use the corresponding interest rate.
                Example: If the tenure is 37 months, and the provided range is 36-60 months, return the interest rate for 36-60 months.
            11. If multiple ranges match, return the rate from the most relevant range.
            12. Maintain clarity and conciseness, providing complete but direct answers.
            13. Keep answers concise, accurate and to the point without unnecessary explanations.

            **Response:**"""
        )
        
        qa_chain = LLMChain(llm=chat_model, prompt=prompt_template)
        
        history_context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
        )
        
        print("Generating response...")  # Debug point 15
        answer = qa_chain.run(
            history=history_context,
            context=context,
            question=question
        )
        print("Faiss response")
        return answer
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")  # Changed from st.error to print
        return "An error occurred while processing your question."

def inference_pinecone(chat_model, question,embedding_model_global, pinecone_index,chat_history):
  import pinecone
  from pinecone import Pinecone
  from langchain_together import ChatTogether
  import numpy as np

  query_embedding = embedding_model_global.embed_query(question)
  query_embedding = np.array(query_embedding)

  search_results =  pinecone_index.query(
      vector=query_embedding.tolist(),
      top_k=4,
      include_metadata=True
  )
  contexts = [result['metadata']['text'] for result in search_results['matches']]

  context = "\n".join(contexts)

  history = "\n".join(
      [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
  )

  prompt = f"""You are a financial advisor specializing in Bajaj Finance Fixed Deposits. Use the following context to answer questions accurately:

            Context: {context}

            Chat History: {history}

            Question: {question}

            Answer to general conversation texts like hello,bye,etc
            *Strict Instructions to Avoid Hallucination:*
            1. Only answer using the provided context.
            2. Do not assume or generate information beyond what is explicitly mentioned in the context.
            3. Always quote numerical values, interest rates, and tenure periods exactly as found in the context.
            4. If multiple interest rates exist, specify whether they apply to general citizens or senior citizens.
            5. For yield-related questions, provide both the FD rate and yield percentage.
            6.If the question requires a numerical calculation (e.g., FD maturity, tax deduction), perform the necessary calculation.**
            7.Use the compound interest formula where required:**  
            [A = P * r * t
            ]
            where:  
            - P = Principal amount  
            - r = Interest rate (in decimal)  
            - n = Compounding frequency per year (1 for annual, 12 for monthly)  
            - t = Time in years  
            8. For tax-related queries**, apply TDS deduction rules:
            - If FD interest *exceeds ₹40,000 (₹50,000 for seniors), deduct **10% TDS*.
            - If PAN is missing, apply *20% TDS*.
            4. If multiple interest rates exist*, clearly specify whether they apply to **general citizens* or *senior citizens*.
            5. For yield-related questions, provide both FD rate and yield percentage.
            If the question asks for an interest rate for a *specific tenure (e.g., 37 months), but the provided information only contains **range-based tenures*, find the correct range and use the corresponding interest rate.
            2. Example: If the tenure is *37 months, and the provided range is **36-60 months, return the interest rate for **36-60 months*.
            3. If multiple ranges match, return the rate from the most relevant range.
            6. Maintain clarity and conciseness, avoiding unnecessary details.

            *Response:*"""

  llm = ChatTogether(api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
                  model=chat_model,  )

  response = llm.predict(prompt)
  print("Pinecone response")
  print(response)
  print(context)
  return response


def inference_weaviate(chat_model, question , vs , chat_history):
    from langchain_together import ChatTogether
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    # Retrieve documents from Weaviate
    retriever = vs.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(question)

    # Extract context from retrieved docs
    contexts = [doc.page_content for doc in retrieved_docs]
    context = "\n\n---\n\n".join(contexts)

    # Define prompt template
    template = """
    You are an expert financial advisor. Use the context and the appended chat history in the question to answer accurately and concisely.

    Context:
    {context}

    {question}

    Answer (be specific and avoid hallucinations):
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    # Run Weaviate retrieval pipeline
    result = chat_model.predict(
        prompt.format(context=context, question=question_with_history)
    )
    print("Weaviate Response")
    return result

def inference_qdrant(chat_model, question, embedding_model_global, qdrant_client, chat_history):
    from qdrant_client.http.models import SearchRequest
    from langchain_together import ChatTogether
    import numpy as np

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    query_embedding = embedding_model_global.embed_query(question_with_history)
    query_embedding = np.array(query_embedding)

    search_results = qdrant_client.search(
        collection_name="text_vectors",
        query_vector=query_embedding,
        limit=2
    )

    contexts = [result.payload['page_content'] for result in search_results]
    context = "\n".join(contexts)

    prompt = f"""
    You are a helpful assistant. Use the following retrieved documents to answer the question:

    Context:
    {context}

    {question_with_history}

    Answer:
    """

    llm = ChatTogether(
        api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model
    )

    response = llm.predict(prompt)

    print("Qdrant Response")
    print(response)
    return response


def inference(vectordb_name, chat_model, question, embedding_model_global, chat_history, pinecone_index_name=None , vs=None , qdrant_client=None):
    if vectordb_name == "Chroma":
        from langchain.vectorstores import Chroma
        retriever = Chroma(persist_directory='db', embedding_function=embedding_model_global).as_retriever()
        return inference_chroma(chat_model, question, retriever, chat_history)

    elif vectordb_name == "FAISS":
        from langchain.vectorstores import FAISS
        import os

        faiss_index_path = "faiss_index/index.faiss"

        if os.path.exists(faiss_index_path):  
            faiss_store = FAISS.load_local("faiss_index", embedding_model_global, allow_dangerous_deserialization=True)
            return inference_faiss(chat_model, question, embedding_model_global, faiss_store.index, faiss_store.docstore, chat_history)
        else:
            return "❌ FAISS index file not found. Please run preprocessing first."
        
    elif vectordb_name == "Pinecone":
        from pinecone import Pinecone

        if pinecone_index_name:
            # Initialize Pinecone
            pinecone = Pinecone(api_key="pcsk_42Yw14_EaKdaMLiAJfWub3s2sEJYPW3jyXXjdCYkH8Mh8rD8wWJ3pS6oCCC9PGqBNuDTuf", environment="us-east-1")
            pinecone_index = pinecone.Index(pinecone_index_name)  # ✅ Load Pinecone index dynamically
            return inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history)
        else:
            return "❌ Pinecone index not found. Please run preprocessing first."
        
    elif vectordb_name == "Weaviate":
        if vs:
            return inference_weaviate(chat_model, question, vs, chat_history)  # ✅ Weaviate Support
        else:
            return "❌ Weaviate vector store not found. Please run preprocessing first."
        
    elif vectordb_name == "Qdrant":
        if qdrant_client:
            return inference_qdrant(chat_model, question, embedding_model_global, qdrant_client, chat_history)  # ✅ Qdrant Support
        else:
            return "❌ Qdrant vector store not found. Please run preprocessing first."

    else:
        return "❌ Invalid vector database selection!"


