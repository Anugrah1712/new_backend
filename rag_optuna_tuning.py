import optuna
import asyncio
import glob
from preprocess import preprocess_text
from inference import get_response
from langchain.schema import Document as LangchainDocument


# 🔸 Simple evaluation: check if important keywords are present
test_cases = [
    {"question": "What is GTSBOT and which architecture is it based on?", "keywords": ["GTSBOT", "Retrieval-Augmented Generation", "contextual assistant"]},
    {"question": "Which input formats and sources does GTSBOT support for analysis?", "keywords": ["PDF", "Word", "URLs", "multimodal input"]},
    {"question": "What vector databases are compatible with GTSBOT?", "keywords": ["Chroma", "FAISS", "Qdrant", "Pinecone", "Weaviate"]},
    {"question": "How does GTSBOT ensure context-aware responses?", "keywords": ["semantic search", "retrieved context", "language model", "prompt"]},
    {"question": "Which large language models are supported by GTSBOT?", "keywords": ["GPT-4", "Llama-3.3-70B", "Meta-Llama-3.1", "Gemini 1.5 Flash"]},
    {"question": "What functionalities does the GTSBOT Developer Console provide?", "keywords": ["prompt customization", "model selection", "vector indexing", "system configuration"]},
    {"question": "How does GTSBOT handle security and privacy for enterprise deployment?", "keywords": ["self-hosted", "Docker", "AES encryption", "no data logging"]},
    {"question": "What features make GTSBOT suitable for customer support use cases?", "keywords": ["technical manuals", "warranty documents", "first-level support", "automated responses"]},
    {"question": "What are the key use cases where GTSBOT adds value?", "keywords": ["Legal", "Education", "Enterprise", "Customer Support", "Consulting"]},
    {"question": "What are some of the common troubleshooting scenarios with GTSBOT?", "keywords": ["API key", "chunking failure", "vector indexing error", "Admin key"]},
    {"question": "What capabilities does GTSBOT offer for multilingual and voice-based interaction?", "keywords": ["speech input", "text-to-speech", "multilingual support"]},
    {"question": "What does the GTSBOT Pro plan include?", "keywords": ["$999", "unlimited uploads", "priority support", "real-time model switching"]},
    {"question": "How does prompt engineering affect GTSBOT's behavior?", "keywords": ["system prompt", "custom instructions", "domain-specific", "structured prompts"]},
    {"question": "How does GTSBOT process and retrieve content from uploaded documents?", "keywords": ["chunking", "embedding", "semantic retrieval", "FastAPI"]},
    {"question": "What deployment options and environments are supported by GTSBOT?", "keywords": ["cloud", "on-premise", "Docker", "air-gapped"]}
]


def evaluate_answer(answer, expected_keywords):
    return sum(k.lower() in answer.lower() for k in expected_keywords) / len(expected_keywords)

def objective(trial):
    chunk_size = trial.suggest_categorical("chunk_size", [256, 512, 768])
    chunk_overlap = trial.suggest_int("chunk_overlap", 50, 200, step=50)
    top_k = trial.suggest_int("top_k", 3, 10)
    temperature = trial.suggest_float("temperature", 0.2, 0.7)
    max_tokens = trial.suggest_int("max_output_tokens", 512, 1024, step=128)

    pdf_files = glob.glob("Datalysis Documentaion.pdf")

    async def run_preprocessing():
        return await preprocess_text(
            files=pdf_files,         # ✅ Pass your PDF files here
            size=chunk_size,
            overlap=chunk_overlap,
            scraped_data=None        # ✅ Not needed when using PDFs
        )

    try:
        split_docs = asyncio.run(run_preprocessing())
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return 0.0

    total_score = 0
    for test in test_cases:
        try:
            answer = get_response(
                question=test["question"],
                top_k=top_k,
                temperature=temperature,
                max_output_tokens=max_tokens,
                custom_docs=split_docs  
            )
            score = evaluate_answer(answer, test["keywords"])
            total_score += score
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return 0.0

    return total_score / len(test_cases)
 
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\n✅🎯 Final Best Parameters after tuning:")
    print("────────────────────────────────────────")
    for param, value in study.best_params.items():
        print(f"{param:<20}: {value}")
    print("────────────────────────────────────────")
    print(f"🏆 Best Score Achieved    : {study.best_value:.3f}")
