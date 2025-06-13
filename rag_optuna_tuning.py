import optuna
import asyncio
import glob
from preprocess import preprocess_text
from inference import get_response
from langchain.schema import Document as LangchainDocument


# 🔸 Simple evaluation: check if important keywords are present
test_cases = [
    {"question": "What is Kunjesh Parekh's current role and where is he working?", "keywords": ["Head of Data Science", "Bajaj Finance", "Pune"]},
    {"question": "What are Kunjesh Parekh's areas of expertise?", "keywords": ["Machine Learning", "Deep Learning", "LLMs", "Reinforcement Learning"]},
    {"question": "Which domains has Kunjesh worked in throughout his career?", "keywords": ["Finance", "Pharmaceutical", "IT", "Manufacturing"]},
    {"question": "What chatbot project did Kunjesh deploy using Langchain and vector databases?", "keywords": ["RAG", "chatbot", "Langchain", "FAISS", "ChromaDB"]},
    {"question": "Which cloud platforms and deployment tools does Kunjesh have experience with?", "keywords": ["AWS", "GCP", "Render", "EC2", "Docker"]},
    {"question": "What contributions has Kunjesh made to Bajaj Finserv's app?", "keywords": ["personalization", "MAU", "DAU", "incremental revenue", "HEART Metrics"]},
    {"question": "What kind of recommendation systems has Kunjesh built for loans and credit cards?", "keywords": ["Personal Loan", "Credit Card", "decision tree", "CTR"]},
    {"question": "How has Kunjesh contributed to translation and vernacular processing tasks?", "keywords": ["vernacular translation", "M2M100", "MarianMT", "BLEU score"]},
    {"question": "What A/B testing framework did Kunjesh establish?", "keywords": ["A/B testing", "statistical significance", "test case design"]},
    {"question": "What is Kunjesh's teaching and mentoring experience?", "keywords": ["trainer", "upGrad", "Dale Carnegie", "data science course", "IMS Proschool"]},
    {"question": "Which companies has Kunjesh worked at previously?", "keywords": ["Merck", "Cognizant", "Jubilant Life Sciences"]},
    {"question": "What image processing projects has Kunjesh worked on?", "keywords": ["tumor prediction", "CNN", "deep learning", "tensorflow", "image processing"]},
    {"question": "What competitions and accolades has Kunjesh achieved in data science?", "keywords": ["top 10%", "GE Analytics", "Kaggle", "Techgig", "Analytics Vidhya"]},
    {"question": "What programming and data tools is Kunjesh proficient in?", "keywords": ["Python", "R", "Pyspark", "SQL", "Databricks"]},
    {"question": "What academic qualifications does Kunjesh hold?", "keywords": ["PhD", "IIT Jodhpur", "MBA", "IIT Roorkee", "B.Tech", "Nirma University"]}
]

def evaluate_answer(answer, expected_keywords):
    return sum(k.lower() in answer.lower() for k in expected_keywords) / len(expected_keywords)

def objective(trial):
    chunk_size = trial.suggest_categorical("chunk_size", [256, 512, 768])
    chunk_overlap = trial.suggest_int("chunk_overlap", 50, 200, step=50)
    top_k = trial.suggest_int("top_k", 3, 10)
    temperature = trial.suggest_float("temperature", 0.2, 0.7)
    max_tokens = trial.suggest_int("max_output_tokens", 512, 1024, step=128)

    pdf_files = glob.glob("Kunjesh_Parekh_202505.pdf")

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
