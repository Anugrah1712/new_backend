import optuna
import asyncio
import glob
from preprocess import preprocess_text
from inference import get_response
from langchain.schema import Document as LangchainDocument


# 🔸 Simple evaluation: check if important keywords are present
test_cases = [
    {"question": "What is the full name and core focus of Rahi Technologies?", "keywords": ["Rahi Platform Technologies", "SaaS", "financial services"]},
    {"question": "Where is Rahi Technologies located?", "keywords": ["4010", "Ganga Trueno", "Airport Road", "Pune", "411014"]},
    {"question": "What is Rahi Technologies’ vision for its clients?", "keywords": ["cloud-native", "omni-channel", "legacy financial systems", "customer lifecycle"]},
    {"question": "What architecture principles does the Rahi platform follow?", "keywords": ["microservices", "DevOps", "API-first", "multi-tenant", "polyglot persistence"]},
    {"question": "Which cloud platforms does Rahi’s platform support?", "keywords": ["AWS", "Azure", "GCP", "multi-cloud"]},
    {"question": "Who founded Rahi Technologies and what is his background?", "keywords": ["Rakesh Bhatt", "30 years", "financial services", "technology"]},
    {"question": "Who is the CTO of Rahi and what is his experience?", "keywords": ["Vivek Kant", "25 years", "digital architecture", "APIs", "cloud infrastructure"]},
    {"question": "Which key leaders contribute to Rahi’s technology and engineering strategy?", "keywords": ["Sagar Pandkar", "Imroz Khan", "Bhavesh Mehta", "Rajib Bhowmick"]},
    {"question": "What values define the work culture at Rahi Technologies?", "keywords": ["ambition", "ownership", "execution rigor", "transparency", "impact"]},
    {"question": "What open positions are available at Rahi Technologies?", "keywords": ["Senior Software Engineer", "DevOps", "Solution Architect", "SDET"]},
    {"question": "Which DevOps practices are emphasized at Rahi Technologies?", "keywords": ["CI/CD", "Docker", "Kubernetes", "Terraform", "Ansible"]},
    {"question": "What monitoring and security tools does Rahi use?", "keywords": ["Grafana", "ELK", "Nagios", "compliance", "DPDP", "DLG"]},
    {"question": "What type of content does Rahi publish in its Knowledge section?", "keywords": ["SaaS platform", "Agile SDLC", "API gateways", "real-time observability"]},
    {"question": "How can someone get in touch with Rahi Technologies?", "keywords": ["contact page", "form", "name", "email", "phone", "subject", "message"]},
    {"question": "How did Rahi Technologies get its name?", "keywords": ["Rakesh Bhatt", "RA", "HI", "traveler", "digital transformation"]}
]

def evaluate_answer(answer, expected_keywords):
    return sum(k.lower() in answer.lower() for k in expected_keywords) / len(expected_keywords)

def objective(trial):
    chunk_size = trial.suggest_categorical("chunk_size", [256, 512, 768])
    chunk_overlap = trial.suggest_int("chunk_overlap", 50, 200, step=50)
    top_k = trial.suggest_int("top_k", 3, 10)
    temperature = trial.suggest_float("temperature", 0.2, 0.7)
    max_tokens = trial.suggest_int("max_output_tokens", 512, 1024, step=128)

    pdf_files = glob.glob("Documentation_Rahi.pdf")

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
