import optuna
import asyncio
import glob
from preprocess import preprocess_text
from inference import get_response
from langchain.schema import Document as LangchainDocument


# 🔸 Simple evaluation: check if important keywords are present
test_cases = [
    {"question": "What is the core focus of LuckPay as a fintech company?", "keywords": ["digital payments", "financial solutions", "fintech", "LuckPay"]},
    {"question": "What is the vision of LuckPay?", "keywords": ["seamless", "secure", "scalable", "digital financial transactions", "businesses"]},
    {"question": "What mission drives LuckPay’s platform and services?", "keywords": ["robust APIs", "automation", "real-time analytics", "digital payments"]},
    {"question": "What foundational pillars define LuckPay’s service offerings?", "keywords": ["scalability", "security", "customization", "support"]},
    {"question": "What services does LuckPay offer under Payment Gateway Tech Solutions?", "keywords": ["gateway setup", "API", "transaction flow", "compliance", "real-time processing"]},
    {"question": "How does LuckPay support Payin and Payout operations?", "keywords": ["tech stack support", "power-optimized APIs", "Payin", "Payout", "integration"]},
    {"question": "What reconciliation services are offered by LuckPay?", "keywords": ["reconciliation", "settlement", "bookkeeping", "GST", "TDS", "compliance"]},
    {"question": "What kind of consultancy does LuckPay provide?", "keywords": ["gateway integration", "fintech strategy", "growth", "regulatory compliance"]},
    {"question": "How does LuckPay assist fintech companies with digital marketing?", "keywords": ["online visibility", "merchant acquisition", "marketing services", "fintech"]},
    {"question": "What taxation services does LuckPay provide?", "keywords": ["taxation", "GST", "TDS", "financial reporting", "compliance"]},
    {"question": "What differentiates LuckPay in terms of technology and support?", "keywords": ["cutting-edge", "end-to-end support", "data-driven", "RBI", "NPCI", "2FA", "KYC"]},
    {"question": "What do client testimonials say about LuckPay’s impact?", "keywords": ["Arun Kumar", "Neha Sharma", "Vikram Singh", "digital marketing", "tech stack", "smooth transactions"]},
    {"question": "What are some key blog topics covered by LuckPay?", "keywords": ["UPI", "authentication", "interoperability", "AI", "fintech trends", "downtime", "tax structures"]},
    {"question": "What policies are defined on LuckPay’s website?", "keywords": ["Privacy Policy", "Terms of Use", "Refund Policy", "Grievance Redressal", "Responsible Disclosure"]},
    {"question": "How can users contact LuckPay or stay engaged?", "keywords": ["newsletter", "sales@luckpay.co", "Facebook", "LinkedIn", "Instagram", "subscribe"]}
]

def evaluate_answer(answer, expected_keywords):
    return sum(k.lower() in answer.lower() for k in expected_keywords) / len(expected_keywords)

def objective(trial):
    chunk_size = trial.suggest_categorical("chunk_size", [256, 512, 768])
    chunk_overlap = trial.suggest_int("chunk_overlap", 50, 200, step=50)
    top_k = trial.suggest_int("top_k", 3, 10)
    temperature = trial.suggest_float("temperature", 0.2, 0.7)
    max_tokens = trial.suggest_int("max_output_tokens", 512, 1024, step=128)

    pdf_files = glob.glob("Documentation LuckPay.pdf")

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
