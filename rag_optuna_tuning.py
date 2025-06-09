import optuna
import asyncio
from preprocess import preprocess_text
from inference import get_response
from langchain.schema import Document as LangchainDocument

# 🔸 Sample input data (no need for files during tuning)
sample_scraped_data = """
GPTBOT Documentation
Name
GPTBOT is an advanced AI-powered contextual assistant built using Retrieval-Augmented Generation (RAG) architecture. Designed to support legal, educational, enterprise, and customer service use cases, GPTBOT allows users to engage in intelligent, context-aware conversations with documents, web content, and structured data. The system is built with modularity in mind and can be deployed both on cloud platforms and on-premise infrastructure using Docker.
Features
GPTBOT offers a comprehensive suite of features that enhance the capabilities of traditional chatbots by combining document understanding, vector search, and large language model capabilities. One of its core strengths is the ability to process multimodal input. Users can upload files such as PDFs, Word documents, and plain text files, or provide URLs to analyze online content in real time.
The backend leverages Retrieval-Augmented Generation, which enables it to retrieve the most relevant context from indexed documents and pass it to the language model for accurate, grounded responses. GPTBOT supports leading vector databases including Chroma, FAISS, Qdrant, Pinecone, and Weaviate. Documents are embedded using a chosen embedding model and indexed for high-speed semantic retrieval.
GPTBOT includes an integrated Developer Console that provides users with full control over vector indexing, prompt customization, model selection, and system configuration. From this interface, administrators can upload new content, adjust system prompts, and switch between supported large language models. The console is protected with an administrative access key to ensure secure usage.
The chatbot supports multiple large language models such as GPT-4, Llama-3.3-70B-Instruct-Turbo, Meta-Llama-3.1-405B-Instruct-Turbo,scb10x-llama3-typhoon-v1-5-8b-instruct, and Gemini 1.5 Flash. This allows users to choose the most appropriate model based on cost, performance, or task-specific requirements.
GPTBOT also supports voice-based interaction. Users can issue queries through speech and receive synthesized responses, enabling hands-free, natural communication. Time-awareness is also embedded into the system, allowing it to respond with appropriate greetings and context based on the current time and user time zone.
Additionally, GPTBOT uses content hashing for intelligent caching, which ensures that duplicate data is not redundantly processed, optimizing performance while preserving privacy.
How It Works
GPTBOT operates through a secure and modular pipeline that begins with content ingestion and ends with a context-aware response. Users can submit documents or links for analysis. The content is processed and split into smaller segments that preserve contextual meaning. Each segment is embedded using an appropriate model and stored in the selected vector database.
When a user submits a query, GPTBOT converts the query into a vector representation and performs a semantic search across the indexed document chunks. The most relevant results are retrieved and passed as context to the selected language model. The model is instructed via a custom prompt to use only the retrieved context when generating answers, ensuring that the responses remain grounded in the user-provided data and do not rely on external assumptions or hallucinations.
The system allows runtime customization of prompts, model switching, database configuration, and content re-indexing. Its backend is built using FastAPI and fully containerized with Docker, making it portable and deployable in secure environments without reliance on external processing or storage.
Use Cases
Legal Sector: Law firms and legal professionals can use GPTBOT to quickly analyze contracts, agreements, and case files. By uploading legal documents, teams can query terms, find relevant clauses, and summarize content without reading the entire document manually.
Education and Research: GPTBOT enables students and researchers to interact with academic papers, notes, or lecture material. It can summarize complex texts, explain concepts, and assist with report writing by referencing the actual content uploaded by the user.
Enterprise Knowledge Base: Companies can upload internal manuals, policy documents, and operational procedures to GPTBOT, allowing employees to ask natural language questions and receive immediate answers based on their organization’s specific documentation.
Customer Support: Businesses can integrate GPTBOT into their customer service workflows. With access to knowledge base articles and product documentation, GPTBOT can serve as a first-level support assistant that accurately resolves common customer queries.
Consulting and Data Analysis: Analysts and consultants can upload client reports, industry data, or strategy documents and ask contextual questions to extract key points, generate insights, or prepare presentations with precision and speed.
Testimonials
Organizations and professionals across sectors have integrated GPTBOT into their workflows with measurable results. A legal consultant working with a mid-sized law firm shared that GPTBOT helped reduce over 60 hours per month in document review time by enabling direct, natural language queries on complex contracts.
In the academic field, a professor of AI studies commented that GPTBOT has significantly enhanced student engagement. Learners now use the tool to clarify difficult topics, summarize readings, and prepare coursework with a higher degree of autonomy.
From an enterprise perspective, a lead data scientist at Bajaj Finserv highlighted the ease of deployment and administration. GPTBOT was containerized and deployed within their infrastructure, with the Developer Console providing complete operational control, from content updates to prompt refinement.
Pricing
GPTBOT is available under a professional pricing plan suitable for high-scale and enterprise environments. The Pro Version is priced at $999 per month. This plan includes unlimited document uploads, unrestricted access to supported vector databases, and real-time model switching between  GPT-4, Llama-3.3-70B-Instruct-Turbo, Meta-Llama-3.1-405B-Instruct-Turbo,scb10x-llama3-typhoon-v1-5-8b-instruct, and Gemini 1.5 Flash. Subscribers gain full access to the Developer Console, where they can control prompts, models, and system behavior. Additionally, the Pro plan includes priority support and feature updates to keep the system secure, scalable, and up to date with the latest advancements.
Security and Privacy
GPTBOT is built with security and privacy as foundational principles, making it suitable for deployment in industries that require stringent data handling practices such as finance, legal, education, and healthcare.
1. Data Confidentiality  All uploaded documents, embeddings, and chat interactions are processed and stored locally within your self-hosted environment. GPTBOT does not transmit any data to external servers or third-party APIs unless explicitly configured by the administrator. This guarantees that sensitive business documents remain private and fully under organizational control.
2. Deployment Security  GPTBOT supports containerized deployment through Docker, enabling isolated and controlled environments. Admin access to configuration features, including the Developer Console, is protected using access keys or tokens to prevent unauthorized tampering.
3. Encryption  For production deployments, GPTBOT can be configured with TLS/SSL for secure transmission of data over HTTP. Document storage can also be secured using AES encryption when used in combination with secure storage policies.
4. No Data Logging by Default  GPTBOT does not store chat histories, user queries, or logs unless explicitly enabled. This default behavior supports compliance with data minimization policies, ensuring no accidental retention of user or corporate data.
5. Role-Based Access (Optional)  Enterprise-grade setups can be extended to integrate with identity providers (e.g., LDAP, OAuth2) to support role-based access, defining permissions for Admins, Analysts, and End-Users.
Prompt Engineering Guide
To ensure GPTBOT delivers accurate, relevant, and well-structured responses, prompt engineering plays a pivotal role. GPTBOT is equipped with a Developer Console, which enables administrators to dynamically modify the assistant’s system prompt based on specific use case requirements, ensuring the assistant operates in line with domain expectations and organizational standards.
The default system prompt is designed to enforce foundational behaviors across all interactions. This includes greeting users based on the current time of day, refusing to answer questions if there is insufficient context available, remaining strictly bound to the provided context without hallucinating or generating information not present in source materials, and adhering to document source boundaries. These behaviors ensure the assistant remains reliable, accurate, and contextually grounded.
Organizations can expand upon this default prompt by appending custom instructions tailored to their specific operational needs. For instance, administrators may wish to adjust the tone of the assistant—whether formal, casual, or technical—depending on the audience. They may also choose to prioritize certain types of documents, such as contracts over internal memos, or fine-tune the assistant’s understanding of domain-specific language. Additionally, disclaimers or footers can be automatically included in responses, especially in regulated industries like finance or law.
To achieve the best results when customizing prompts, several best practices should be followed. Clear delimiters, such as triple dashes (---), should be used when writing multiple directives to distinguish them cleanly. Prompts should avoid being overly verbose or contradictory, as this can lead to inconsistent assistant behavior. Lastly, all changes should be tested incrementally using the Developer Console to observe their effect on the assistant’s output.
Examples of prompt customization for specific use cases include adding directives like “Always cite the clause number when referencing contracts” in legal scenarios, instructing the assistant to “Always simplify the response for high school-level understanding” in educational deployments, or stating “Never make assumptions beyond uploaded company policy documents” in human resources or compliance contexts. These customizations ensure that GPTBOT operates reliably within the bounds of your domain while enhancing user trust and experience.
Use Case Playbooks
GPTBOT is highly adaptable and can be tailored to serve diverse functions across industries. The following playbooks illustrate typical use cases where GPTBOT provides measurable value.
In the Legal Document Assistant use case, GPTBOT is employed to summarize, retrieve, and explain legal agreements, contracts, and policies. By uploading multi-page documents such as NDAs, service agreements, or employment contracts, legal teams can query the assistant with questions like “What is the termination clause?” or “Is there a non-compete provision in this contract?” GPTBOT responds with precise references to the source content. To ensure legal clarity, prompts can include disclaimers like “This is not legal advice.”
In the Internal Knowledgebase for Employees use case, GPTBOT serves as an internal helpdesk assistant that provides immediate answers to employee queries using uploaded HR manuals, onboarding documents, or standard operating procedures (SOPs). Organizations benefit from having an always-available assistant that can accurately respond to queries about leave policies, reimbursement procedures, or code of conduct guidelines. Administrators can fine-tune GPTBOT by including constraints such as “Do not make up policy if not present in the documents.”
For Customer Support Knowledge Bots, GPTBOT automates responses to client queries using technical manuals, warranty documents, service records, or historical support tickets. By uploading these documents, organizations can allow GPTBOT to assist users with questions like “How do I reset my device?” or “What does the warranty cover?” Using structured prompts that enforce citing specific product names or sections ensures clarity and consistency in responses.
In an educational context, GPTBOT functions as an Educational Tutor Bot. Teachers, tutors, or academic institutions can upload syllabus documents, textbooks, notes, or whitepapers, enabling students to query the assistant for definitions, explanations, or problem-solving methods. To enhance comprehension, prompts can instruct the assistant to simplify explanations and provide illustrative examples, especially when catering to younger learners or non-expert audiences.
Each of these use cases demonstrates the power and flexibility of GPTBOT when configured with the right context and customized prompts. The tool's ability to provide accurate, reference-based answers makes it highly valuable across functional domains.
Troubleshooting and Error Codes
Even with its robust architecture, GPTBOT may occasionally encounter issues due to configuration errors, missing context, or integration limitations. The following section outlines common problems, likely causes, and recommended resolutions to help administrators troubleshoot effectively.
One of the most common issues is when GPTBOT does not respond to user input. This typically occurs when the underlying Large Language Model (LLM) API key is either not set or has expired. To resolve this, administrators should navigate to the Developer Console, locate the “Set API Key” section, and ensure a valid and active API key is configured.
Another frequent issue is when GPTBOT replies with a message stating, “I cannot find enough information to answer your question.” This generally indicates that the current documents either lack the necessary context or were not correctly indexed. To fix this, administrators should verify that the documents were successfully uploaded and chunked. If necessary, the documents can be reprocessed, and users should be guided to ask questions aligned with the available content.
If users experience delays in response times, the likely culprits are large document sizes, constrained system resources, or slow responses from the LLM provider. In such cases, system administrators should check CPU and memory utilization, and optionally switch to a faster model such as Gemini 1.5 Flash if supported.
When errors occur during vector indexing—for example, an error message stating “Failed to index documents”—the root cause is often misconfigured credentials or memory limitations in the vector database. Administrators should confirm that the vector DB configuration in the Developer Console is accurate and that the database server is online and accessible.
Occasionally, the Developer Console may fail to open. This issue is usually caused by an unset or incorrect Admin key or due to browser caching problems. Administrators should ensure that the ADMIN_KEY environment variable is correctly set and try clearing the browser cache before reattempting access.
Lastly, a problem where documents do not chunk properly may stem from unsupported file formats or corrupted uploads. GPTBOT currently supports common formats such as PDF and DOCX. Re-uploading the file in a compatible format and reviewing backend logs for parsing errors can help isolate and resolve the issue.
By proactively addressing these common errors, administrators can ensure that GPTBOT continues to deliver seamless, reliable service to end-users.

Frequently Asked Questions (FAQs)
1. What document types are supported by GPTBOT? GPTBOT supports a variety of file formats including PDF and  DOCX. It also supports URL-based content analysis for public web pages. The system parses and segments content into meaningful chunks to enable accurate retrieval and response.
2. Does GPTBOT work offline or on-premise? Yes. GPTBOT can be deployed entirely within a secure, private environment using Docker. This includes support for deployment in air-gapped systems and networks where internet access is restricted or prohibited.
3. Can GPTBOT be customized for my organization’s tone and behavior?  Absolutely. The system includes a prompt editor that allows administrators to define the assistant's tone, behavior, restrictions, and response logic. This is especially useful for compliance, branding, and specific communication styles.
4. Is my data stored or shared with third parties? No. GPTBOT is designed for maximum data privacy and security. All files and processed data remain within your deployment environment. No content is shared externally or stored on third-party servers unless explicitly configured by your team.
5. Can GPTBOT extract and understand tables, structured data, or FAQs from documents?  Yes. GPTBOT has built-in logic to handle structured information such as tables, lists, and FAQs. It can present extracted data in a clear, concise manner and respond to queries based on tabular or sectioned formats.
6. Can multiple users interact with GPTBOT at once?  Yes. GPTBOT supports concurrent sessions and can be scaled horizontally to handle multiple user interactions simultaneously. This makes it ideal for deployment in enterprise and customer support environments.
7. Does GPTBOT maintain chat history or context between sessions?  By default, GPTBOT treats each session independently to prioritize user privacy. However, optional session memory or persistent conversation history can be configured based on deployment needs.
8. What models does GPTBOT support?  GPTBOT supports leading LLMs including GPT-4, Llama-3.3-70B-Instruct-Turbo, Meta-Llama-3.1-405B-Instruct-Turbo,scb10x-llama3-typhoon-v1-5-8b-instruct, and Gemini 1.5 Flash. The model can be selected dynamically via the Developer Console, allowing flexibility based on budget or task complexity.
9. Can I upload multiple documents at once?  Yes. GPTBOT supports batch uploading of multiple files. Once uploaded, the system processes and indexes the content automatically, making it available for contextual retrieval.
10. What vector databases does GPTBOT support?  GPTBOT is compatible with Chroma,FAISS, Qdrant, Pinecone, and Weaviate. The vector store can be switched from the Developer Console without requiring re-deployment.
11. Can GPTBOT summarize documents?  Yes. Users can ask GPTBOT to summarize entire documents or specific sections. The assistant uses embedded context to generate summaries that are concise, relevant, and based solely on the content provided.
12. Can GPTBOT generate citations or reference sources in responses?  Yes. GPTBOT can be configured to cite the exact chunk or section from which the information is retrieved. This helps maintain transparency and trust, especially in academic or legal use cases.
13. How secure is GPTBOT for use in regulated industries? GPTBOT can be deployed in compliance with organizational security standards, including encryption at rest and in transit, admin-key protected interfaces, containerized isolation, and air-gapped hosting. It is suitable for use in sectors that require strict data confidentiality such as finance, law, and healthcare.
14. Is GPTBOT mobile-friendly or available as an app? While GPTBOT is primarily deployed as a web-based interface, it can be integrated into mobile apps or adapted into a native Android/iOS experience. The backend supports RESTful APIs that can be called from any frontend environment.
15. Can GPTBOT respond using voice or speech? Yes. GPTBOT includes speech input and text-to-speech output capabilities. Users can interact using their voice and receive spoken responses, making it suitable for hands-free scenarios.
16. Does GPTBOT support multi-language documents and queries? GPTBOT is capable of understanding and responding to content in multiple languages, depending on the capabilities of the underlying model selected. It supports multilingual document ingestion and can respond appropriately to queries in supported languages.
17. Is GPTBOT available as a SaaS or only self-hosted? GPTBOT is primarily offered as a self-hosted solution to maintain maximum data control. However, a managed SaaS version can be made available upon request for organizations looking for cloud-managed deployment.
18. What kind of customer support is available?  Pro version subscribers receive priority support, including onboarding assistance, deployment guidance, and ongoing technical support. Support requests can be made via GPTBOT@AI or through the in-product support command.
Contact Information
For inquiries, support, partnership opportunities, or product demos, please contact the GPTBOT team at GPTBOT@AI. If you are using the chatbot interface, you may also type “Contact Support” to be guided through the help system.
"""

# 🔸 Simple evaluation: check if important keywords are present
test_cases = [
    {"question": "What is Rahi Technologies and what does the company do?", "keywords": ["Rahi Technologies", "SaaS", "financial services"]},
    {"question": "Where is Rahi Technologies located?", "keywords": ["location", "Pune", "Ganga Trueno"]},
    {"question": "What is the core mission and vision of Rahi Technologies?", "keywords": ["mission", "vision", "cloud-native", "omni-channel"]},
    {"question": "What kind of platform architecture does Rahi Technologies use?", "keywords": ["architecture", "microservices", "API-first"]},
    {"question": "Does Rahi support integration with other services or APIs?", "keywords": ["API", "integration", "public", "private", "partner"]},
    {"question": "How does Rahi ensure security in its platform design?", "keywords": ["security", "encryption", "authentication"]},
    {"question": "What cloud providers and database technologies does Rahi work with?", "keywords": ["AWS", "Azure", "GCP", "SQL", "NoSQL", "polyglot persistence"]},
    {"question": "Who is the CEO of Rahi Technologies and what is his background?", "keywords": ["CEO", "Rakesh Bhatt", "leadership"]},
    {"question": "Can you tell me about the leadership team at Rahi Technologies?", "keywords": ["CTO", "VP", "leadership", "experience"]},
    {"question": "Who is responsible for DevOps and engineering practices at Rahi?", "keywords": ["DevOps", "engineering", "Sagar Pandkar", "CI/CD"]},
    {"question": "What roles are currently open at Rahi Technologies?", "keywords": ["open positions", "Senior Software Engineer", "careers"]},
    {"question": "What is the work culture like at Rahi Technologies?", "keywords": ["culture", "inclusive", "ambition", "ownership"]},
    {"question": "How does Rahi empower employees in terms of ownership and career growth?", "keywords": ["employees", "ownership", "wealth", "impact"]},
    {"question": "What tools and practices are used by the DevOps team at Rahi?", "keywords": ["DevOps", "CI/CD", "Terraform", "Docker", "Kubernetes"]},
    {"question": "How does Rahi ensure compliance and security in its CI/CD pipeline?", "keywords": ["compliance", "security", "DPDP", "DLG", "pipeline"]},
]


def evaluate_answer(answer, expected_keywords):
    return sum(k.lower() in answer.lower() for k in expected_keywords) / len(expected_keywords)

def objective(trial):
    chunk_size = trial.suggest_categorical("chunk_size", [256, 512, 768])
    chunk_overlap = trial.suggest_int("chunk_overlap", 50, 200, step=50)
    top_k = trial.suggest_int("top_k", 3, 10)
    temperature = trial.suggest_float("temperature", 0.2, 0.7)
    max_tokens = trial.suggest_int("max_output_tokens", 512, 1024, step=128)

    async def run_preprocessing():
        return await preprocess_text(
            files=[],  # No files used for tuning
            size=chunk_size,
            overlap=chunk_overlap,
            scraped_data=sample_scraped_data
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
