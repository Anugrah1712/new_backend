import optuna
import asyncio
from preprocess import preprocess_text
from inference import get_response
from langchain.schema import Document as LangchainDocument

# 🔸 Sample input data (no need for files during tuning)
sample_scraped_data = """
ANALYTICS
DATA
SCIENCE
EDUCATION
_
_
_
Company Overview
Datalysis is an education provider specializing in Analytics and Data Science education. The
company offers courses designed to meet the growing demand for skilled professionals in data
analysis, machine learning, artificial intelligence, and big data. Datalysis empowers students,
working professionals, and career changers with tools and knowledge needed for today's
data-driven world. Their curriculum balances theoretical understanding with hands-on
application, preparing learners to solve real-world problems and build industry-relevant
expertise.To enroll/register with the Datalysis Program or courses, interested individuals can
apply on the Datalysis website or email their request to datalysis.india@gmail.com.
Data Analytics Program
The Data Analytics program at Datalysis focuses on equipping learners with skills to analyze,
interpret, and visualize data. Participants learn to work with Excel, SQL, and Power BI, gaining a
strong foundation in descriptive and inferential statistics. The program teaches data cleaning
and preprocessing techniques, creating dashboards using Power BI, writing efficient SQL
queries, exploratory data analysis and reporting, and business problem solving through
analytics. This program is designed for students new to data, business professionals aiming to
enhance decision-making, and individuals interested in data-driven roles. The next batch starts
from June 1st, with interested parties encouraged to contact Datalysis for more information.To
enroll with the Datalysis Program or our courses, interested individuals can apply on the
Datalysis website or email their request to datalysis.india@gmail.com.
Machine Learning Program
Datalysis offers a Machine Learning course that covers supervised and unsupervised learning
algorithms, model evaluation techniques, and real-world applications of machine learning. The
course utilizes Python and libraries like Scikit-learn throughout the curriculum. Students learn
linear and logistic regression, decision trees and ensemble methods, clustering algorithms and
dimensionality reduction, model evaluation and optimization techniques, and an introduction to
deep learning concepts. This program is ideal for learners with basic Python knowledge looking
to expand into AI and predictive modeling. The next batch starts from June 1st, with contact
information available for interested participants.To enroll with the Datalysis Program or our
courses, interested individuals can apply on the Datalysis website or email their request to
datalysis.india@gmail.com.
Python for Data Science Program
The Python for Data Science course at Datalysis is beginner-friendly and introduces Python
from the ground up, focusing on its application in data science. Learners gain proficiency in
using Python for data manipulation, visualization, and simple analysis tasks. The curriculum
covers Python fundamentals including data types, loops, and functions, data manipulation with
Pandas, numerical computations using NumPy, data visualization with Matplotlib and Seaborn,
and writing clean, efficient, and readable code. This program is suitable for beginners in
programming or professionals transitioning into data science roles. The next batch begins on
June 1st, with contact details available for more information.To enroll with the Datalysis Program
or our courses, interested individuals can apply on the Datalysis website or email their request
to datalysis.india@gmail.com.
Business Intelligence Tools Program
The Business Intelligence Tools program teaches students to extract insights from raw data
using industry-standard tools like Tableau and Power BI. This course emphasizes building
dashboards, data models, and visual stories that support business decision-making.
Participants learn connecting and transforming data sources, building interactive dashboards in
Power BI and Tableau, using DAX and calculated fields, applying BI to real-world business use
cases, and visual best practices and storytelling with data. The program is designed for
professionals in business, marketing, sales, or operations who want to leverage data for smarter
decision-making. The next batch starts from June 1st, with interested individuals encouraged to
contact Datalysis for more information.To enroll with the Datalysis Program or our courses,
interested individuals can apply on the Datalysis website or email their request to
datalysis.india@gmail.com.
AI & Big Data Technologies Program
Datalysis provides an advanced-level AI & Big Data Technologies course that explores the
intersection of artificial intelligence and big data systems. Students learn about AI concepts
along with tools and platforms that support large-scale data processing. The curriculum covers
fundamentals of AI and its real-world applications, introduction to Natural Language Processing
and Computer Vision, working with Hadoop, Spark, and cloud data ecosystems, building
scalable machine learning pipelines, and ethics and governance in AI and big data projects.
This program is designed for experienced data professionals, software engineers, and tech
enthusiasts seeking advanced knowledge in AI and big data infrastructure. The next batch
begins on June 1st, with contact information available for further details.To enroll with the
Datalysis Program or our courses, interested individuals can apply on the Datalysis website or
email their request to datalysis.india@gmail.com.
Target Audience
Datalysis courses are designed to accommodate a wide range of learners including university
students looking to build a strong foundation in data science and AI, working professionals
seeking upskilling or transitioning into analytical roles, and career changers eager to enter the
fast-growing tech and data sectors.
Learning Experience
The learning experience at Datalysis prioritizes engagement and practical application. Students
work on real-world projects solving challenges using industry datasets across domains like
finance, healthcare, retail, and technology. The programs feature expert mentorship from
experienced instructors with deep domain expertise. Datalysis offers flexible learning modes
allowing students to choose from classroom-based, live online, or hybrid formats. Additionally,
students receive career support including resume building, mock interviews, and job placement
assistance.
CONSULTANCY
SERVICES
_
Consultancy Services Overview
Datalysis offers end-to-end data science consultancy services tailored to business needs,
extending beyond their education and training programs. Their team of experts collaborates with
organizations to solve complex problems, drive informed decision-making, and unlock new
opportunities through data. The company delivers actionable, scalable, and measurable
solutions that align with clients' strategic objectives, ranging from building predictive models to
designing enterprise-grade BI dashboards.To enroll with the Datalysis Program or our courses,
interested individuals can apply on the Datalysis website or email their request to
datalysis.india@gmail.com.
Data Strategy Consulting
Datalysis provides Data Strategy Consulting to help organizations develop a long-term vision
and roadmap for becoming truly data-driven. Their consultants work with leadership and IT
teams to assess current capabilities, identify opportunities, and implement data governance
frameworks. Deliverables include data maturity assessment and gap analysis, roadmap for data
infrastructure and tooling, data governance and compliance frameworks, and strategic planning
for analytics initiatives. This service is ideal for businesses in early stages of digital
transformation or planning data-driven innovation. The next engagement starts from June 1st,
with interested organizations encouraged to contact Datalysis for more information.To enroll with
the Datalysis Program or our courses, interested individuals can apply on the Datalysis website
or email their request to datalysis.india@gmail.com.
Predictive Analytics Projects
The Predictive Analytics Projects service at Datalysis focuses on designing and implementing
machine learning models tailored to specific business challenges, such as customer churn
prediction, sales forecasting, or fraud detection. The service includes business use case
scoping, model development and training using real data, KPI-based model evaluation and
optimization, and deployment-ready model handoff. This offering is targeted at organizations
looking to implement or scale predictive capabilities across departments. The next project cycle
begins June 1st, with contact information available for interested parties.To enroll with the
Datalysis Program or our courses, interested individuals can apply on the Datalysis website or
email their request to datalysis.india@gmail.com.
Business Intelligence Dashboards
Datalysis offers Business Intelligence Dashboard services where their BI experts build
interactive, easy-to-navigate dashboards that transform raw data into meaningful insights. They
help clients identify key performance metrics and visualize them clearly to support faster, better
decision-making. The service includes dashboard design using Power BI or Tableau, data
source integration and automation, user-friendly visual interfaces, and executive-level reporting
with drill-down capabilities. This service is designed for managers and executives seeking
real-time visibility into operations and performance. The next service cycle starts June 1st, with
interested organizations encouraged to contact Datalysis for more details.To enroll with the
Datalysis Program or our courses, interested individuals can apply on the Datalysis website or
email their request to datalysis.india@gmail.com.
Model Deployment & Evaluation
The Model Deployment & Evaluation service at Datalysis ensures that models are deployed
securely and monitored for performance in real business environments. They offer complete
model lifecycle support from testing to scaling. The service includes API development for model
integration, cloud/on-premise deployment (AWS, Azure, GCP), performance monitoring and
model retraining, and security and version control best practices. This offering is ideal for
companies with existing data science initiatives needing support for operationalization. The next
service cycle begins June 1st, with contact information available for interested organizations.To
enroll with the Datalysis Program or our courses, interested individuals can apply on the
Datalysis website or email their request to datalysis.india@gmail.com.
End-to-End Analytics Solutions
Datalysis provides comprehensive End-to-End Analytics Solutions, covering the entire spectrum
from data ingestion to insight delivery. Whether clients are launching a new data platform or
revamping their reporting pipeline, Datalysis delivers solutions that are robust, secure, and
business-ready. Deliverables include data pipeline architecture and implementation, ETL/ELT
process design and automation, custom analytics applications and dashboards, and continuous
support and optimization. This service is designed for enterprises seeking a full-service partner
to design, build, and manage their analytics ecosystem. The next service cycle starts June 1st,
with interested organizations encouraged to contact Datalysis for additional information.To enroll
with the Datalysis Program or our courses, interested individuals can apply on the Datalysis
website or email their request to datalysis.india@gmail.com.
Partnership Benefits
Organizations partnering with Datalysis benefit from cross-industry expertise spanning finance,
retail, healthcare, and tech sectors, with domain knowledge applied to every project. Datalysis
builds scalable solutions that grow with clients' business needs and employs a collaborative
approach featuring transparent communication and co-creation with internal teams. The
company maintains an outcome-oriented delivery methodology, ensuring measurable impact
aligned with clients' business KPIs.
CORPORATE
TRAINING
_
Corporate Training Overview
Datalysis partners with organizations to provide high-impact corporate training solutions that
align with business goals and evolving industry needs. Their programs are designed to help
companies build internal capabilities, upskill their workforce, and foster a data-driven culture
across departments. Datalysis understands that every organization has unique challenges, skill
gaps, and strategic priorities, which is why they offer customized, hands-on training modules
tailored to the specific requirements of each team.To enroll with the Datalysis Program or our
courses, interested individuals can apply on the Datalysis website or email their request to
datalysis.india@gmail.com.
"""

# 🔸 Simple evaluation: check if important keywords are present
test_cases = [
    {
        "question": "What is Datalysis and what does it offer?",
        "keywords": ["Datalysis", "education provider", "Analytics", "Data Science", "courses"]
    },
    {
        "question": "How can I enroll in a Datalysis course?",
        "keywords": ["enroll", "register", "Datalysis website", "email", "datalysis.india@gmail.com"]
    },
    {
        "question": "Who should join Datalysis programs?",
        "keywords": ["target audience", "students", "working professionals", "career changers", "data roles"]
    },
    {
        "question": "What tools and topics are covered in the Data Analytics Program?",
        "keywords": ["Data Analytics", "Excel", "SQL", "Power BI", "statistics", "data cleaning"]
    },
    {
        "question": "What is taught in the Machine Learning Program?",
        "keywords": ["Machine Learning", "supervised learning", "unsupervised learning", "Python", "Scikit-learn"]
    },
    {
        "question": "Is any prior experience required for the Python for Data Science course?",
        "keywords": ["Python for Data Science", "beginner-friendly", "programming basics", "Pandas", "NumPy"]
    },
    {
        "question": "What does the Business Intelligence Tools Program include?",
        "keywords": ["Business Intelligence", "Tableau", "Power BI", "dashboards", "DAX", "data visualization"]
    },
    {
        "question": "What will I learn in the AI & Big Data Technologies course?",
        "keywords": ["AI", "Big Data", "NLP", "Computer Vision", "Hadoop", "Spark", "cloud"]
    },
    {
        "question": "When do the upcoming batches begin?",
        "keywords": ["batches", "start date", "June 1st", "enrollment", "course schedule"]
    },
    {
        "question": "What are the available learning modes at Datalysis?",
        "keywords": ["learning modes", "classroom", "live online", "hybrid", "flexible"]
    },
    {
        "question": "Do the programs include hands-on projects?",
        "keywords": ["projects", "practical", "real-world datasets", "finance", "healthcare", "technology"]
    },
    {
        "question": "What career support does Datalysis offer?",
        "keywords": ["career support", "resume building", "mock interviews", "job placement", "mentorship"]
    },
    {
        "question": "What consultancy services does Datalysis provide?",
        "keywords": ["consultancy", "predictive models", "BI dashboards", "data-driven decisions", "business solutions"]
    },
    {
        "question": "What is included in the Data Strategy Consulting service?",
        "keywords": ["data strategy", "maturity assessment", "roadmap", "governance", "compliance"]
    },
    {
        "question": "How can organizations or learners contact Datalysis for more details?",
        "keywords": ["contact", "website", "email", "datalysis.india@gmail.com", "information"]
    },
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
