import optuna
import asyncio
from preprocess import preprocess_text
from inference import get_response
from langchain.schema import Document as LangchainDocument

# 🔸 Sample input data (no need for files during tuning)
sample_scraped_data = """
Kunjesh Kamleshbhai Parekh (Web Profile: https://kunjeshweb.vercel.app) Current Location: Pune Email: kunjeshparekh90@gmail.com
Phone: +91 7984885953
Total Experience 12+ Years
 Full stack data scientist and Team Lead with experience of developing & deploying Machine Learning, Deep Learning,
Reinforcement Learning and Large language Models (LLMs)
 Converting Business Problems into data-driven ML projects
 Finance, NBFC (Non - Banking Financial Company), Pharmaceutical, IT and Manufacturing Domains Experience
 Dale Carnegie certified trainer – training industry individuals and students for various GenAI, ML and Data Science courses
 GenAI Project Deployment Link:
○ Live Chatbot on: https://datalysis-website.vercel.app (Deployed chatbot for datalysis – a training institute)
○ RAG based auto configurable chatbot – can scrape any website and document
○ Requires less than 6hrs to deploy for any website
○ Easy deployment using NPM package – built on react base, deployed on EC2
 Programming Skills / Languages /Tools: Python, PySpark, R, Excel (Advanced + VBA), SQL, Databricks
 5+ years of team lead experience
 GITHub Profile (Only a handful of projects mentioned): https://github.com/kunjesh90
 ML Competitions (Completed 50+ Competitions on Analytics Vidhya, Techgig, Hacker Rank, Kaggle, Driven Data etc.).
Highlights below:
o Hateful Meme Prediction (A competition by Facebook on driven data – Ranked in top 10%) [Text+Image Classification]
o Secured All India Rank 7 in the Data Science Competition (A PD modelling competition) held by GE Analytics on techgig
o Twitter Sentiment Analysis Case on Analytics Vidhya with Secured Rank in top 5%
o Completed Mechanism of Action competition on Kaggle with Mean Log Loss 0.01798 while the Rank 1 had 0.01599
o Fashion MNIST competition on analytics Vidhya
o AmExpert 2019 – Machine Learning Hackathon: Secured Rank in top 30%
o HR Analytics Case on Analytics Vidhya - Secured Rank in top 10%
Designation: Head of Data Science & Engineering July’23 – till date
Designation: Deputy National Lead (Data Science & Engineering) Sep’21 – June’23 (22 Months)
Designation: Senior Lead (Data Science & Engineering) Mar’20 – Aug’21 (18 Months)
 Leading a team of 50+ Data Scientists, Data Engineers, and BI Analysts to deliver various performance metrics for different
products of Bajaj in the digital commerce space
 Working on Click Stream - User Behavior Data Analysis of Bajaj Finserv super app and Bajaj Mobikwik app
 Creating solutions to pitch the right product to the right customers to minimize the risk
 Dealing with business stakeholders to understand their dashboarding requirements for various KPIs
 Created HEART Metrics (Happiness, Engagement, Adoption, Retention & Task Success) to gauge the performance of Bajaj
Finserv super app
 Working on strategies to improve the MAU, DAU, customer retention on the app
Project 1: RAG based chatbot for customer queries related to Fixed Deposits of Bajaj Finance
 Created RAG-based chatbot answering customer queries related to fixed deposits of Bajaj Finance
 Vector databases used: FAISS, ChromaDB
 Base LLM: LLaMA70B and chatgpt APIs using Langchain
 Deployed backend on AWS EC2 as well as Render
Project 2: LLM for vernacular translation
 Used M2M100, and MarianMT models for vernacular translation tasks of FAQs and T&Cs for Bajaj Finserv app
 Achieved BLEU score of > 0.8 for all the translation tasks, used mean win rates to select models for different sections
Project 3: Bajaj Finserv app & Web Personalization
 Identify opportunities on the Bajaj Finserv app for personalization
 Built and productionized ML models which have resulted in on asset click rate improvement by 30% to 90% for the
personalized sections on the app – icons, banners, nudges, notifications, etc.
 Achieved INR 7000 Mn+ of incremental revenue from personalizing various properties on app and web in calendar year 2024
 Recommended data driven-app design to optimize user journey on the app
Project 4: A/B testing
 Standardized A/B testing frame to drive the data driven decision making which is statistically significant
 End-to-end platform creation from test case design to reporting for A/B test
 The output of the test is used to scale up on production build / performance marketing activities
Project 5: Recommendations for Personal Loan & Credit Card
 Understand customer-level data (demographic, bureau, in app behavior) and Recommend the Personal Loan (PL) and Credit
Card (CC) offers accordingly
 Developed decision tree algorithm to check the customer affinity for PL & CC products and built the recommendation model
on the significant variables
 Metric Improved: Click Through Rate (CTR) improved to 4-4.5% as compared to 1% for the PL and CC product campaigns
Project 6: Payments Transactions
 Built recommendation models to nudge the customer for various BBPS, UPI, PPI transactions on the app
 Scaled up the BBPS transactions from 8K/month to 2MM/month
 Helped in achieving 3MM+ UPI handles
Project 7: Data Architecture Design for Events on Clevertap to store the Super app Data
 Created a framework to effectively identify events and event properties of the super app sections
 Stored the data effectively within the constraint of only 512 events with 256 event properties
Other Projects : Face match (Image processing using AI) , Playstore Ratings and Reviews Classification (NLP) and Profile Matching,
Geographic location mapping for skip tracing, Offer pool enhancement and risk score card development using SDK data, Data Mart &
Feature Mart Design, Fine-tuned LLM (LLaMA 2) & FLAN-T5 using PEFT mechanism (LoRA) for call center chat bot – prompt
engineering, Worked on prompt-to-prompt diffusion model and stable diffusion model for image processing tasks
Rewards & Recognitions
 Received Super-Heroes one of the highest ranked awards in Bajaj Finance
 Received special award from Chief Business Officer for the excellent performance
 Received Data Science Star contributor award for streamlining the recruitment process of Data Scientists
PROFESSIONAL EXPERIENCE
Merck & Co.’s Global Center for Analytics & Forecasting - Aspect Ratio, Pune India September 2016 – February 2020 (42 Months)
Designation: Team Lead (Data Science & Analytics) Designation: Senior Analyst (April ’18- Feb’20) (23 Months)
(September ’16 – March’18) (19 Months)
 Led a team of 26 analysts for various Data Science and Data Analytics Projects of Merck (One of the world’s Largest Pharmaceutical
Company).
 Led Data Science team which builds forecasting models for key immuno-oncology (anti PD-1) product and vaccines of Merck in
the pipeline.
 Worked on predictive analytics that involves an end to end cycle to create the forecast.
 Developed forecasting models for demand and revenue forecasting, covering statistical trending, event modeling and reporting.
The models are built in Excel using python backend with exhaustive VBA and advanced excel formulas
 Used data science techniques for various statistical analysis (ex., Hypothesis testing, Regression analysis, Cluster analysis, CART
analysis, Random forest, Deep Neural Networks, CNN, RNN, text analytics, Gradient Boosting, Cross Validation techniques etc.)
 Used python (numpy, scikitlearn, matplotlib, pandas, tensorflow) for various statistical analysis and preparing algorithms to solve
specific problems related to image processing and logistic regression.
 Worked on @Risk professionals software for creating Monte Carlo Simulations.
 Developed various customized excel and spotfire (Data visualization tool) reports as per the requirements.
 Created monthly demand forecasts and quarterly revenue forecasts to plan for marketing interventions.
Achievements
 Received the excellent performer’s award thrice from Merck for leading and developing the forecasting models.
 Received one-star award for optimizing the prediction model and reducing the cycle time.
 Data Camp and EDx Certified: Python for Data Science, R for Data Science, pyspark,SQL for data science
Projects
Forecasting Model Development:
 Led a team of 19+ individuals to design and develop the calculations heavy oncology forecast models on web with calculation
backend in python, assumptions input in excel as well as on web and the reporting on web as well as in excel.
 Led the team to create a central SQL database management for storing the forecast of every cycle and create the region wide
reports directly from the database on Spotfire.
 Led the team to develop desktop based model applications with frontend in Electron JS,R-Shiny and a backend in Python and R.
Text Analytics:
 Designed a Shiny Web app in R for text analytics.
 The app was able to create summaries of large Decision Research reports using natural language processing in R.
 The report may vary up to 1000+ pages which can be made abridged to 1 page using text analytics algorithm developed
independently
 Sentiment analysis about various drugs of Merck.
Market Share & Revenue Prediction:
 Used various data science techniques (Regression analysis, Cluster analysis, CART analysis, SVM, Time Series Analysis, Neural
Networks etc.) to predict the market shares & Revenues of various Merck products (especially oncology & vaccine) in different
geographies through machine learning algorithms.
Link Clinical Trials to Revenue:
 Merck has 500+ Clinical Trials going on for its key immuno-oncology product in mono + combination therapies.
 Designed a framework to accurately forecast the revenue for next 5 years (for budgeting/resource allocation purpose) by keeping
the probable outcomes of the clinical trials
Image Processing:
 Predict the tumor at initial stage through the image processing, which will make the physical observations of the MRI redundant
and the tumor prediction will be more precise and will be detected at the initial stage through the ML algorithms. The algorithm
is designed on tensorflow (keras) with deep learning techniques (CNN+ Deep Neural Network).
Freelance Data Science Trainer
 Working as Freelance Data Science Course teacher at a couple of data science training institutes (upGrad, IMS Proschool) and
colleges of Pune.
 Awarded Best Mentor by upGrad for the Oct -Dec ’21 quarter
Programming Languages: html, css, js, react, C, C++
ACADEMICS
Degree Institute / University / Board Year %/CGPA
PhD (Artificial Intelligence) Indian Institute of Technology, Jodhpur (School of AI) 2023-2027 8.00 *Pursuing
MBA (Finance & Operations) Department of Management Studies, IIT Roorkee 2013-15 8.46
CAT Indian Institute of Management (IIM) 2012 98.99 %tile
B.Tech. (Electrical Engineer) Institute of Technology,NirmaUniversity,Ahmedabad 2007-2011 8.09
Class XII Sett R.J.J.HighSchool,Navsari,Gujarat Board 2006-2007 91.33%
Class X Sett R.J.J.HighSchool,Navsari,Gujarat Board 2004-2005 92.14%
 Teaching Statistics & Analytics concepts (ex., Hypothesis testing, Regression analysis, Cluster analysis, CART analysis Naive
Bayesian, Time Series Analysis, Deep Learning, tensorflow, keras etc.) and their application using Python and R programming.
 Teaching how to develop machine learning algorithms
 Chegg Subject Matter Expert for Statistics
PROFESSIONAL EXPERIENCE
Cognizant Technologies and Solutions, India. (Pune) May 2015 – August 2016 (16 Months)
Designation: Business Analyst & Data Analytics
Actively supported all the opportunities in the North America, Asia Pacific, Middle East, Europe and India. Supported the pre-sales
functions of the Communications & Technology (ComTech) and Government Vertical (SBUs namely Online, Hi-Tech, ISV and RoW).
Interacting with Client Partners, Account Mangers and other Horizontal Point of Contacts across all the geographies.
Roles and Responsibilities
Marketing Analytics: Applying statistical techniques (namely Regression, Logistic Regression, CART analysis etc.) for targeting right
customers and cross selling.
 Worked with the Big Data Analytics team to improve the prediction efficiency of the algorithms
 Running the story point estimation sessions for the agile projects to create sprint & release planning.
 Responding the RFP (Request for Proposal), RFI (Request for Information), RFQ (Request for Quotation) etc. as a bid owner.
 Understand the proposal requirements, preparing the bid plan and Solution review presentations.
 Understanding various financial pricing models and provide the most competitive budgetary quote to the client to ensure a
winning bid.
 Preparing internal account review research reports of the existing clients by understanding their existing landscape and targeting
their pain points for account mining.
 Preparing the Capability & Defense Presentations for the clients for various connects.
 Cognizant Academy Certified expert in the Agile Story Point Estimation
SUMMER INTERNSHIP
Toshniwal Equity Services Pvt Ltd, Mumbai,India May 2014-July 2014 (2 Months)
Roles and Responsibilities:
 Developed an algorithm using META Stock Software for the technical analysis of the stocks for short term buy-sell signals and
achieved an overall accuracy of more than 60% in daily stocks prediction.
 Fundamental analysis of equities and their long-term prediction.
 Financial statements analysis of the companies to understand the financial ratios, cash flows, fund flows
 Prepared industry reports and company reports for investors.
 Understanding derivatives & valuation of options by Black-Scholes Model and trading in derivatives using various strategies.
PROFESSIONAL EXPERIENCE
Jubilant Life Sciences Limited at Bharuch, India. July 2011-June 2013 (24 Months)
Designation: Senior Engineer
Roles and Responsibilities:
 Real time Electrical Power Trading was handed based upon the live electricity market price and demand
 Worked in the Green Field Project where project planning, installation, commissioning and operations & Maintenance of the
electrical equipment were being handled
 Transmission and distribution of the electricity within the Captive Power Plant was being handled
 Handled various electrical liaising issues and approvals related to SEZ (Special Economic Zone)
Software Skills
MS Excel: Advance Excel, Macros, VBA Coding and form controls
R, Python, Pyspark: Can handle large data sets in R & Python for statistical analysis (ex., Regression analysis, Cluster analysis, CART
analysis, Deep Learning etc.), FastAPI
Deployment: Amazon EC2, render, GCP (Google Cloud Platform), heroku, vercel, docker
Azure Data Bricks: Processing big data using pyspark with effective storing in HIVE
SQL: Elementary knowledge
MS Power Point: Prepare corporate presentations for clients, Create fancy templates
ACADEMIC ACHIEVEMENTS
 Got the 1st prize in “CORPOSTRAT” a Data Analytics and Financial Valuation case study competition organized in the annual
tech fest Cognizance’14 at IIT Roorkee in 2014.
 Secured All India Rank 5 in “Wealth Management” examination conducted by FLIP (Finitiatives Learning India Pvt. Ltd) 2013.
 Underwent training on Private Equity, Firm and Equity Valuations, Mergers & Acquisitions, Structured Finance and Issue
 FLIP certified Treasury and Capital Markets analyst.
POSITIONS OR RESPONSIBILITIES
 Worked as the President of “Vittarth the finance club of DoMS IIT Roorkee”: actively organized events & knowledge sessions
 Founder of a club named “PlanB” at the IIT Roorkee level for sharing the business knowledge through events, class sessions,
magazines, guest lectures, seminars etc. for MBA and B.Tech students.
 Started a monthly business magazine with the coordination of Vittarth and PlanB members.
 Led a financial event named “CORPORATA” in the annual cultural festival (Thomso-2013) of IIT Roorkee.
EXTRA CURRICULAR ACHIEVEMENTS
 Chess champion of the DoMS IIT Roorkee in the annual sports day 2014-15.
 Chess Champion of the Nirma University in the annual sports day 2010-11.
 1st Runner up in the Chess Competition held at Cognizant Technologies and Solutions at Chennai in 2015.
 1st Rank in the District level Quiz competition “GUJCOST” in 2006.
"""

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
