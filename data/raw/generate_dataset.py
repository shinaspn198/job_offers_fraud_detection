"""
Generate a synthetic, balanced legit/fraud job-posting dataset.
No external download needed. Run: python generate_dataset.py
Output: combined_dataset.csv (text, label) 0=legit, 1=fraud
"""
import random
import pandas as pd

random.seed(42)

COMPANIES = ["TechNova Solutions", "BrightPath Consulting", "Orion Analytics", "Zenith Retail Group",
             "Clearwater Logistics", "Nimbus Cloud Systems", "Falcon Manufacturing", "Meridian Health Group",
             "Sterling Financial Partners", "Evergreen Media", "Quantum Robotics", "Harbor Point Bank",
             "BlueRock Construction", "Vertex Pharmaceuticals", "Golden Gate University", "PixelCraft Studios"]

ROLES = ["Software Engineer", "Marketing Intern", "Data Analyst", "Customer Support Executive",
         "HR Coordinator", "Graphic Designer", "Sales Associate", "Content Writer",
         "Accounts Assistant", "Operations Manager", "Business Development Intern", "Backend Developer",
         "Research Assistant", "Social Media Intern", "Warehouse Supervisor", "QA Tester"]

CITIES = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Pune", "Hyderabad", "Kochi", "Remote"]

LEGIT_TEMPLATES = [
    "{company} is hiring a {role} for our {city} office. Responsibilities include collaborating with cross-functional teams, delivering high-quality work, and meeting project deadlines. Minimum {exp} years of relevant experience required. We offer a competitive salary, health insurance, and paid time off. Interested candidates should apply through our careers portal at {company_lower}.com/careers with an updated resume.",
    "We are looking for a motivated {role} to join {company} in {city}. The ideal candidate has strong communication skills and a background in {field}. This is a full-time position with standard working hours. Please submit your application along with your resume and portfolio via our official website.",
    "{company} ({city}) is expanding its team and seeking a {role}. Duties include supporting daily operations, contributing to team goals, and reporting to the department head. Bachelor's degree preferred. Benefits include provident fund, medical insurance, and annual bonus. Apply via LinkedIn or our company careers page.",
    "Join {company} as a {role} in our growing {city} branch. You'll work alongside experienced professionals on real projects, receive structured onboarding, and have opportunities for growth within the company. No upfront fees of any kind are required to apply. Send your CV to careers@{company_lower}.com.",
    "{company} is conducting interviews for the position of {role}. Candidates will go through a structured process: resume screening, technical/HR interview, and offer discussion. Salary is as per industry standards and discussed transparently during the interview. Office located in {city}.",
]

FRAUD_TEMPLATES = [
    "URGENT HIRING!!! {role} needed immediately at {company}. Earn upto ${amt}/week working from home, NO EXPERIENCE NEEDED. Limited seats available, apply NOW. Contact us only on WhatsApp at +1-555-0199 to secure your spot. Small registration fee of $49 required to activate your account.",
    "Congratulations! You have been shortlisted for {role} at {company}. No interview required — direct joining! To confirm your offer, please pay a refundable security deposit of ${amt} via bank transfer. Reply on Telegram immediately as seats are limited.",
    "Work from home {role} position!! Guaranteed salary of ${amt}/month. Just pay a one-time processing fee of $30 to get your starter kit and login credentials. Contact HR only on WhatsApp, no calls. Hurry, offer closes today!!!",
    "{company} is offering instant job placement for {role}. 100% guaranteed job, no resume or interview needed. Send $25 registration fee to activate application and receive offer letter same day. Message us on WhatsApp for fast processing.",
    "EASY MONEY OPPORTUNITY! {role} needed urgently, earn ${amt} weekly, flexible hours, no skills required. Send your bank details and a small advance payment of $40 to reserve your position. Limited seats, act now before offer expires!!!",
]

FIELDS = ["marketing", "software development", "finance", "customer service", "data science", "design"]

def fill_legit():
    company = random.choice(COMPANIES)
    tpl = random.choice(LEGIT_TEMPLATES)
    return tpl.format(
        company=company,
        company_lower=company.lower().replace(" ", ""),
        role=random.choice(ROLES),
        city=random.choice(CITIES),
        exp=random.choice([1, 2, 3, 4, 5]),
        field=random.choice(FIELDS),
    )

def fill_fraud():
    company = random.choice(COMPANIES + ["Global Career Hub", "QuickJobs Inc", "WorkFromHome Direct", "EasyHire Now"])
    tpl = random.choice(FRAUD_TEMPLATES)
    return tpl.format(
        company=company,
        role=random.choice(ROLES),
        amt=random.choice([500, 800, 1000, 1500, 2000, 3000]),
    )

N = 700  # per class

legit_texts = list({fill_legit() for _ in range(N * 2)})[:N]
fraud_texts = list({fill_fraud() for _ in range(N * 2)})[:N]

rows = [(t, 0) for t in legit_texts] + [(t, 1) for t in fraud_texts]
random.shuffle(rows)

df = pd.DataFrame(rows, columns=["text", "label"])
df.to_csv("combined_dataset.csv", index=False)
print("Saved combined_dataset.csv | rows:", len(df))
print(df["label"].value_counts())
