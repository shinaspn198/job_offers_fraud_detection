# 🛡️ Smart Internship & Job Fraud Detection AI

An AI-powered backend system designed to detect potentially fraudulent internship and job opportunities using **Machine Learning, Natural Language Processing (NLP), OCR, explainable AI, and Retrieval-Augmented Generation (RAG)**.

The system analyzes job descriptions, internship offers, and uploaded documents to identify suspicious patterns and provide an understandable fraud assessment.

---

## 🚀 Project Overview

Online job and internship fraud has become increasingly common, with fraudulent offers often containing patterns such as:

* Registration or processing fees
* Security deposits
* Requests for bank details
* Guaranteed job or placement claims
* Suspicious WhatsApp/Telegram communication
* Unrealistic salary or stipend promises
* Urgent payment requests

This project combines **machine learning and AI-based analysis** to identify these patterns and help users evaluate suspicious opportunities.

---

## ✨ Key Features

### 🤖 Machine Learning Fraud Detection

Uses a trained **Scikit-learn classification model** to classify job/internship content as:

* **Legitimate**
* **Fraudulent**

The model uses **TF-IDF/NLP features and engineered fraud-related features** to improve detection.

### 🔍 Fraud Pattern Detection

The system analyzes suspicious indicators such as:

* Registration fee
* Processing fee
* Security deposit
* Bank details
* Guaranteed job
* Guaranteed placement
* WhatsApp
* Telegram
* Fraud-related keywords
* Suspicious salary/stipend patterns
* Excessive uppercase text
* Excessive exclamation marks

### 📄 OCR Document Analysis

Users can analyze internship or job offer documents.

The system supports document text extraction using:

* **Tesseract OCR**
* **pdfplumber**

This allows text to be extracted from uploaded documents before fraud analysis.

### 🧠 Explainable AI

Instead of returning only a prediction, the backend can identify suspicious indicators contributing to the fraud assessment.

This makes the prediction easier for users to understand.

### 💬 RAG-Based AI Analysis

The project includes a **Retrieval-Augmented Generation (RAG)** component for working with stored job/internship information.

The RAG pipeline uses:

* Sentence Transformers
* Embeddings
* Cosine similarity
* Retrieved relevant information
* LLM-based response generation

### ⚡ FastAPI Backend

The application is built using **FastAPI**, providing a lightweight and scalable API architecture.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │       User           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Job / Internship   │
                    │      Information     │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │   Text Input    │         │ Document Upload │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 │                           ▼
                 │                  ┌─────────────────┐
                 │                  │ OCR / PDF Text  │
                 │                  │    Extraction   │
                 │                  └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering │
                    │      + NLP           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ ML Fraud Detection   │
                    │      Model           │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌─────────────────┐    ┌─────────────────┐
          │ Fraud Prediction│    │ Explainability  │
          └────────┬────────┘    └────────┬────────┘
                   │                      │
                   └──────────┬───────────┘
                              ▼
                    ┌──────────────────────┐
                    │   RAG / AI Analysis  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Final Result    │
                    │  + Risk Explanation  │
                    └──────────────────────┘
```

---

# 🧠 Machine Learning Pipeline

The ML pipeline follows these major steps:

```text
Raw Dataset
     ↓
Data Preparation
     ↓
Text Processing
     ↓
TF-IDF Vectorization
     ↓
Fraud Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Saved Model
     ↓
FastAPI Prediction
```

The system combines textual information with engineered fraud indicators rather than depending only on individual keywords.

---

# 📊 Feature Engineering

The model uses features representing suspicious patterns in job and internship content.

Examples include:

| Feature             | Purpose                                    |
| ------------------- | ------------------------------------------ |
| Fraud Keyword Count | Measures suspicious terminology            |
| Uppercase Count     | Detects unusual emphasis                   |
| Exclamation Count   | Detects excessive punctuation              |
| WhatsApp            | Detects suspicious communication channels  |
| Telegram            | Detects suspicious communication channels  |
| Registration Fee    | Detects payment requests                   |
| Processing Fee      | Detects suspicious charges                 |
| Security Deposit    | Detects deposit requests                   |
| Guaranteed Job      | Detects unrealistic guarantees             |
| Bank Details        | Detects financial information requests     |
| Salary/Stipend      | Provides additional contextual information |

---

# 📁 Project Structure

```text
fraud-detector/
│
├── backend/
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── features.py
│   │   ├── explain.py
│   │   ├── extract.py
│   │   └── rag.py
│   │
│   ├── data/
│   │   ├── build_dataset.py
│   │   ├── train_model.py
│   │   ├── combined_dataset.csv
│   │   │
│   │   └── raw/
│   │       └── generate_dataset.py
│   │
│   ├── generate_dataset.py
│   ├── requirements.txt
│   └── .gitignore
│
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* Scikit-learn
* TF-IDF
* Logistic Regression
* Feature Engineering

### NLP

* Natural Language Processing
* Text Vectorization
* Sentence Transformers
* Cosine Similarity

### OCR & Document Processing

* Tesseract OCR
* pytesseract
* pdfplumber

### Generative AI

* RAG
* LLM integration
* Embeddings

### Development Tools

* Git
* GitHub
* PowerShell
* Virtual Environment

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detector.git
```

Move into the backend directory:

```bash
cd fraud-detector/backend
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\activate
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# ▶️ Running the Backend

From the `backend` directory:

```powershell
python -m uvicorn app.main:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

---

# 📚 API Documentation

FastAPI automatically provides interactive API documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

Swagger UI can be used to test the API endpoints directly from the browser.

---

# 🔐 Environment Variables

Sensitive API keys should **never be committed to GitHub**.

Create a `.env` file when required:

```env
GROQ_API_KEY=your_api_key_here
```

The `.env` file should remain inside `.gitignore`.

Never publish real API keys in:

* GitHub repositories
* README files
* Source code
* Screenshots
* Public documentation

---

# 🧪 Testing the System

After starting FastAPI, open:

```text
http://127.0.0.1:8000/docs
```

Use the Swagger interface to test the available API endpoints.

Example suspicious content:

```text
Congratulations!

You have been selected for a work-from-home internship.

Please pay a registration fee of ₹2000
and a security deposit to confirm your position.

Contact us through WhatsApp immediately.
```

The system analyzes the content and identifies relevant fraud indicators.

---

# 🔎 Example Fraud Indicators

```text
Registration Fee
Security Deposit
Bank Details
Guaranteed Job
WhatsApp
Processing Fee
```

These indicators are combined with the trained ML model rather than relying on a single keyword.

---

# 🧠 RAG Workflow

The RAG component follows this workflow:

```text
Job / Internship Data
        ↓
Text Processing
        ↓
Sentence Transformer
        ↓
Vector Embeddings
        ↓
Similarity Search
        ↓
Relevant Information
        ↓
LLM
        ↓
AI Response
```

This allows the system to retrieve relevant stored information before generating an AI response.

---

# 🎯 Project Goals

The main goals of this project are to:

* Detect suspicious internship and job opportunities
* Reduce the risk of users falling for fraudulent offers
* Analyze uploaded offer documents
* Extract text automatically using OCR
* Provide understandable fraud explanations
* Combine traditional ML with modern Generative AI
* Provide a scalable FastAPI backend
* Build a practical AI application for real-world use

---

# 🚀 Future Improvements

Planned improvements can include:

* Advanced transformer-based classification
* Better multilingual fraud detection
* Improved OCR accuracy
* URL and domain reputation analysis
* Company verification
* Email metadata analysis
* Real-time fraud intelligence
* Advanced RAG memory
* User authentication
* Database integration
* Production deployment
* Continuous model retraining
* Improved false-positive handling

---

# 📌 Disclaimer

This system provides an AI-assisted assessment of potentially fraudulent job and internship opportunities.

A prediction should not be treated as definitive proof that an opportunity is fraudulent or legitimate. Users should independently verify companies, recruiters, contact information, payment requests, and official sources before taking action.

---

# 👨‍💻 Developer

**Shinas PN**

BCA — Artificial Intelligence Specialization

Aspiring AI Engineer

---

# ⭐ Project Highlights

```text
Python
       +
Machine Learning
       +
NLP
       +
OCR
       +
Explainable AI
       +
RAG
       +
Generative AI
       +
FastAPI
       =
Smart Internship & Job Fraud Detection AI
```

If you find this project useful, consider giving the repository a ⭐ on GitHub.
