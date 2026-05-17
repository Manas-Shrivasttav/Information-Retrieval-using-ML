# 🔍 Information Retrieval using Machine Learning
### Root Cause Analysis of Public Grievances for the Government of India

[![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-Topic%20Modelling%20%7C%20Sentiment%20Analysis-blue?style=flat)]()
[![Method](https://img.shields.io/badge/Method-LDA%20Mallet%20%7C%20VADER-orange?style=flat)]()
[![Organisation](https://img.shields.io/badge/Organisation-Quality%20Council%20of%20India-green?style=flat)](https://qcin.org)
[![Data](https://img.shields.io/badge/Data-Confidential%20(Govt.%20of%20India)-red?style=flat)]()

---

## ⚠️ Data Confidentiality Notice

The dataset used in this project was sourced from the **National Informatics Centre (NIC)** via the **CPGRAMS portal** (Centralised Public Grievance Redress and Monitoring System). It contains grievance records across 21 Government Departments and Ministries of India.

This data is **not included in this repository** as it is government-owned and confidential. The code is shared purely as a **methodology demonstration**. 

The notebook is fully reusable for any text corpus where root cause extraction is needed — not limited to government grievances. It can be adapted for customer complaints, support tickets, or survey responses.

---

## 🧠 Project Overview

Every month, thousands of citizens file grievances against government departments through CPGRAMS — but manually reading through lakhs of rows of unstructured text to find root causes is impossible at scale.

This project applied **unsupervised NLP and machine learning** to automate that process — using **LDA Mallet topic modelling** and **VADER sentiment analysis** to extract root causes, filter genuine grievances, and surface actionable intelligence for government policymakers.

The pipeline was applied to grievance data across **21 Departments/Ministries**. The numbers from the code output and findings mentioned anywhere in this repository are of one case/department.

---

## 🎯 Objectives

- Automatically identify root causes of citizen grievances using topic modelling
- Filter genuine negative grievances from suggestions and positive feedback using sentiment analysis
- Provide policymakers with data-backed root cause reports and visualisations
- Replace a slow, manual, analyst-heavy process with a scalable ML pipeline

---

## 🏗️ Pipeline Architecture

```
Grievance Data (Excel)
~30,000+ rows · 29 attributes
        │
        ▼
  Column Filtering
  [RegistrationNo, GrievanceDescription, State]
        │
        ▼
  Data Cleaning
  ├── Remove missing values
  ├── Remove duplicate grievances
  ├── Remove regional language rows (non-English)
  └── Remove emails, newlines, extra whitespace (regex)
        │
        ▼
  ┌─────────────────────────────────────────────────┐
  │        VADER Sentiment Analysis                 │
  │  → Classify: Positive / Negative / Neutral      │
  │  → Filter: retain Negative (true grievances)    │
  │  Result: 47% negative                           │
  └─────────────────────────────────────────────────┘
        │
        ▼ (negative grievances only)
  ┌─────────────────────────────────────────────┐
  │        NLP Pre-processing                   │
  │  ├── Tokenisation                           │
  │  ├── Stopword removal (NLTK)                │
  │  ├── Bigrams & Trigrams (Gensim Phrases)    │
  │  └── Lemmatisation (spaCy)                  │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  Optimal K Selection (Coherence Score)      │
  │  → Test k = 2 to 25                         │
  │  → Plot coherence score vs num topics       │
  │  → Select k at peak coherence               │
  │  Result: k=8 (coherence = 0.62, DFS case)  │
  └─────────────────────────────────────────────┘
        │
        ▼
  LDA Mallet Topic Model
  [Gibbs Sampling — more accurate than Gensim LDA]
        │
        ▼
  ┌──────────────────────┬──────────────────────┐
  │  pyLDAvis            │  Dominant Topic       │
  │  Visualisation       │  per Grievance        │
  │  (interactive HTML)  │  (Excel export)       │
  └──────────────────────┴──────────────────────┘
        │
        ▼
  Root Cause Interpretation
  [Analyst reads keywords per topic → assigns root cause label]
        │
        ▼
  Tableau Visualisation
  ├── Pareto chart (root cause frequency)
  ├── Bubble chart (root cause prominence)
  ├── India map (grievances by state)
  └── Bar chart (sentiment distribution)
```

---

## 📈 Key Results:

### Sentiment Analysis (VADER)

| Sentiment | Percentage | Interpretation |
|---|---|---|
| Positive | 53% | Suggestions, queries — not genuine grievances |
| **Negative** | **47%** | **True grievances — used for root cause analysis** |
| Neutral | 5% | Informational |

### Topic Modelling (LDA Mallet)

| Topic No. | Topic Name | % of Grievances |
|---|---|---|
| 1 | General Insurance | 31.08% |
| 2 | LIC | 26.16% |
| 3 | Health Insurance | 13.18% |
| 4 | Authority Issue | 12.72% |
| 5 | Court Complaints | 11.65% |

**Key finding:** The top 2 root causes account for **60.38%** of all grievances filed — meaning the department can resolve the majority of citizen issues by focusing on just two priority areas.

### Coherence Score

| Parameter | Value |
|---|---|
| Optimal number of topics (k) | 8 |
| Best coherence score | **0.62** |
| Method | LDA Mallet (Gibbs Sampling) |

### Geographical Distribution

Uttar Pradesh accounted for the highest grievance volume (for this case/department), while eastern states of India had the lowest filing rates — indicating a access gap.

---

## 🛠️ Tech Stack

`Python` `Gensim` `LDA Mallet` `NLTK` `spaCy` `pyLDAvis` `VADER (vaderSentiment)` `Pandas` `NumPy` `Matplotlib` `Tableau` `Regex`

---

## 📁 Repository Structure

```
├── notebook/
│   └── information_retrieval_nlp.ipynb   # Full pipeline — topic modelling + sentiment analysis
└── README.md
```

> **Note:** The `data/` folder is intentionally absent. The pipeline expects an Excel file with a `GrievanceDescription` column. See the Data section below for how to use your own data.

---

## 🚀 How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
gensim
nltk
spacy
pyldavis
vaderSentiment
matplotlib
openpyxl
```

Download required NLTK and spaCy resources:

```python
import nltk
nltk.download('stopwords')

import spacy
# python -m spacy download en_core_web_sm
```

### Using Your Own Data

The notebook expects an Excel file with **at minimum** these columns:

| Column | Description |
|---|---|
| `RegistrationNo` | Unique ID per grievance |
| `GrievanceDescription` | Free-text grievance filed by citizen |
| `State` | State where grievance was filed |

Load your file:
```python
data = pd.read_excel('your_grievance_data.xlsx')
```

**The notebook is fully reusable** for any text corpus where root cause extraction is needed — not limited to government grievances. It can be adapted for customer complaints, support tickets, or survey responses.

### LDA Mallet Setup

LDA Mallet requires a separate installation of the MALLET Java toolkit:

1. Download MALLET from [http://mallet.cs.umass.edu/](http://mallet.cs.umass.edu/)
2. Extract and note the path to `mallet` (or `mallet.bat` on Windows)
3. Update the path in the notebook:
```python
mallet_path = 'path/to/mallet-2.0.8/bin/mallet'
```

> **Tip:** If MALLET setup is complex, the notebook also includes a standard Gensim LDA fallback — though coherence scores will be slightly lower.

---

## 🔬 Why LDA Mallet over Standard LDA?

Standard Gensim LDA uses **Variational Bayes sampling** — fast but less accurate. LDA Mallet uses **Gibbs Sampling**, which is slower but consistently produces more coherent, interpretable topics. For government policy applications where accuracy of root causes matters, Mallet is the better choice.

---

## 💡 Real-World Impact

This pipeline was deployed across **21 Government Departments and Ministries**. The outputs were used to:

- Identify top root causes driving citizen grievances in each department
- Prioritise policy interventions based on frequency and sentiment
- Present statistically-backed insights to government leadership at DARPG
- Replace a fully manual, time-intensive process with an automated ML pipeline

---

## 📄 License

Code methodology is open for academic and research use. Data is not included and must not be recreated or inferred.
