# 🧠 Behavioral AI: Intent-to-Action Predictive Engine

An interactive machine learning pipeline that analyzes daily micro-habits to predict the statistical probability of achieving long-term macro-goals. Built with **Python, XGBoost, and Streamlit**.

This project bridges the gap between behavioral psychology and data science, utilizing an advanced gradient boosting algorithm to quantify how friction (e.g., excessive screen time) and constructive habits (e.g., deep work, sleep) influence goal outcomes.

## 🚀 Live Demo (Local)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172434?style=for-the-badge&logo=XGBoost&logoColor=white)

---

## 🛠️ Architecture & Workflow

This engine operates on a two-part decoupled architecture:

### 1. The Data Engine (`build_ai.py`)
* **Synthetic Data Generation:** Generates a 5,000-row behavioral dataset simulating human habits (Sleep, Screen Time, Focus Hours). 
* **Statistical Noise Injection:** Applies Gaussian noise to the mathematical success formula to simulate real-world unpredictability and prevent algorithmic overfitting.
* **XGBoost Classification:** Trains an `XGBClassifier` to map the non-linear relationships between variables, outputting a serialized `.pkl` model.

### 2. The Interactive UI (`app.py`)
* **Asynchronous Inference:** A Streamlit dashboard that loads the serialized XGBoost model into memory.
* **Dynamic Probability Mapping:** Users adjust slider metrics to pass realtime payload arrays into the model's `predict_proba()` function, rendering smooth success-probability curves.

---

## 💻 Local Installation & Setup

Want to run this predictive model on your own machine? Follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/behavioral-ai-predictive-engine.git](https://github.com/YourUsername/behavioral-ai-predictive-engine.git)
cd behavioral-ai-predictive-engine