<h1 align="center">🚀 Indian Startup Funding Dashboard</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Streamlit%20%26%20Plotly-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ML-Powered%20by%20XGBoost-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Category-Startup%20Analytics-success?style=for-the-badge"/>
</p>

<p align="center"><strong>Visual Analytics and Funding Prediction Tool for India’s Dynamic Startup Ecosystem</strong></p>

---

## 🧩 Introduction: The Case of Nilla

Meet **Nilla**, a passionate startup founder with the drive and the dream. But soon after launching, she's overwhelmed by one question:

> 💸 “How do I get funding?”
>  
> 🔍 “Which investors are looking for startups like mine?”

Diving into spreadsheets and data reports offers no real guidance. That’s where our solution steps in…

### 🎯 We built an **interactive, data-powered dashboard** to answer questions like:
- What sectors are investors most interested in?
- Which cities are attracting the most funding?
- What’s the predicted funding potential of your startup profile?

---

## 🧠 Problem Statement

Despite an abundance of data, **actionable insights on Indian startup funding are rare**. Founders and investors lack:
- 🔍 Clarity on funding trends across industries and locations
- 🔗 Tools to understand investor behavior patterns
- 📊 Predictive tools to estimate funding potential

> **This dashboard bridges that gap** using intuitive visualizations and machine learning.

---

## 🌟 Why This Project Matters

India’s startup ecosystem is booming — but traditional databases fail to provide **instant, interactive, and predictive answers**.

With this dashboard:
- 🧠 Founders get clear insights into their funding landscape
- 💼 Investors identify growth-ready sectors and hotspots
- 📈 Users predict realistic funding expectations based on real startup traits

---

## 🎯 Project Objectives

✅ **Visualize Funding Trends** across years and industries  
✅ **Highlight Top Startups, Cities, and Investment Types**  
✅ **Enable Real-Time Filtering** by Year, Industry, City  
✅ **Predict Funding Amount** using ML (XGBoost Regressor)  
✅ **Support Informed Decisions** for startups, investors, analysts

---

## 🗺️ Project Roadmap & Deliverables

### 🔹 Interactive Dashboard (Streamlit + Plotly)
- Year/City/Industry/Investment Type filters
- Multi-tab interface: **Trends**, **Top Startups**, **Funding Prediction**

### 🔹 Funding Trends Visualization
- Line charts, bar graphs, and scatter plots
- Top-funded startups, cities, and sectors

### 🔹 Machine Learning Module
- **XGBoost Regressor** model
- Predicts funding amount in INR (Crores) from inputs like:
  - Year
  - Industry Vertical
  - City
  - Investment Type

### 🔹 Feature Importance Insights
- Ranks top drivers of funding outcomes

---

## 🔍 Dataset Overview

📂 **Source**: Kaggle + curated sources  
🧹 **Cleaned & Preprocessed** for modeling and analysis  

### 🗝️ Key Fields:
- `Startup Name`, `City`, `Industry Vertical`, `SubVertical`  
- `Investment Type`, `Investors Name`, `Amount in INR (Cr)`, `Year`

### 🧪 Engineered Fields:
- `Investor Count` – total investors per round  
- `Funding Amount (Numeric)` – standardized for ML  
- Encoded categorical fields for modeling

---

## 🛠️ Tech Stack

| Category | Tools/Libraries |
|----------|------------------|
| Dashboard | Streamlit |
| Visuals | Plotly |
| ML Modeling | Scikit-learn, XGBoost |
| Data Handling | Pandas, NumPy |
| Language | Python |

---

## 🔮 Funding Prediction Model

- **Model**: XGBoost Regressor
- **Inputs**:
  - Year
  - Industry
  - City
  - Investment Type
- **Output**:
  - Predicted Funding Amount (in Crores)
- **Extras**: Feature importance insights


---

## ✅ How to Run


# 1. Clone the repository
git clone https://github.com/keerthana777z/Spreadsheet.git
cd Spreadsheet  # update this path if needed

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit
streamlit run app.py

---
📈 Conclusion
This project delivers a complete data-driven platform for navigating the complex Indian startup funding space. From deep trend analysis to funding prediction, it enables entrepreneurs, VCs, and analysts to make smarter, faster decisions — visually and interactively.

👩‍💻 Author
AR Keerthana
https://github.com/keerthana777z

🙌 Thank You!
“Backed by data. Driven by insight. Built for India’s startup future.”
