# 🏦 End-to-End Credit Risk Scoring Engine

## 📌 Overview
A full-stack financial machine learning application designed to predict the **Probability of Default (PD)** for loan applicants. This project simulates a real-world banking environment by ingesting raw data into a SQL database, training a predictive model, and deploying a user-friendly dashboard for loan officers.

## 🛠️ Tech Stack
* **Data Engineering:** Python, SQL (SQLite), Pandas
* **Machine Learning:** Scikit-Learn (Logistic Regression), SMOTE (Imbalanced Learning)
* **Deployment:** Streamlit (Web App), Pickle (Serialization)
* **Visualization:** Plotly, Matplotlib

## 🚀 Key Features
1. **SQL Data Pipeline:** Raw CSV ingestion into a relational database.
2. **Risk Modeling:** Logistic Regression model optimized for recall to minimize financial risk.
3. **Live Dashboard:** Interactive UI for stakeholders to input applicant details and receive instant risk decisions.
4. **Portfolio Analytics:** Visualizations of loan distribution and risk factors.

## 💻 How to Run Locally
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the ETL pipeline: `python database_setup.py`
4. Train the model: Run the Jupyter Notebook in `/notebooks`
5. Launch the App: `streamlit run app.py`
