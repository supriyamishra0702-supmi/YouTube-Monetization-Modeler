# YouTube-Monetization-Modeler
YouTube Monetization Modeler is an end-to-end Machine Learning project designed to provide data-driven insights for digital content creators.

# 🎥 YouTube Monetization Modeler
### **Social Media Analytics: Predictive Revenue Dashboard**

A professional end-to-end Machine Learning solution built to estimate YouTube ad revenue. This project transforms raw performance metrics into a functional tool for business planning and content strategy, achieving a high-precision **94.97% Accuracy**.

---

## 🎯 Problem Statement
As digital content becomes a primary source of income, predicting ad revenue is essential for financial planning. This project builds a predictive model to estimate **ad_revenue_usd** based on performance and contextual features, implemented via an interactive **Streamlit** web application.

---

## 🚀 Key Results & Performance
*   **High-Precision Model**: Achieved an **R² Score of 94.97%** using a RandomForest Regressor.
*   **Robust Data Engineering**: Cleaned and processed **122,000+ rows**, handling 5% missing values and 2% duplicates.
*   **Interactive Dashboard**: Real-time revenue forecasting based on user-inputted metrics.
*   **Feature Insight**: Identified **Watch Time** and **Engagement Rate** as the most significant drivers of revenue.

---
## 🖥️ Application Interface


| High Performance Video | Strategic Insights & Metrics |
| :---: | :---: |
|<img width="926" height="481" alt="app_demo" src="https://github.com/user-attachments/assets/d2aa6497-891d-49f7-88ed-b96f48bccf91" />
 | <img width="923" height="440" alt="app_demo1" src="https://github.com/user-attachments/assets/6fba2b66-ca6c-432f-a221-1302ed0799f8" />
 |
| *Revenue Forecast for High Views* | *Engagement & CPM Analysis* |

---

## 📈 Data Visualizations (EDA)
<img width="646" height="284" alt="heat_map" src="https://github.com/user-attachments/assets/94fdc5d3-2b85-49d1-8c3c-9a32baa4fcd1" />

*Visualizing the strong correlation between Watch Time and Ad Revenue.*

## 📊 Business Use Cases
*   **Content Strategy Optimization**: Help creators identify which content genres yield highest returns.
*   **Revenue Forecasting**: Enable media companies to predict expected income from future uploads.
*   **Creator Support Tools**: Provide engagement benchmarks to optimize algorithm reach.
*   **Ad Campaign Planning**: Allow advertisers to forecast ROI based on performance metrics.

---

## 📈 Model Evaluation Metrics

| Metric | Value |
| :--- | :--- |
| **Model Type** | **Random Forest Regressor** |
| **R² Score** | **94.97%** |
| **Mean Absolute Error (MAE)** | **$3.60** |
| **Status** | **Production Ready** |

---

## 🛠️ Skills & Tech Stack
*   **Language**: Python (Pandas, NumPy)
*   **Machine Learning**: Scikit-learn (Regression, Feature Engineering, Categorical Encoding)
*   **Data Visualization**: Matplotlib, Seaborn
*   **Deployment**: Streamlit
*   **EDA**: Outlier Detection, Missing Value Handling, Correlation Analysis

---

## 📂 Project Structure
- `app.py`: The interactive Streamlit dashboard.
- `clean_data.py`: Script for data cleaning and preprocessing.
- `train_model.py`: Script for model shootout and training.
- `final_model.pkl`: The trained and exported model.
- `label_encoders.pkl`: Encoded categorical mappings.

---

## ⚙️ Installation & Usage
1. **Clone the Repo** and install requirements:
   ```bash
   pip install pandas scikit-learn streamlit joblib matplotlib seaborn
   ```
2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

---
**Developed by:** [Supriya Mishra]  
**Technical Tags:** `Linear Regression`, `Random Forest`, `EDA`, `Feature Engineering`, `Streamlit`, `Social Media Analytics`
