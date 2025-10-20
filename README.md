# 🧠 Fraud Detection & Transaction Behavior Mining Project

## 📘 Overview
This project applies **data mining and machine learning** techniques to detect fraudulent financial transactions and uncover behavioral patterns from a large, complex dataset.  
It was completed as part of a **data mining competition**, addressing the full pipeline from **data acquisition** to **insight generation** through **classification, clustering, and association rule mining (ARM)**.

---

## 🚀 Project Workflow

### 🧩 Step 1 – Dataset Acquisition & Exploration
- Two related datasets:
  - **Transaction data:** `(71,381 rows × 394 columns)`
  - **Identity data:** `(144,233 rows × 41 columns)`
- After merging → `(71,381 × 434)` records.
- Data types: 376 float64, 14 object, 4 int64.
- Key challenge: heavy missingness (>90%) in several ID columns.

**Goal:** Understand structure, missingness, and potential predictive attributes for fraud detection.

---

### 🎯 Step 2 – Business Scenarios & Problem Definition
Formulated core business questions:
1. What patterns distinguish **fraudulent** from **legitimate** transactions?  
2. Which **customer/device features** are highly correlated with fraud?  
3. Can we group transactions by behavioral similarity (risk segmentation)?  

---

### 🧹 Step 3 – Data Preprocessing
- Handled missing values and outliers.  
- Encoded categorical variables using **LabelEncoder / OneHotEncoder**.  
- Scaled numerical attributes with **StandardScaler**.  
- Selected relevant features based on correlation and domain logic.  

**Outcome:** Cleaned, ready-to-model dataset for classification, clustering, and ARM.

---

### ⚙️ Step 4 – Association Rule Mining (ARM)
- Used **FP-Growth** and **Apriori** algorithms from `mlxtend` to find frequent itemsets.  
- Mined rules with metrics: **support, confidence, and lift**.  
- Discovered meaningful co-occurrences such as:  
  > High transaction amount + certain device type → increased fraud probability.  

**Result:** Exported frequent itemsets and association rules for business insights.

---

### 🤖 Step 5 – Classification (Fraud Detection Model)
- Built predictive models using **Random Forest**, **Logistic Regression**, and **Gradient Boosting**.  
- Evaluated via **accuracy, precision, recall, F1-score, and ROC-AUC**.  
- Achieved strong performance with ROC-AUC ≈ **0.93**.  

**Result:** Reliable fraud prediction model identifying high-risk transactions.

---

### 🔍 Step 6 – Clustering (Behavioral Segmentation)
- Applied **K-Means** clustering with PCA visualization.  
- Identified 3 major behavioral clusters:
  1. Low-risk: small, consistent transactions.  
  2. Medium-risk: irregular, moderate transactions.  
  3. High-risk: large, erratic transactions often linked to fraud.  

**Result:** Customer segmentation for focused fraud monitoring.

---

### 🏁 Step 7 – Final Insights & Recommendations
- Fraud rate ≈ **2.7%**, confirming strong class imbalance.  
- Missing data in identity features remains a key modeling challenge.  
- Combined outputs (classification + ARM + clustering) yield actionable strategies for:  
  - Early fraud detection  
  - Targeted investigation  
  - Customer risk profiling  

---

## 🧰 Key Technologies
- **Programming Language:** Python  
- **Core Libraries:** pandas, numpy, scikit-learn, mlxtend, matplotlib, seaborn  
- **Algorithms Used:**  
  - Classification → RandomForest, Logistic Regression  
  - Clustering → K-Means, PCA  
  - ARM → FP-Growth, Apriori  

---

## 📊 Results Summary

| Task | Method | Outcome |
|------|---------|----------|
| Data Exploration | EDA | Identified structure & missingness |
| Preprocessing | Encoding + Scaling | Cleaned data |
| ARM | FP-Growth | Discovered high-lift itemsets |
| Classification | RandomForest | ROC-AUC ≈ 0.93 |
| Clustering | K-Means | 3 behavioral clusters |
| Insights | Combined | Actionable fraud risk intelligence |

---

## 🧾 Conclusion
This project demonstrates the end-to-end application of **data mining** techniques to a real-world **fraud detection** problem.  
It integrates **supervised and unsupervised learning**, **association analysis**, and **data preprocessing** to derive meaningful, actionable insights from a large, complex dataset.

---

## 📎 Author
**Precious Wafula**  
_Data Science & AI Enthusiast | Competition Project_

---

## 📂 Repository Structure
