# 📱 Mobile Price Classification & Comparison System

A end-to-end Machine Learning application built using Python, PyQt5, Scikit-Learn, and SVM/Random Forest. Predicts mobile phone price categories based on hardware specifications.

## 📌 Project Architecture & Workflow

- **Step 1: Raw Dataset Collection** (`step1_explore.py`)
- **Step 2: Data Cleaning & Outlier Winsorization** (`step2_cleaning.py`)
- **Step 3: Feature Engineering** (`step3_wrangling.py`)
- **Step 4: Data Augmentation via SMOTE** (`step4_augmentation.py`)
- **Step 5: Exploratory Data Visualization** (`step5_visualization.py`)
- **Step 6: Multi-method Feature Selection** (`step6_feature_selection.py`)
- **Step 7: Machine Learning Baseline Training** (`step7_model_training.py`)
- **Step 8: Hyperparameter Optimization** (`step8_tuning.py`)
- **Step 9: Model Evaluation & Metric Comparison** (`step9_evaluation.py`)
- **Step 10: PyQt5 Desktop GUI Application** (`mobile_price_gui.py`)

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/mobile-price-classification.git
   cd mobile-price-classification

2. **Create and Activate a Virtual Environment:**
   ```Bash:
   Windows:
   python -m venv venv
   venv\Scripts\activate

   macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

3. **Install Dependencies:**
   ```Bash:
   pip install -r requirements.txt

🚀 How to Run the Application
- **To launch the interactive PyQt5 GUI directly:**
- **python mobile_price_gui.py**

**Running Pipeline Steps Individually:**
```Bash:
python step1_explore.py
python step2_cleaning.py
python step3_wrangling.py
python step4_augmentation.py
python step5_visualization.py
python step6_feature_selection.py
python step7_model_training.py
python step8_tuning.py
python step9_evaluation.py

📊 Models & Performance
- **Algorithm 1: Random Forest Classifier**
- **Algorithm 2: Support Vector Machine (SVM)**

