# Employee Attrition Risk Predictor

An interactive HR analytics application that uses logistic regression to predict employee attrition risk. HR teams can adjust an employee profile in real time and see attrition probability scores across business travel frequency levels, with uncertainty bounds derived from cross-validation.

# Visualization Link
https://econ3916finalproject-rycrzy7dj7w743pculvcrw.streamlit.app/
---

## Stakeholders

**Primary Users — HR Managers and People Operations Teams**

This tool is designed for HR practitioners who need to identify employees at elevated attrition risk before they resign. The model surfaces a ranked risk score across three business travel scenarios (No Travel, Travel Rarely, Travel Frequently) so that managers can make informed, targeted retention decisions.

**Important**: The model is a *screening signal*, not a definitive prediction. Scores should be interpreted comparatively and used alongside direct managerial judgment. The system intentionally prioritizes recall (catching more true attrition cases) over precision, which means false positives are expected and acceptable in this use case.

---

## What Was Done

### 1. Exploratory Data Analysis

The IBM HR Analytics dataset (1,470 employees, 35 features) was analyzed for data quality, distributions, and potential predictors.

- Confirmed zero missing values across all columns
- Identified and retained Tukey fence outliers (~7–8%) as valid edge cases rather than errors
- Assessed multicollinearity among tenure-related variables (YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, TotalWorkingYears)
- Dropped three constant-value columns: `EmployeeCount`, `StandardHours`, `Over18`
- Found a moderately imbalanced target: ~16% attrition rate

### 2. Modeling

Two candidate models were evaluated: Logistic Regression (baseline) and Random Forest (alternative).

**Logistic Regression was selected** for the following reasons:
- Coefficient-level interpretability — HR teams can understand *why* an employee is flagged
- Lower risk of overfitting on a dataset of this size
- Comparable cross-validated F1 performance to Random Forest

**Class imbalance** was addressed via `class_weight='balanced'`, which upweights the minority (attrition) class during training.

| Metric | Value |
|---|---|
| Accuracy | 77% |
| Precision (attrition class) | 37% |
| Recall (attrition class) | 64% |
| F1 Score (attrition class) | 0.47 |
| Cross-Validated F1 (5-fold) | 0.514 ± 0.0547 |

The cross-validation error margin (±5.47%) is surfaced in the app as error bars on each risk score.

### 3. Feature Importance

Logistic regression coefficients were visualized to rank feature influence. Key predictors include overtime status, job involvement, environment satisfaction, job satisfaction, and years since last promotion. A disclaimer is included throughout: predictive importance does not imply causal effect.

### 4. Application

A Streamlit web app (`app.py`) exposes the model via an interactive dashboard:

- **Sidebar controls**: Sliders and dropdowns for 15+ employee profile attributes (age, job role, marital status, overtime, satisfaction scores, tenure, etc.)
- **Prediction output**: Bar chart with attrition risk scores (0–100%) for three business travel categories, with ±5.47% error bars
- **Comparative framing**: Scores are best interpreted relative to each other, not as absolute probabilities (due to class reweighting)

---

## Setup

### Prerequisites

- Python 3.9 or higher
- `pip`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd econ3916_final_project

# (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Reproducibility

### Dataset

The model was trained on the **IBM HR Analytics Employee Attrition & Performance** dataset, available publicly on Kaggle. The dataset file `ibm_hr_dataset.csv` is included in this repository.

- 1,470 employee records, 35 columns
- No preprocessing outside of dropping constant columns and one-hot encoding categoricals

### Trained Artifacts

Pre-trained artifacts are included so the app runs without retraining:

| File | Description |
|---|---|
| `model.pkl` | Trained `LogisticRegression` object (`class_weight='balanced'`) |
| `scaler.pkl` | Fitted `StandardScaler` for feature normalization |
| `feature_columns.pkl` | Ordered list of 31 feature column names expected by the model |

### Retraining from Scratch

To retrain the model, run all cells in `3916-final-project-starter.ipynb` in order. The notebook covers:

1. Data loading and cleaning
2. EDA and outlier analysis
3. Train/test split (80/20, stratified by attrition)
4. Model training and evaluation
5. Cross-validation
6. Artifact serialization (`joblib.dump`)

Running the notebook end-to-end will overwrite `model.pkl`, `scaler.pkl`, and `feature_columns.pkl` with freshly trained versions.

### Environment

Exact dependency versions are pinned in `requirements.txt`. The key versions used during development:

```
streamlit==1.56.0
pandas==3.0.2
numpy==2.4.4
scikit-learn==1.8.0
matplotlib==3.10.8
seaborn==0.13.2
joblib==1.5.3
```

---

## Project Structure

```
econ3916_final_project/
├── app.py                            # Streamlit dashboard
├── 3916-final-project-starter.ipynb  # Full analysis notebook (EDA → modeling)
├── ibm_hr_dataset.csv                # IBM HR Analytics dataset
├── model.pkl                         # Trained logistic regression model
├── scaler.pkl                        # Feature scaler
├── feature_columns.pkl               # Model feature schema
└── requirements.txt                  # Pinned dependencies
```
