# 📱 MindCheck — Social Media Addiction Risk Predictor

> **Capstone Project** | Machine Learning + Full-Stack Web Application  
> Predicts a student's social media addiction risk level (Low / Moderate / High) using behavioural and wellness data, served via a beautiful interactive web interface.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Novelty & Key Contributions](#novelty--key-contributions)
3. [Dataset](#dataset)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [Feature Engineering](#feature-engineering)
   - [Model Training & Comparison](#model-training--comparison)
   - [Model Performance & Accuracy](#model-performance--accuracy)
   - [Selecting the Best Model](#selecting-the-best-model)
5. [Backend — Flask API](#backend--flask-api)
6. [Frontend — Web Interface](#frontend--web-interface)
7. [Project Structure](#project-structure)
8. [Getting Started](#getting-started)
9. [API Reference](#api-reference)
10. [Screenshots](#screenshots)
11. [Results Summary](#results-summary)

---

## 🧠 Project Overview

**MindCheck** is an end-to-end machine learning project that:

1. **Trains a classifier** on real-world student survey data to predict the level of social media addiction (`Low`, `Moderate`, or `High`).
2. **Deploys the best model** as a lightweight REST API backed by Flask.
3. **Presents the predictions** through a polished, mobile-friendly single-page web application — no sign-up, no data stored.

The user fills in a short questionnaire (age, sleep hours, daily social media usage, mental health score, relationship status, etc.) and receives an instant risk assessment with a numeric risk score (0–100), a class probability breakdown, and personalised digital-wellness tips.

---

## 🌟 Novelty & Key Contributions

| Contribution | Description |
|---|---|
| **Domain-aware feature engineering** | Four new features (`Usage_Sleep_Ratio`, `Wellbeing_Score`, `Sleep_Deprived`, `Heavy_User`) are derived from raw inputs before model training to capture non-linear interactions. |
| **Multi-model comparison with CV** | Random Forest, XGBoost, and SVM (RBF) are benchmarked on Accuracy, ROC-AUC (macro OvR), and 5-fold Cross-Validation score so the choice of final model is data-driven rather than assumed. |
| **Continuous Risk Score (0–100)** | Rather than only outputting a class label, the model computes a continuous scalar risk score by computing a weighted dot product of class probabilities (`Low × 0`, `Moderate × 50`, `High × 100`). This allows fine-grained comparison within a class. |
| **Human-readable, actionable output** | Predictions are translated into personalised, tier-specific wellness tips surfaced directly in the UI. |
| **No-signup, privacy-first UX** | User responses are never persisted on the server; the prediction result is passed client-side via `sessionStorage` only. |
| **Animated, adaptive results page** | The results page dynamically changes its colour palette (green / amber / red) and icons based on the predicted risk tier. |

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Custom student survey — `Students Social Media Addiction.csv` |
| **Rows** | 705 students |
| **Columns** | 13 (including `Student_ID`, `Country`, and the target `Addicted_Score`) |
| **Missing Values** | **0** |
| **Duplicates** | **0** |
| **Target Column** | `Addicted_Score` — integer score ranging from **2 to 9** |

### Target Class Mapping

| Score Range | Risk Class | Label Used |
|---|---|---|
| 2 – 4 | Low addiction risk | `Low` |
| 5 – 6 | Moderate addiction risk | `Moderate` |
| 7 – 9 | High addiction risk | `High` |

### Input Features (11)

| Feature | Type | Description |
|---|---|---|
| `Age` | Integer | Student age (18–24) |
| `Gender` | Categorical | Male / Female |
| `Academic_Level` | Categorical | High School / Undergraduate / Graduate |
| `Avg_Daily_Usage_Hours` | Float | Average hours on social media per day |
| `Most_Used_Platform` | Categorical | e.g. Instagram, TikTok, YouTube, Facebook … |
| `Affects_Academic_Performance` | Categorical (Yes/No) | Self-reported academic impact |
| `Sleep_Hours_Per_Night` | Float | Hours of sleep per night |
| `Mental_Health_Score` | Integer (1–10) | Self-reported mental health (1 = worst, 10 = best) |
| `Relationship_Status` | Categorical | Single / In Relationship / Complicated |
| `Conflicts_Over_Social_Media` | Integer (0–5) | Number of conflicts attributed to social media |

---

## 🤖 Machine Learning Pipeline

### Feature Engineering

Before model training, **four domain-motivated features** are added to enrich the representation:

```python
df['Usage_Sleep_Ratio'] = df['Avg_Daily_Usage_Hours'] / (df['Sleep_Hours_Per_Night'] + 0.1)
df['Wellbeing_Score']   = df['Mental_Health_Score'] - df['Conflicts_Over_Social_Media']
df['Sleep_Deprived']    = (df['Sleep_Hours_Per_Night'] < 7).astype(int)
df['Heavy_User']        = (df['Avg_Daily_Usage_Hours'] > 5).astype(int)
```

| Feature | Rationale |
|---|---|
| `Usage_Sleep_Ratio` | Captures the trade-off between screen time and rest — a key addiction signal |
| `Wellbeing_Score` | Net mental wellness after subtracting social-media-driven conflict |
| `Sleep_Deprived` | Binary flag for clinically-recommended sleep threshold |
| `Heavy_User` | Binary flag for heavy daily usage (> 5 hours) |

### Preprocessing

A `sklearn.Pipeline` + `ColumnTransformer` handles all preprocessing automatically:

- **Numerical features** → `StandardScaler`
- **Categorical features** → `OneHotEncoder` (handle_unknown='ignore')

Preprocessing is **fitted only on the training set** and applied to the test set to prevent data leakage.

### Train / Test Split

| Set | Size |
|---|---|
| Training | 80% (stratified) |
| Test | 20% (stratified) |

Stratification ensures all three risk classes are represented proportionately in both sets.

---

### Model Training & Comparison

Three classifiers are trained and evaluated end-to-end inside sklearn Pipelines:

| Model | Notes |
|---|---|
| **Random Forest** | 200 estimators, `random_state=42` |
| **XGBoost** | `use_label_encoder=False`, `eval_metric='mlogloss'` |
| **SVM (RBF)** | `probability=True`, `C=10`, `gamma='scale'` |

All models are evaluated on:
- **Test Accuracy**
- **ROC-AUC (macro OvR)** — area under the multi-class one-vs-rest ROC curve
- **5-fold Cross-Validation Accuracy** (mean ± std)

---

### Model Performance & Accuracy

```
=================================================================
    SOCIAL MEDIA ADDICTION RISK MODEL  —  FINAL SUMMARY
=================================================================
  Dataset          : 705 students × 11 features (after drop)
  Task             : 3-class risk classification
  Classes          : Low (score 2-4) | Moderate (5-6) | High (7-9)
  Train / Test     : 80% / 20%  (stratified split)
  Engineered Feats : 4 new features added
-----------------------------------------------------------------
  Random Forest           Acc=0.965  AUC=0.996  CV=0.986±0.016
  XGBoost                 Acc=0.972  AUC=0.985  CV=0.987±0.017
  SVM (RBF)               Acc=0.993  AUC=0.998  CV=0.990±0.010  <-- BEST
=================================================================
  Best Model       : SVM (RBF)
  Models saved to  : ./models/
=================================================================
```

### 5-Fold Cross-Validation Scores

| Model | Fold Scores | Mean CV |
|---|---|---|
| Random Forest | [1.000, 0.9929, 0.9787, 0.9574, 1.000] | **0.9858** |
| XGBoost | [1.000, 1.000, 0.9787, 0.9574, 1.000] | **0.9872** |
| SVM (RBF) | [1.000, 0.9929, 0.9787, 0.9787, 1.000] | **0.9901** |

---

### Selecting the Best Model

The best model is chosen programmatically by **highest ROC-AUC** score on the held-out test set:

> 🏆 **SVM (RBF)** with:
> - **Test Accuracy: 99.29%**
> - **ROC-AUC: 0.9984**
> - **5-fold CV Mean: 99.01%**

The best model pipeline is serialised with `joblib` to `models/best_model_pipeline.pkl`, along with a human-readable `models/metadata.json` that stores class mappings, feature lists, and all benchmark results.

---

## 🖥️ Backend — Flask API

**File:** `files/app.py`

The Flask server exposes three endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the quiz / input form (index.html) |
| `/result` | GET | Serves the results page (result.html) |
| `/predict` | POST | Accepts JSON payload, returns prediction JSON |
| `/health` | GET | Simple health-check — returns `{"status": "ok"}` |

### Prediction Endpoint — `/predict`

**Request Body (JSON):**
```json
{
  "age": 20,
  "gender": "Female",
  "academic_level": "Undergraduate",
  "avg_daily_usage_hours": 5.0,
  "most_used_platform": "Instagram",
  "affects_academic_performance": "Yes",
  "sleep_hours_per_night": 6.0,
  "mental_health_score": 6,
  "relationship_status": "Single",
  "conflicts_over_social_media": 3
}
```

**Response (JSON):**
```json
{
  "label": "High",
  "risk_score": 78.5,
  "probabilities": {
    "Low": 2.1,
    "Moderate": 19.3,
    "High": 78.6
  },
  "tips": [
    "Your usage patterns suggest a strong dependency — take action now.",
    "Enable screen time limits on all your devices immediately.",
    "Talk to a counselor or trusted person about your digital habits.",
    "Replace at least 1 hour of social media with physical activity daily.",
    "Turn off all non-essential notifications to reduce compulsive checking."
  ]
}
```

### Risk Score Computation

```python
risk_score = round(float(np.dot(proba, [0, 50, 100])), 1)
```

The risk score is a continuous value between 0 and 100:
- **0–25**: Solidly Low risk
- **25–75**: Moderate range
- **75–100**: High risk zone

### Wellness Tips

The server provides curated, tier-specific tips for each risk class:

| Risk Level | Focus of Tips |
|---|---|
| **Low** | Positive reinforcement — keep healthy habits |
| **Moderate** | Specific behavioural suggestions (app timers, digital detox before bed, offline activities) |
| **High** | Urgent, actionable advice (screen time limits, counselling, notification management) |

---

## 🎨 Frontend — Web Interface

**Files:** `files/templates/index.html` and `files/templates/result.html`

### Design Language

- **Typography:** Google Fonts — *Playfair Display* (headings, heavy/decorative) + *Nunito* (body, clean/readable)
- **Colour Palette:** Coral (#ff6b6b) → Orange (#ff9f43) → Yellow (#ffd32a) gradient theme
- **Animations:** CSS keyframe animations (`fadeDown`, `fadeUp`, `popIn`, `bounceIn`, `float`)
- **Micro-interactions:** Hover scale effects on buttons, animated slider thumb, animated tip cards

### Index / Quiz Page (`index.html`)

The input form is divided into 4 semantic sections:

| Section | Fields |
|---|---|
| **About You** | Age (number input), Gender (select), Study Level (select) |
| **Your Phone Habits** | Daily usage hours (animated range slider), Platform (select) |
| **Sleep & Wellbeing** | Sleep hours (range slider), Mental health score (range slider) |
| **Social Life** | Relationship Status (select), Conflicts (toggle button), Academic impact (toggle button) |

**UX / interaction detail:**
- Range sliders are styled with a live fill gradient that tracks the thumb position via a CSS custom property `--pct`.
- Toggle buttons (Yes/No) replace `<select>` for binary questions for a cleaner tap target.
- The submit button enters a loading state (spinner + label change) during the async fetch call.
- The prediction result is passed to the results page via `sessionStorage` — no server round-trip needed.

### Results Page (`result.html`)

The results page is fully **dynamic** and **theme-aware**:

- The entire page background gradient, blob colours, and accent colour switch to one of three colour palettes:
  - 🟢 **Low**: Green (#11998e → #38ef7d)
  - 🟡 **Moderate**: Amber (#f7971e → #ffd200)
  - 🔴 **High**: Red (#ff416c → #ff4b2b)
- An SVG ring meter animates the fill to reflect the numeric risk score.
- Animated horizontal probability bars show the class breakdown.
- Tip cards slide in sequentially with staggered animation delays.

---

## 📁 Project Structure

```
CAPSTONE/
│
├── Capstone_Project_Final.ipynb    # Main ML notebook (EDA → training → export)
├── Students Social Media Addiction.csv  # Raw dataset (705 records, 13 columns)
├── Capstone_Project_Final.pdf      # Project report (PDF)
├── Capstone Project Presentation.pdf   # Presentation slides (PDF)
│
└── files/
    ├── app.py                      # Flask backend server
    │
    ├── models/                     # (Generated by notebook)
    │   ├── best_model_pipeline.pkl     # SVM (RBF) — serialised sklearn Pipeline
    │   ├── random_forest_pipeline.pkl
    │   ├── xgboost_pipeline.pkl
    │   ├── svm_pipeline.pkl
    │   └── metadata.json               # Benchmark results + feature/class mappings
    │
    └── templates/
        ├── index.html              # Quiz / input form
        └── result.html             # Prediction results page
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install flask scikit-learn xgboost joblib numpy pandas
```

> **Note:** The trained models live in `files/models/`. They are generated by running `Capstone_Project_Final.ipynb` end-to-end (requires a working Python environment with Jupyter and the above packages, plus `matplotlib`, `seaborn`).

### Run the Flask Server

```bash
cd files
python app.py
```

The server starts at **http://127.0.0.1:5000** in debug mode.

Navigate to `http://127.0.0.1:5000` in your browser to use the quiz.

### Re-train the Model

Open and run **`Capstone_Project_Final.ipynb`** cell by cell (or *Run All*) in Jupyter Notebook / JupyterLab. The notebook will:

1. Load and inspect the dataset
2. Perform exploratory data analysis (EDA) with visualisations
3. Engineer features
4. Preprocess and split data
5. Train Random Forest, XGBoost, and SVM
6. Evaluate and compare all models
7. Select the best model and save everything to `./models/`

---

## 🔌 API Reference

### `POST /predict`

#### Request Headers
```
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `age` | integer | ✅ | Student age |
| `gender` | string | ✅ | `"Male"` or `"Female"` |
| `academic_level` | string | ✅ | `"High School"`, `"Undergraduate"`, or `"Graduate"` |
| `avg_daily_usage_hours` | float | ✅ | 0.0 – 12.0 |
| `most_used_platform` | string | ✅ | `"Instagram"`, `"TikTok"`, `"YouTube"`, etc. |
| `affects_academic_performance` | string | ✅ | `"Yes"` or `"No"` |
| `sleep_hours_per_night` | float | ✅ | 3.0 – 10.0 |
| `mental_health_score` | integer | ✅ | 1 – 10 |
| `relationship_status` | string | ✅ | `"Single"`, `"In Relationship"`, or `"Complicated"` |
| `conflicts_over_social_media` | integer | ✅ | 0 – 5 |

#### Response

| Field | Type | Description |
|---|---|---|
| `label` | string | `"Low"`, `"Moderate"`, or `"High"` |
| `risk_score` | float | Continuous score from 0 to 100 |
| `probabilities.Low` | float | Probability (%) for Low class |
| `probabilities.Moderate` | float | Probability (%) for Moderate class |
| `probabilities.High` | float | Probability (%) for High class |
| `tips` | string[] | 3–5 personalised wellness recommendations |

---

## 📸 Screenshots

### Quiz Page

<img src="assets/quiz-page.png" alt="Quiz Page" width="600" />

### Results — Low Risk
<img src="assets/result-low.png" alt="Low Risk Result" width="600" />

### Results — High Risk
<img src="assets/result-high.png" alt="High Risk Result" width="600" />

---

## 📈 Results Summary

| Model | Test Accuracy | ROC-AUC (Macro OvR) | 5-Fold CV Mean |
|---|---|---|---|
| Random Forest | 96.45% | 0.9963 | 98.58% ± 1.62% |
| XGBoost | 97.16% | 0.9851 | 98.72% ± 1.70% |
| **SVM (RBF) ← Best** | **99.29%** | **0.9984** | **99.01% ± 0.96%** |

The **Support Vector Machine with RBF kernel (C=10, gamma='scale')** outperforms the other two models on both test accuracy and ROC-AUC, and achieves the tightest cross-validation variance (±0.96%), indicating strong generalisation across all data folds.

---

## 📝 License

This project was developed as a university capstone and is intended for academic and educational purposes.

---

## 🙌 Acknowledgements

- Dataset collected via student surveys
- UI design inspired by modern health & wellness applications
- ML pipeline built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [Flask](https://flask.palletsprojects.com/)
