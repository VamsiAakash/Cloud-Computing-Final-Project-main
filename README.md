# 🚦 Real-Time Traffic Accident Severity Prediction

> A complete end-to-end Big Data pipeline for real-time traffic accident severity prediction using Apache Kafka, Apache Spark, Machine Learning, and a live Streamlit dashboard.

**ITCS 6190 — Cloud Computing for Data Analysis | Spring 2026 | UNC Charlotte**

---

## 👥 Team Members

| # | Name | Student ID | University |
|---|------|-----------|------------|
| 1 | **Pranathi Thotli** | 801425061 | UNC Charlotte |
| 2 | **Vamsi Aakash Samudrala** | 801425922 | UNC Charlotte |
| 3 | **Vaishnav Reddy Dandu** | 801411660 | UNC Charlotte |

---

## 📌 Project Overview

Current traffic management systems are entirely **REACTIVE** — by the time an accident is detected and communicated, drivers are already stuck in congestion. This project builds a **proactive real-time system** that:

- Streams 7.73 million US accident records through **Apache Kafka**
- Processes and analyzes data using **Apache Spark Structured Streaming**
- Runs **5 complex Spark SQL queries** to identify dangerous hotspots
- Predicts accident severity (1–4) using **Random Forest (Spark MLlib)** and **XGBoost**
- Displays everything on a **live Streamlit dashboard** with real-time alerts

---

## 🎯 Problem Statement

| Problem | Description |
|---------|-------------|
| **Late Detection** | Accidents reported minutes after they occur — following vehicles approach unaware |
| **No Severity Assessment** | No system instantly classifies whether an accident needs a tow truck or full emergency closure |
| **No Hotspot Intelligence** | Traffic centers cannot instantly query which roads are most dangerous under current conditions |
| **Data Bottleneck** | Millions of records accumulating continuously — traditional batch systems too slow |

---

## 🚦 Severity Classes — Target Variable

> **Severity = How badly the accident disrupted traffic flow (NOT injury level)**

| Level | Label | Duration | Response | Frequency |
|-------|-------|----------|----------|-----------|
| **1** | Very Minor | 2–5 min | None needed | Rarest |
| **2** | Light/Moderate | 10–30 min | Police + Tow Truck | Most Common (55%) |
| **3** | Serious | 45min–2hr | Police + Ambulance + Tow | Common (44%) |
| **4** | Critical/Severe | 3–6+ hrs | Full Emergency Response | Only 2.65% |

---

## 📁 Project Structure

```
Checkin_MLComponent_Pipiline/
│
├── 📂 data/
│   ├── sample.csv                    # 100K sample dataset
│   └── US_Accidents_March23.csv      # Full 7.73M dataset (3.2 GB)
│
├── 📂 models/
│   ├── rf_model.pkl                  # XGBoost trained model (81.87%)
│   └── rf_severity_model/            # Spark MLlib RF model (60.52%)
│
├── 📂 plots/
│   ├── complex_heatmap.png           # Q1 — Severity heatmap by hour/day
│   ├── complex_danger_score.png      # Q2 — Danger score per state
│   ├── complex_weather_risk.png      # Q3 — Weather risk index
│   ├── complex_compound_risk.png     # Q4 — Junction compound risk
│   └── complex_road_segments.png     # Q5 — Top 10 dangerous roads
│
├── dashboard_final.py                # 🖥️ Main Streamlit dashboard
├── ML_Model.py                       # 🌲 Spark MLlib Random Forest
├── ComplexQueries.py                 # 🔍 5 Complex Spark SQL queries
├── ComplexPlots.py                   # 📊 Complex query visualizations
├── retrain_xgb.py                    # ⚡ XGBoost model (full dataset)
├── retrain.py                        # 🌲 sklearn RF (backup)
├── fix_accuracy.py                   # 🔧 Utility script
├── Makefile                          # 🛠️ Project automation
├── requirements.txt                  # 📦 Python dependencies
└── README.md                         # 📖 This file
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | US Accidents Dataset |
| **File** | US_Accidents_March23.csv |
| **Source** | Kaggle — Sobhan Moosavi |
| **Records** | 7,728,394 |
| **Columns** | 46 |
| **Coverage** | 49 US States |
| **Period** | February 2016 – March 2023 |
| **File Size** | ~3.2 GB |

### 46 Columns — 6 Attribute Groups

| Group | Count | Examples |
|-------|-------|---------|
| Identity & Core | 3 | ID, Source, Severity |
| Temporal | 2 | Start_Time, End_Time |
| Location | 9 | Lat, Lng, Street, City, State |
| Weather | 10 | Temperature, Visibility, Wind Speed, Humidity |
| Road Features | 13 | Junction, Traffic_Signal, Crossing |
| Day / Night | 4 | Sunrise_Sunset, Civil_Twilight |

---

## 🏗️ System Architecture — Full Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│   DATA SOURCE   │────▶│  KAFKA PRODUCER  │────▶│  SPARK STREAMING    │────▶│ OUTPUT SINK  │
│                 │     │                  │     │                     │     │              │
│ US_Accidents    │     │ Row by row CSV   │     │ 5/15/60 min windows │     │ Parquet files│
│ 7.73M records   │     │ JSON to Kafka    │     │ 10-min watermark    │     │ CSV by State │
│ 46 columns      │     │ topic:           │     │ Structured Stream   │     │ Alert files  │
└─────────────────┘     │ traffic-accidents│     └─────────────────────┘     └──────────────┘
                        └──────────────────┘              │
                                                          │
                              ┌───────────────────────────┼──────────────────────┐
                              ▼                           ▼                      ▼
                    ┌──────────────────┐      ┌──────────────────┐    ┌──────────────────┐
                    │   SPARK SQL      │      │   MLlib MODEL    │    │   EVALUATION     │
                    │                  │      │                  │    │                  │
                    │ 5 Complex Queries│      │ Random Forest    │    │ Accuracy + F1    │
                    │ Hotspot Analysis │      │ XGBoost          │    │ Confusion Matrix │
                    │ Weather Risk     │      │ Severity 1–4     │    │ Feature Importance│
                    └──────────────────┘      └──────────────────┘    └──────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────┐
                                              │  STREAMLIT       │
                                              │  DASHBOARD       │
                                              │  localhost:8501  │
                                              │  Real-time alerts│
                                              └──────────────────┘
```

---

## ⚙️ Prerequisites

- Python 3.8+
- Java 8 or 11 (required for Apache Spark)
- Apache Spark 3.x
- macOS / Linux / Windows (WSL recommended)

### Check Java version
```bash
java -version
```

### Check Python version
```bash
python3 --version
```

---

## 🚀 Installation & Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Cloud-Computing-Final-Project.git
cd Cloud-Computing-Final-Project
```

### Step 2 — Install Dependencies
```bash
make install
```
Or manually:
```bash
pip3 install -r requirements.txt --break-system-packages
```

### Step 3 — Verify Setup
```bash
make check
```
Expected output:
```
✅ sample.csv found
✅ US_Accidents_March23.csv found
✅ rf_model.pkl found
✅ dashboard_final.py found
✅ plots/ folder found
```

---

## 🔄 Execution — Step by Step

### Step 1 — Train Spark MLlib Random Forest Model

```bash
make train-spark
```
Or directly:
```bash
python3 ML_Model.py
```

**What it does:**
- Loads 100K sample from `data/sample.csv`
- Engineers 11 features from raw columns
- Runs Spark MLlib Pipeline: Imputer → VectorAssembler → RandomForestClassifier
- 80/20 train/test split (80,201 training / 19,799 test)
- Evaluates accuracy, F1, precision, recall
- Saves model to `models/rf_severity_model` (Spark format)

**Expected output:**
```
✅ Accuracy    : 0.6052 (60.52%)
✅ Weighted F1 : 0.6048 (60.48%)
✅ Precision   : 0.6043 (60.43%)
✅ Recall      : 0.6052 (60.52%)
✅ Model saved → models/rf_severity_model
```

---

### Step 2 — Train XGBoost Model on Full 7.7M Dataset

```bash
make train
```
Or directly:
```bash
python3 retrain_xgb.py
```

**What it does:**
- Loads full 7.73M dataset from `data/US_Accidents_March23.csv`
- Extracts Hour, Month, Day_of_Week from Start_Time
- Label encodes Weather_Condition, State, Sunrise_Sunset
- Uses 15 engineered features
- Trains XGBoost with 300 trees, max_depth=8, learning_rate=0.1
- 80/20 stratified split (6,381,047 training / 1,276,210 test)
- Saves model to `models/rf_model.pkl` (joblib format)

**Expected output:**
```
Shape: (7728394, 46)
Using 15 features
Training on 6381047 samples
Training XGBoost (2-3 minutes)...
========================================
ACCURACY: 81.87%
========================================
Model saved to models/rf_model.pkl
```

---

### Step 3 — Run 5 Complex Spark SQL Queries

```bash
make sql
```
Or directly:
```bash
python3 ComplexQueries.py
```

**What it does:**
- Loads dataset into Spark SQL temporary view
- Runs 5 analytical queries on the full dataset

**Queries and results:**

| Query | Description | Key Finding |
|-------|-------------|-------------|
| **Q1** | Severity Heatmap — Hour vs Day of Week | Sunday 6AM = highest avg severity (2.598) |
| **Q2** | Danger Score per State (custom formula) | CA = 1.2494, OH = 1.1231 |
| **Q3** | Weather Risk Index per Condition | Partly Cloudy = 50.17% risk (highest) |
| **Q4** | Junction + Time of Day Compound Risk | Junction+No Signal = avg severity 2.606 |
| **Q5** | Top 10 Most Dangerous Road Segments | I-405 N Los Angeles = 517 serious accidents |

---

### Step 4 — Generate Complex Query Plots

```bash
make plots
```
Or directly:
```bash
python3 ComplexPlots.py
```

**What it does:**
- Runs the same 5 queries using Spark
- Generates 5 matplotlib visualizations
- Saves all plots to `plots/` folder

**Generated files:**
```
plots/complex_heatmap.png         ← Q1 severity heatmap
plots/complex_danger_score.png    ← Q2 state danger scores
plots/complex_weather_risk.png    ← Q3 weather risk index
plots/complex_compound_risk.png   ← Q4 junction compound risk
plots/complex_road_segments.png   ← Q5 dangerous road segments
```

---

### Step 5 — Launch Streamlit Dashboard

```bash
make dashboard
```
Or directly:
```bash
streamlit run dashboard_final.py
```

Open browser at: **http://localhost:8501**

---

## 🖥️ Dashboard Pages

### 🏠 Overview
- 5 KPI cards: Total records, Features, Accuracy, Severity classes, States
- Complete pipeline architecture diagram
- Severity scale with color coding
- Real dataset preview

### 📊 EDA & Visualizations (5 tabs)
| Tab | Visualization |
|-----|--------------|
| 1 — US Map | Choropleth map of all 49 states by accident count |
| 2 — Hourly | Accident frequency by hour of day (rush hour peaks) |
| 3 — Weather | Severity distribution by weather condition |
| 4 — States | Top states by accident count and average severity |
| 5 — Correlation | Pearson correlation of features with severity |

### 🔍 SQL Analytics
- **5 Simple Queries** — basic aggregations and counts
- **5 Complex Queries** — exact queries from ComplexQueries.py with live results and Spark plots

### 🤖 ML Model & Metrics
- Model architecture and pipeline stages
- Feature importance chart (Traffic_Signal_int = 38.25%)
- Classification report for all 4 severity classes
- Confusion matrix heatmap

### 🎯 Live Prediction
- Input 15 accident features using sliders
- XGBoost model predicts severity in milliseconds
- Confidence probability bars for all 4 classes

### ⚡ Kafka Stream Simulator
- Simulates real-time Kafka consumer
- Live event feed table with predictions
- Severity scatter chart over time
- Alert counter for S3/S4 events

---

## 🤖 Machine Learning Models

### Model 1 — Random Forest (Spark MLlib)

```python
# Pipeline stages:
# 1. Imputer      → fill missing numeric values with mean
# 2. VectorAssembler → combine 11 features into vector
# 3. RandomForestClassifier → 50 trees, maxDepth=10

Features: Temperature(F), Visibility(mi), Wind_Speed(mph), Humidity(%),
          Precipitation(in), Distance(mi), Hour_of_Day, Day_of_Week,
          Is_Night, Junction_int, Traffic_Signal_int

Training: 80,201 samples | Test: 19,799 samples
Accuracy: 60.52% | F1: 60.48%
Saved: models/rf_severity_model (Spark folder format)
```

### Model 2 — XGBoost (Full Dataset)

```python
# Configuration:
model = XGBClassifier(
    n_estimators=300,    # 300 sequential trees
    max_depth=8,         # max questions per tree
    learning_rate=0.1,   # correction rate per tree
    subsample=0.8,       # 80% rows per tree
    colsample_bytree=0.8 # 80% features per tree
)

Features: 15 engineered features including Weather_Condition,
          State, Sunrise_Sunset (label encoded)

Training: 6,381,047 samples | Test: 1,276,210 samples
Accuracy: 81.87% | Weighted F1: 79%
Saved: models/rf_model.pkl (joblib format)
```

### Classification Report — XGBoost

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| S1 Minor | 0.70 | 0.08 | 0.15 | 13,090 |
| S2 Moderate | **0.84** | **0.96** | **0.89** | 1,002,605 |
| S3 Serious | 0.64 | 0.37 | 0.47 | 227,930 |
| S4 Critical | 0.58 | 0.07 | 0.13 | 32,585 |
| **Weighted Avg** | **0.80** | **0.82** | **0.79** | 1,276,210 |

---

## 🔍 5 Complex Spark SQL Queries — Key Findings

### Q1 — Severity Heatmap (Hour × Day of Week)
```sql
SELECT HOUR(Start_Time) as hour_of_day,
       CASE DAYOFWEEK(Start_Time) WHEN 1 THEN 'Sunday' ... END as day_of_week,
       COUNT(*) as total_accidents,
       ROUND(AVG(Severity), 3) as avg_severity
FROM accidents
GROUP BY hour_of_day, day_of_week
HAVING COUNT(*) > 100
ORDER BY avg_severity DESC LIMIT 10
```
**Finding:** Sunday at 6 AM has the highest average severity of **2.598**

---

### Q2 — Custom Danger Score per State
```sql
SELECT State,
       ROUND((AVG(Severity) * 0.4) +
             (SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) * 0.6), 4)
       as danger_score
FROM accidents GROUP BY State
HAVING COUNT(*) > 500
ORDER BY danger_score DESC LIMIT 10
```
**Finding:** California = **1.2494** (highest danger score)

---

### Q3 — Weather Risk Index
```sql
SELECT Weather_Condition,
       ROUND(SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
       as risk_index_pct
FROM accidents WHERE Weather_Condition IS NOT NULL
GROUP BY Weather_Condition HAVING COUNT(*) > 1000
ORDER BY risk_index_pct DESC LIMIT 10
```
**Finding:** Partly Cloudy = **50.17%** risk index (highest)

---

### Q4 — Junction + Time of Day Compound Risk
```sql
SELECT Junction, Traffic_Signal,
       CASE WHEN HOUR(Start_Time) BETWEEN 6 AND 9 THEN 'Morning Rush' ... END as time_of_day,
       ROUND(AVG(Severity), 3) as avg_severity,
       ROUND(SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_severity_pct
FROM accidents GROUP BY Junction, Traffic_Signal, time_of_day
HAVING COUNT(*) > 200
ORDER BY avg_severity DESC LIMIT 12
```
**Finding:** Junction + No Signal = avg severity **2.606**, high severity rate **60.63%**

---

### Q5 — Top 10 Most Dangerous Road Segments
```sql
SELECT Street, City, State,
       SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as serious_accidents
FROM accidents WHERE Street IS NOT NULL
GROUP BY Street, City, State HAVING COUNT(*) > 5
ORDER BY serious_accidents DESC, avg_severity DESC LIMIT 10
```
**Finding:** I-405 N, Los Angeles = **517 serious accidents** (most dangerous road)

---

## 🛠️ Makefile Commands

```bash
make help          # Show all available commands
make install       # Install all Python dependencies
make train         # Train XGBoost on full 7.7M dataset
make train-spark   # Train Random Forest via Spark MLlib
make sql           # Run 5 complex Spark SQL queries
make plots         # Generate complex query visualizations
make dashboard     # Launch Streamlit dashboard
make all           # Full pipeline: install → train → dashboard
make check         # Verify all files are present
make clean         # Remove cache and temporary files
```

---

## 📦 Requirements

```
streamlit==1.43.2
pandas
numpy
scikit-learn
xgboost
plotly
joblib
pyspark
matplotlib
```

Install all:
```bash
pip3 install -r requirements.txt --break-system-packages
```

---

## 📈 Results Summary

| Metric | Random Forest (Spark) | XGBoost (Full Dataset) |
|--------|----------------------|----------------------|
| Dataset | 100K sample | 7.73M full |
| Algorithm | RandomForestClassifier | XGBClassifier |
| Trees | 50 | 300 |
| Features | 11 | 15 |
| Train samples | 80,201 | 6,381,047 |
| Test samples | 19,799 | 1,276,210 |
| **Accuracy** | **60.52%** | **81.87% ✅** |
| Weighted F1 | 60.48% | 79% |
| Precision | 60.43% | 80% |

---

## 🔮 Future Work

| Enhancement | Description |
|-------------|-------------|
| **Live Weather API** | Connect real-time weather to predict conditions before accidents |
| **SMS/Email Alerts** | Auto-notify authorities when S3/S4 predicted |
| **State-Level Partitioning** | Separate Kafka topic per state for parallel processing |
| **Auto Model Retraining** | Monthly scheduled retraining on new data |
| **SMOTE Oversampling** | Address class imbalance for S1 and S4 |
| **AWS Deployment** | Deploy on AWS EMR with auto-scaling Spark clusters |

---

## 🎓 Course Information

- **Course:** ITCS 6190 — Cloud Computing for Data Analysis
- **University:** UNC Charlotte
- **Semester:** Spring 2026
- **Professor:** Marco Vieira

---

## 📚 References

- Moosavi, S., et al. "A Countrywide Traffic Accident Dataset." arXiv, 2019.
- Apache Kafka Documentation: https://kafka.apache.org/documentation/
- Apache Spark MLlib Guide: https://spark.apache.org/docs/latest/ml-guide.html
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Streamlit Documentation: https://docs.streamlit.io/


##  Demo Video & Dataset

| Resource | Link |
|----------|------|
| **12-Min Demo Video** | [Watch on Google Drive](https://drive.google.com/drive/u/1/folders/1o536vuibHBMx2vbTQYZj8lOQnjL-fPq7) |
|  **Full Dataset (7.73M records)** | [Download from Google Drive](https://drive.google.com/drive/u/1/folders/1o536vuibHBMx2vbTQYZj8lOQnjL-fPq7) |

> ⚠️ Dataset (3.2 GB) and Demo Video (12 min) are hosted on Google Drive due to GitHub file size limits.
> Download `US_Accidents_March23.csv` and place it in the `data/` folder before running.
```


*This project shifts traffic accident management from reactive response to proactive intelligence.* 🚦
