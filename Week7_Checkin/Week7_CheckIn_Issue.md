# Week 7 Check-in — Data Ingestion + Exploratory Data Analysis (EDA)

**Course:** ITCS 6190 — Cloud Computing for Data Analysis
**Team:** Real-Time Traffic Accident Severity Prediction
**Date:** March 2026

---

## ✅ Progress Since Last Meeting

We had a productive week! Here is what we got done:

We locked in our dataset — the **US Accidents (2016–2023)** dataset from Kaggle by Sobhan Moosavi. This is a massive real-world dataset with **7,728,394 rows and 46 columns** covering 49 US states from February 2016 to March 2023. The file is about 3.2 GB which is well above the 50,000 row requirement.

We successfully loaded the full dataset using the **PySpark DataFrame API** and inspected the schema — all 46 columns came through correctly with the right data types. Timestamps loaded as timestamps, booleans as booleans, doubles as doubles — everything clean.

We ran a full **Exploratory Data Analysis (EDA)** including summary statistics, null value analysis, 5 visualizations, and 5 Spark SQL queries. All results are shown below.

**In short — we understand our dataset inside and out and we are ready to build on top of it.**

---

## 📊 Key EDA Findings

Here are the most important things we discovered about the dataset:

- The dataset has **7,728,394 total accident records** across 46 columns
- **Average Severity = 2.21** — meaning most accidents are on the lighter end
- **Severity 2 is by far the most common** — it makes up about 55% of all records
- **Peak accident hours are 11 AM and 8 PM** — rush hour and evening commute
- **Los Angeles, CA is the top city** with the most accidents in our sample
- **Junctions without traffic signals are the most dangerous** — avg severity of 2.586 compared to 2.155 for non-junction roads with signals
- This confirms that **Junction and Traffic_Signal** are critical features for our ML model

---

## 🗑️ Columns We Are Dropping from ML Features

Our null analysis revealed two columns with too many missing values to be useful:

- **Wind_Chill(F)** — 25.87% null (nearly 2 million missing values) → dropping this
- **End_Lat and End_Lng** — 44.03% null (over 3.4 million missing values) → dropping these

Everything else has less than 3% nulls so we are keeping the rest.

---

## 📈 Visualizations

**Plot 1 — Severity Distribution**

Shows that Severity 2 dominates at 55% of all records, followed by Severity 3 at 44.8%. Severity 1 and 4 are very rare. This class imbalance means we will use **Weighted F1-Score** as our primary ML evaluation metric rather than plain accuracy — because a model that just predicts Severity 2 for everything would look accurate but be totally useless.

**Plot 2 — Accidents by Hour of Day**

Clear peaks at morning rush (7–9 AM) and a bigger evening peak around 8 PM. Lowest accident frequency is between 3–5 AM as expected. This confirms Hour of Day is a strong predictor for our ML model.

**Plot 3 — Top 10 Cities by Accident Count**

Los Angeles leads by a huge margin with 9,792 accidents in our sample, followed by Sacramento and San Diego. All top 10 cities are in California — which reflects the geographic distribution of our 100K sample.

**Plot 4 — Average Severity by Weather Condition**

Partly Cloudy conditions actually show the highest average severity, followed by Scattered Clouds and Mostly Cloudy. All weather conditions score above the 2.0 baseline. This tells us Weather_Condition is an important feature but the relationship is more nuanced than simply "bad weather = worse accidents."

**Plot 5 — Accidents by Year**

Our 100K sample was heavily drawn from 2016 data. This is because the `head` sampling picked the first 100K rows which are from the earliest dates. For our final pipeline we will use the full 7.73M row dataset which covers all years evenly from 2016 to 2023.

---

## 🔍 Spark SQL Query Results

We ran 5 sample queries on the dataset to demonstrate our understanding:

**Q1 — Top States by Total Accidents**
California completely dominates with 99,272 out of 100,000 records in our sample. Ohio and West Virginia appear with very small counts. In the full 7.73M dataset, the distribution is more spread across all 49 states.

**Q2 — Rush Hour vs Off-Peak Severity**
Off-Peak hours had 72,899 accidents with avg severity 2.45, while Rush Hour had 27,101 accidents with avg severity 2.443. The severity is almost identical between the two — meaning accidents are consistently severe regardless of time of day. This is an interesting finding that will influence how we weight the Hour of Day feature in our ML model.

**Q3 — Weather Conditions by Severity**
Clear weather had the most accidents (57,255) with avg severity 2.44, followed by Overcast (10,255) with 2.43. Most accidents happening in clear weather makes sense — that is simply when most people are driving. Snow and fog produce fewer accidents in count but at higher severity levels.

**Q4 — Junction + Traffic Signal Impact**
This was our most insightful query. Junction=True with Signal=False produces the highest average severity of 2.586. Junction=False with Signal=True produces the lowest at 2.155. This tells us that traffic signals genuinely reduce accident severity at intersections — which validates both Junction and Traffic_Signal as key ML features.

**Q5 — Top Cities by High Severity Only (3 or 4)**
Los Angeles leads with 5,181 high severity accidents, followed by San Diego (2,246) and Sacramento (1,789). These are the exact types of incidents our real-time pipeline would detect and alert on immediately.

---

## 🚧 Current Challenges and Blockers

**Challenge 1 — Dataset Size**
The full 7.73M row CSV is 3.2 GB and runs very slowly on a local MacBook Air. We ran into timeout issues when trying to generate plots from the full dataset. We solved this by creating a 100K row sample for EDA visualization while keeping the full dataset linked externally on Kaggle for the final pipeline as per course guidelines.

**Challenge 2 — Python Version Compatibility**
We ran into a `distutils` module error because Python 3.13 removed it. We fixed this by installing `setuptools` which brought it back.

**Challenge 3 — PySpark round() Conflict**
Importing `round` from PySpark overrode Python's built-in `round` function causing a type error. We fixed this by using `builtins.round` for plain Python rounding and keeping PySpark's `round` only inside SQL queries.

---

## 📅 Plan for Next Week

Next week we are going to start building the actual pipeline on top of this EDA foundation. Here is what we are planning:

We are going to write at least 3 complex non-trivial **Spark SQL queries** — things like week-over-week accident growth by state, top road segments by rolling 15-minute severity, and weather-triggered high severity correlation. These will go into our Week 10 Pull Request with test assertions to validate correctness.

We are also going to start setting up the **Kafka producer** — the component that reads rows from our CSV and streams them one by one as if they are live accident reports coming in. This is the foundation of the entire streaming pipeline and we want it running before Week 11.

Finally we will start sketching out the **Spark Structured Streaming job** — the watermark settings, window sizes (5/15/60 minutes), and the schema definition for parsing the incoming JSON messages from Kafka.

By the end of next week we want to have data flowing from CSV through Kafka into Spark and see the first streaming output in our console.

---

## 📁 Repository Structure So Far

```
traffic_project/
├── EDA.py                         ← PySpark EDA script
├── data/
│   ├── sample_100k.csv            ← 100K sample for testing (in repo)
│   └── US_Accidents_March23.csv   ← Full 7.73M dataset (local only, linked via Kaggle)
└── plots/
    ├── eda_severity_distribution.png
    ├── eda_accidents_by_hour.png
    ├── eda_top_cities.png
    ├── eda_weather_vs_severity.png
    └── eda_accidents_by_year.png
```

**Dataset Link:** https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

---

*ITCS 6190 — Spring 2026 — UNC Charlotte*
