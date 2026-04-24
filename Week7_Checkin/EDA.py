# ============================================
# EDA — US Accidents Dataset (2016-2023)
# ITCS 6190 — Spring 2026
# UNC Charlotte
# ============================================

import builtins  # use Python's built-in round, not PySpark's

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, hour, year
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import os

os.makedirs("plots", exist_ok=True)

# ── Start Spark ──────────────────────────────
spark = SparkSession.builder \
    .appName("US_Accidents_EDA") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ── Load Dataset ─────────────────────────────
DATA_PATH = "data/sample.csv"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(DATA_PATH)

print("✅ Dataset loaded successfully!")
print(f"   Rows    : {df.count():,}")
print(f"   Columns : {len(df.columns)}")

df.createOrReplaceTempView("accidents")

# ── Schema ───────────────────────────────────
print("\n📋 Schema Overview:")
df.printSchema()

# ── Summary Stats ────────────────────────────
print("\n📊 Summary Statistics:")
df.select("Severity","Distance(mi)","Temperature(F)",
          "Visibility(mi)","Wind_Speed(mph)","Humidity(%)") \
  .describe().show()

# ── Null Value Analysis ──────────────────────
print("\n🔍 Null Value Count per Column:")
null_counts = []
total_rows = df.count()
for c in df.columns:
    null_count = df.filter(col(c).isNull()).count()
    null_pct = builtins.round(null_count / total_rows * 100, 2)
    null_counts.append((c, null_count, null_pct))

null_df = pd.DataFrame(null_counts,
    columns=["Column", "Null Count", "Null %"]) \
    .sort_values("Null %", ascending=False)

print(null_df[null_df["Null %"] > 0].to_string(index=False))

# ════════════════════════════════════════════
# PLOT 1 — SEVERITY DISTRIBUTION
# ════════════════════════════════════════════
print("\n📊 Plot 1 — Severity Distribution...")
sev_df = df.groupBy("Severity").count().orderBy("Severity").toPandas()

colors = ["#16A34A","#D97706","#EA580C","#DC2626"]
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(sev_df["Severity"].astype(str), sev_df["count"],
              color=colors, edgecolor="white", width=0.55)
ax.set_title("Accident Count by Severity Level", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Severity  (1 = Very Minor  →  4 = Severe)", fontsize=11)
ax.set_ylabel("Number of Accidents", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
for bar, row in zip(bars, sev_df.itertuples()):
    pct = row.count / sev_df["count"].sum() * 100
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30000,
            f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_facecolor("#F8FAFC")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plots/eda_severity_distribution.png", dpi=150)
plt.show()
print("✅ Plot 1 saved")

# ════════════════════════════════════════════
# PLOT 2 — ACCIDENTS BY HOUR OF DAY
# ════════════════════════════════════════════
print("\n📊 Plot 2 — Accidents by Hour...")
hourly = df.withColumn("hour", hour("Start_Time")) \
           .groupBy("hour").count().orderBy("hour").toPandas()

fig, ax = plt.subplots(figsize=(12,5))
ax.fill_between(hourly["hour"], hourly["count"], alpha=0.3, color="#0284C7")
ax.plot(hourly["hour"], hourly["count"], color="#0284C7",
        linewidth=2.5, marker="o", markersize=4)
ax.axvspan(7,  9,  alpha=0.15, color="#DC2626", label="Morning Rush (7–9 AM)")
ax.axvspan(16, 18, alpha=0.15, color="#EA580C", label="Evening Rush (4–6 PM)")
ax.set_title("Accident Frequency by Hour of Day", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Hour of Day  (0 = Midnight)", fontsize=11)
ax.set_ylabel("Number of Accidents", fontsize=11)
ax.set_xticks(range(0,24))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}K"))
ax.legend(fontsize=10)
ax.set_facecolor("#F8FAFC")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plots/eda_accidents_by_hour.png", dpi=150)
plt.show()
print("✅ Plot 2 saved")

# ════════════════════════════════════════════
# PLOT 3 — TOP 10 CITIES
# ════════════════════════════════════════════
print("\n📊 Plot 3 — Top 10 Cities...")
top_cities = df.groupBy("City","State").count() \
               .orderBy(col("count").desc()).limit(10).toPandas()
top_cities["label"] = top_cities["City"] + ", " + top_cities["State"]

fig, ax = plt.subplots(figsize=(10,6))
bars = ax.barh(top_cities["label"][::-1], top_cities["count"][::-1],
               color="#E87722", edgecolor="white")
ax.set_title("Top 10 Cities by Total Accident Count", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Number of Accidents", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}K"))
for bar, val in zip(bars, top_cities["count"][::-1]):
    ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.set_facecolor("#F8FAFC")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plots/eda_top_cities.png", dpi=150)
plt.show()
print("✅ Plot 3 saved")

# ════════════════════════════════════════════
# PLOT 4 — WEATHER VS SEVERITY
# ════════════════════════════════════════════
print("\n📊 Plot 4 — Weather vs Severity...")
weather_sev = df.groupBy("Weather_Condition") \
                .agg(avg("Severity").alias("avg_severity"), count("*").alias("total")) \
                .filter(col("total") > 5000) \
                .orderBy(col("avg_severity").desc()).limit(12).toPandas()

fig, ax = plt.subplots(figsize=(11,6))
colors_w = ["#DC2626" if v>=2.5 else "#EA580C" if v>=2.2 else "#D97706"
            for v in weather_sev["avg_severity"]]
ax.barh(weather_sev["Weather_Condition"][::-1],
        weather_sev["avg_severity"][::-1],
        color=colors_w[::-1], edgecolor="white")
ax.set_title("Average Severity by Weather Condition\n(conditions with 5,000+ accidents)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Average Severity Score", fontsize=11)
ax.set_xlim(1.8, 3.2)
ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="Severity 2.0 baseline")
ax.legend(fontsize=9)
ax.set_facecolor("#F8FAFC")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plots/eda_weather_vs_severity.png", dpi=150)
plt.show()
print("✅ Plot 4 saved")

# ════════════════════════════════════════════
# PLOT 5 — ACCIDENTS BY YEAR
# ════════════════════════════════════════════
print("\n📊 Plot 5 — Accidents by Year...")
yearly = df.withColumn("year", year("Start_Time")) \
           .groupBy("year").count().orderBy("year").toPandas()
yearly = yearly[yearly["year"].between(2016,2023)]

fig, ax = plt.subplots(figsize=(9,5))
ax.bar(yearly["year"].astype(str), yearly["count"],
       color="#0D1B2A", edgecolor="#E87722", linewidth=1.5, width=0.6)
ax.set_title("Total Accidents per Year (2016–2023)", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Number of Accidents", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
for i, row in enumerate(yearly.itertuples()):
    ax.text(i, row.count+15000, f"{row.count:,}",
            ha="center", fontsize=8.5, fontweight="bold", color="#0D1B2A")
ax.set_facecolor("#F8FAFC")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plots/eda_accidents_by_year.png", dpi=150)
plt.show()
print("✅ Plot 5 saved")

# ════════════════════════════════════════════
# SPARK SQL QUERIES
# ════════════════════════════════════════════
print("\n" + "="*55)
print("📊 SPARK SQL — SAMPLE QUERIES")
print("="*55)

print("\n🔍 Q1: Top 10 States by Total Accidents")
spark.sql("""
    SELECT State, COUNT(*) as total_accidents,
           ROUND(AVG(Severity), 2) as avg_severity
    FROM accidents
    GROUP BY State ORDER BY total_accidents DESC LIMIT 10
""").show()

print("\n🔍 Q2: Rush Hour vs Off-Peak Severity")
spark.sql("""
    SELECT
      CASE WHEN HOUR(Start_Time) BETWEEN 7 AND 9
                OR HOUR(Start_Time) BETWEEN 16 AND 18
           THEN 'Rush Hour' ELSE 'Off Peak' END AS time_period,
      COUNT(*) as accident_count,
      ROUND(AVG(Severity), 3) as avg_severity
    FROM accidents GROUP BY time_period ORDER BY avg_severity DESC
""").show()

print("\n🔍 Q3: Top Weather Conditions by Severity")
spark.sql("""
    SELECT Weather_Condition, COUNT(*) as total,
           ROUND(AVG(Severity), 2) as avg_severity,
           SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as high_severity_count
    FROM accidents WHERE Weather_Condition IS NOT NULL
    GROUP BY Weather_Condition HAVING COUNT(*) > 10000
    ORDER BY avg_severity DESC LIMIT 8
""").show()

print("\n🔍 Q4: Junction + Traffic Signal Impact")
spark.sql("""
    SELECT Junction, Traffic_Signal,
           COUNT(*) as accidents,
           ROUND(AVG(Severity), 3) as avg_severity
    FROM accidents GROUP BY Junction, Traffic_Signal
    ORDER BY avg_severity DESC
""").show()

print("\n🔍 Q5: Top 10 Cities — High Severity Only (3 or 4)")
spark.sql("""
    SELECT City, State, COUNT(*) as high_severity_accidents
    FROM accidents WHERE Severity >= 3
    GROUP BY City, State ORDER BY high_severity_accidents DESC LIMIT 10
""").show()

print("\n✅ EDA Complete!")
print("   5 plots saved → /plots folder")
print("   5 SQL query results printed above")
print("   Upload plots to your GitHub Issue!")

spark.stop()
