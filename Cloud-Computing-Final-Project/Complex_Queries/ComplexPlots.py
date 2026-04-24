# ============================================
# Complex Query Plots
# ITCS 6190 — Spring 2026 — UNC Charlotte
# ============================================

import builtins
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when, avg, count, round, sum
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import os

os.makedirs("plots", exist_ok=True)

spark = SparkSession.builder \
    .appName("Complex_Plots") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/sample.csv")

df.createOrReplaceTempView("accidents")
print("✅ Dataset loaded:", df.count(), "rows")

# ════════════════════════════════════════════
# PLOT 1 — Severity Heatmap Hour vs Day
# ════════════════════════════════════════════
print("\n📊 Plot 1 — Severity Heatmap Hour vs Day of Week...")
heatmap_df = spark.sql("""
    SELECT
        HOUR(Start_Time) as hour_of_day,
        CASE DAYOFWEEK(Start_Time)
            WHEN 1 THEN 'Sun'
            WHEN 2 THEN 'Mon'
            WHEN 3 THEN 'Tue'
            WHEN 4 THEN 'Wed'
            WHEN 5 THEN 'Thu'
            WHEN 6 THEN 'Fri'
            WHEN 7 THEN 'Sat'
        END as day_of_week,
        ROUND(AVG(Severity), 3) as avg_severity
    FROM accidents
    GROUP BY hour_of_day, day_of_week
""").toPandas()

pivot = heatmap_df.pivot(index="day_of_week", columns="hour_of_day", values="avg_severity")
day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
pivot = pivot.reindex(day_order)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=2.0, vmax=2.8)
ax.set_xticks(range(24))
ax.set_xticklabels(range(24), fontsize=8)
ax.set_yticks(range(7))
ax.set_yticklabels(day_order, fontsize=10)
plt.colorbar(im, ax=ax, label="Avg Severity")
ax.set_title("Accident Severity Heatmap — Hour of Day vs Day of Week",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Hour of Day (0 = Midnight)", fontsize=11)
ax.set_ylabel("Day of Week", fontsize=11)
plt.tight_layout()
plt.savefig("plots/complex_heatmap.png", dpi=150)
plt.show()
print("✅ Plot 1 saved → plots/complex_heatmap.png")

# ════════════════════════════════════════════
# PLOT 2 — Danger Score per State
# ════════════════════════════════════════════
print("\n📊 Plot 2 — Danger Score per State...")
danger_df = spark.sql("""
    SELECT
        State,
        COUNT(*) as total_accidents,
        ROUND(AVG(Severity), 3) as avg_severity,
        ROUND(
            (AVG(Severity) * 0.4) +
            (SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) * 0.6),
        4) as danger_score
    FROM accidents
    GROUP BY State
    HAVING COUNT(*) > 100
    ORDER BY danger_score DESC
    LIMIT 10
""").toPandas()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(danger_df["State"][::-1], danger_df["danger_score"][::-1],
               color="#DC2626", edgecolor="white")
ax.set_title("Danger Score per State\n(weighted formula: severity + high severity ratio)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Danger Score", fontsize=11)
for bar, val in zip(bars, danger_df["danger_score"][::-1]):
    ax.text(bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_facecolor("#F8FAFC")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plots/complex_danger_score.png", dpi=150)
plt.show()
print("✅ Plot 2 saved → plots/complex_danger_score.png")

# ════════════════════════════════════════════
# PLOT 3 — Weather Risk Index
# ════════════════════════════════════════════
print("\n📊 Plot 3 — Weather Risk Index...")
weather_df = spark.sql("""
    SELECT
        Weather_Condition,
        COUNT(*) as total_accidents,
        ROUND(
            SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2) as risk_index_pct,
        ROUND(AVG(Severity), 3) as avg_severity
    FROM accidents
    WHERE Weather_Condition IS NOT NULL
    GROUP BY Weather_Condition
    HAVING COUNT(*) > 1000
    ORDER BY risk_index_pct DESC
    LIMIT 8
""").toPandas()

fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#DC2626" if v >= 48 else "#EA580C" if v >= 43 else "#D97706"
          for v in weather_df["risk_index_pct"]]
bars = ax.barh(weather_df["Weather_Condition"][::-1],
               weather_df["risk_index_pct"][::-1],
               color=colors[::-1], edgecolor="white")
ax.set_title("Weather Risk Index — % of Accidents Becoming High Severity",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Risk Index (%)", fontsize=11)
ax.axvline(x=43, color="gray", linestyle="--", alpha=0.5, label="43% baseline")
for bar, val in zip(bars, weather_df["risk_index_pct"][::-1]):
    ax.text(bar.get_width() + 0.3,
            bar.get_y() + bar.get_height()/2,
            f"{val}%", va="center", fontsize=9)
ax.legend(fontsize=9)
ax.set_facecolor("#F8FAFC")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plots/complex_weather_risk.png", dpi=150)
plt.show()
print("✅ Plot 3 saved → plots/complex_weather_risk.png")

# ════════════════════════════════════════════
# PLOT 4 — Compound Risk Junction + Time
# ════════════════════════════════════════════
print("\n📊 Plot 4 — Junction + Time of Day Compound Risk...")
compound_df = spark.sql("""
    SELECT
        CONCAT(
            CASE WHEN Junction = true THEN 'Junction' ELSE 'No Junction' END,
            ' + ',
            CASE WHEN Traffic_Signal = true THEN 'Signal' ELSE 'No Signal' END
        ) as road_type,
        CASE
            WHEN HOUR(Start_Time) BETWEEN 6 AND 9  THEN 'Morning Rush'
            WHEN HOUR(Start_Time) BETWEEN 10 AND 15 THEN 'Midday'
            WHEN HOUR(Start_Time) BETWEEN 16 AND 19 THEN 'Evening Rush'
            ELSE 'Night'
        END as time_of_day,
        ROUND(AVG(Severity), 3) as avg_severity
    FROM accidents
    GROUP BY road_type, time_of_day
    HAVING COUNT(*) > 200
    ORDER BY avg_severity DESC
""").toPandas()

pivot2 = compound_df.pivot(index="road_type", columns="time_of_day", values="avg_severity")
fig, ax = plt.subplots(figsize=(10, 5))
im2 = ax.imshow(pivot2.values, aspect="auto", cmap="RdYlGn_r", vmin=2.0, vmax=2.8)
ax.set_xticks(range(len(pivot2.columns)))
ax.set_xticklabels(pivot2.columns, fontsize=10)
ax.set_yticks(range(len(pivot2.index)))
ax.set_yticklabels(pivot2.index, fontsize=9)
for i in range(len(pivot2.index)):
    for j in range(len(pivot2.columns)):
        val = pivot2.values[i][j]
        if not pd.isna(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
plt.colorbar(im2, ax=ax, label="Avg Severity")
ax.set_title("Compound Risk — Road Type vs Time of Day",
             fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("plots/complex_compound_risk.png", dpi=150)
plt.show()
print("✅ Plot 4 saved → plots/complex_compound_risk.png")

# ════════════════════════════════════════════
# PLOT 5 — Top 10 Dangerous Road Segments
# ════════════════════════════════════════════
print("\n📊 Plot 5 — Top 10 Most Dangerous Road Segments...")
roads_df = spark.sql("""
    SELECT
        CONCAT(Street, ', ', City) as road_label,
        COUNT(*) as total_accidents,
        ROUND(AVG(Severity), 3) as avg_severity,
        SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as serious_accidents
    FROM accidents
    WHERE Street IS NOT NULL
    GROUP BY Street, City, State
    HAVING COUNT(*) > 5
    ORDER BY serious_accidents DESC, avg_severity DESC
    LIMIT 10
""").toPandas()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(roads_df["road_label"][::-1],
               roads_df["serious_accidents"][::-1],
               color="#E87722", edgecolor="white")
ax.set_title("Top 10 Most Dangerous Road Segments\n(by serious accident count — Severity 3 or 4)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Serious Accidents (Severity 3 or 4)", fontsize=11)
for bar, val in zip(bars, roads_df["serious_accidents"][::-1]):
    ax.text(bar.get_width() + 2,
            bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.set_facecolor("#F8FAFC")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plots/complex_road_segments.png", dpi=150)
plt.show()
print("✅ Plot 5 saved → plots/complex_road_segments.png")

print("\n" + "=" * 55)
print("✅ All 5 Complex Plots Done!")
print("   Saved in → plots/ folder")
print("=" * 55)

spark.stop()
