# ============================================
# Complex Spark SQL Queries
# ITCS 6190 — Spring 2026 — UNC Charlotte
# Real-Time Traffic Accident Severity Prediction
# ============================================

import builtins
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Complex_SQL_Queries") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/sample.csv")

df.createOrReplaceTempView("accidents")
print("✅ Dataset loaded:", df.count(), "rows")
print("=" * 60)

# ============================================
# Q1 — Severity Heatmap by Hour + Day of Week
# ============================================
print("\n🔍 Q1: Severity Heatmap — Hour of Day vs Day of Week")
print("    Finding which hour + day combo produces worst accidents")
spark.sql("""
    SELECT
        HOUR(Start_Time) as hour_of_day,
        CASE DAYOFWEEK(Start_Time)
            WHEN 1 THEN 'Sunday'
            WHEN 2 THEN 'Monday'
            WHEN 3 THEN 'Tuesday'
            WHEN 4 THEN 'Wednesday'
            WHEN 5 THEN 'Thursday'
            WHEN 6 THEN 'Friday'
            WHEN 7 THEN 'Saturday'
        END as day_of_week,
        COUNT(*) as total_accidents,
        ROUND(AVG(Severity), 3) as avg_severity
    FROM accidents
    GROUP BY hour_of_day, day_of_week
    HAVING COUNT(*) > 100
    ORDER BY avg_severity DESC
    LIMIT 10
""").show()

# ============================================
# Q2 — Danger Score per State
# ============================================
print("\n🔍 Q2: Danger Score per State (custom weighted formula)")
print("    Combines accident count + severity into one danger score")
spark.sql("""
    SELECT
        State,
        COUNT(*) as total_accidents,
        ROUND(AVG(Severity), 3) as avg_severity,
        SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as high_severity_count,
        ROUND(
            (AVG(Severity) * 0.4) +
            (SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) * 0.6),
        4) as danger_score
    FROM accidents
    GROUP BY State
    HAVING COUNT(*) > 500
    ORDER BY danger_score DESC
    LIMIT 10
""").show()

# ============================================
# Q3 — Weather Risk Index
# ============================================
print("\n🔍 Q3: Weather Risk Index per Condition")
print("    % of accidents that become high severity per weather type")
spark.sql("""
    SELECT
        Weather_Condition,
        COUNT(*) as total_accidents,
        SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as high_severity_count,
        ROUND(
            SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2) as risk_index_pct,
        ROUND(AVG(Severity), 3) as avg_severity
    FROM accidents
    WHERE Weather_Condition IS NOT NULL
    GROUP BY Weather_Condition
    HAVING COUNT(*) > 1000
    ORDER BY risk_index_pct DESC
    LIMIT 10
""").show()

# ============================================
# Q4 — Junction + Time of Day Compound Risk
# ============================================
print("\n🔍 Q4: Junction + Time of Day Compound Risk")
print("    Which junction type is most dangerous at which time of day")
spark.sql("""
    SELECT
        Junction,
        Traffic_Signal,
        CASE
            WHEN HOUR(Start_Time) BETWEEN 6 AND 9  THEN 'Morning Rush'
            WHEN HOUR(Start_Time) BETWEEN 10 AND 15 THEN 'Midday'
            WHEN HOUR(Start_Time) BETWEEN 16 AND 19 THEN 'Evening Rush'
            ELSE 'Night'
        END as time_of_day,
        COUNT(*) as accidents,
        ROUND(AVG(Severity), 3) as avg_severity,
        ROUND(
            SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2) as high_severity_pct
    FROM accidents
    GROUP BY Junction, Traffic_Signal, time_of_day
    HAVING COUNT(*) > 200
    ORDER BY avg_severity DESC
    LIMIT 12
""").show()

# ============================================
# Q5 — Top 10 Most Dangerous Road Segments
# ============================================
print("\n🔍 Q5: Top 10 Most Dangerous Road Segments")
print("    Streets with highest serious accident count and avg severity")
spark.sql("""
    SELECT
        Street,
        City,
        State,
        COUNT(*) as total_accidents,
        ROUND(AVG(Severity), 3) as avg_severity,
        MAX(Severity) as worst_severity,
        SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as serious_accidents
    FROM accidents
    WHERE Street IS NOT NULL
    GROUP BY Street, City, State
    HAVING COUNT(*) > 5
    ORDER BY serious_accidents DESC, avg_severity DESC
    LIMIT 10
""").show(truncate=False)

print("\n" + "=" * 60)
print("✅ All 5 Complex Queries Complete!")
print("=" * 60)

spark.stop()
