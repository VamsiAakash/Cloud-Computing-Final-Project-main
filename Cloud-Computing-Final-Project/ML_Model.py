# ============================================
# ML Model — Random Forest Classifier
# ITCS 6190 — Spring 2026 — UNC Charlotte
# Real-Time Traffic Accident Severity Prediction
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import time

spark = SparkSession.builder \
    .appName("Traffic_Severity_ML") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 55)
print("  TRAFFIC ACCIDENT SEVERITY PREDICTION — MLlib")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────
print("\n📂 Loading dataset...")
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/sample.csv")

print(f"   Rows    : {df.count():,}")
print(f"   Columns : {len(df.columns)}")

# ── Feature Engineering ───────────────────────────────
print("\n⚙️  Engineering features...")

df = df.withColumn("Hour_of_Day", hour("Start_Time")) \
       .withColumn("Day_of_Week", dayofweek("Start_Time")) \
       .withColumn("Is_Night", when(col("Sunrise_Sunset") == "Night", 1).otherwise(0)) \
       .withColumn("Junction_int", when(col("Junction") == True, 1).otherwise(0)) \
       .withColumn("Traffic_Signal_int", when(col("Traffic_Signal") == True, 1).otherwise(0))

# ── Select Features ───────────────────────────────────
feature_cols = [
    "Temperature(F)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Humidity(%)",
    "Precipitation(in)",
    "Distance(mi)",
    "Hour_of_Day",
    "Day_of_Week",
    "Is_Night",
    "Junction_int",
    "Traffic_Signal_int"
]

# Keep only needed columns and drop nulls in Severity
df = df.select(feature_cols + ["Severity"]).dropna(subset=["Severity"])

print(f"   Features used : {len(feature_cols)}")
print(f"   Rows after cleaning : {df.count():,}")

# ── Handle Nulls with Imputer ─────────────────────────
print("\n🔧 Imputing missing values with mean...")
numeric_cols = [
    "Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)",
    "Humidity(%)", "Precipitation(in)", "Distance(mi)"
]
imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols)

# ── Assemble Feature Vector ───────────────────────────
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

# ── Label: Severity (1-4) needs to be 0-indexed ───────
df = df.withColumn("label", col("Severity") - 1)

# ── Train / Test Split (time-based) ───────────────────
print("\n✂️  Splitting data — 80% train / 20% test...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"   Training rows : {train_df.count():,}")
print(f"   Test rows     : {test_df.count():,}")

# ── Random Forest Model ───────────────────────────────
print("\n🌲 Training Random Forest Classifier...")
print("   (This will take 2-5 minutes on sample data)")
start = time.time()

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(stages=[imputer, assembler, rf])

model = pipeline.fit(train_df)
elapsed = time.time() - start
print(f"   ✅ Training complete in {elapsed:.1f} seconds!")

# ── Predictions ───────────────────────────────────────
print("\n📊 Running predictions on test set...")
predictions = model.transform(test_df)

# ── Evaluation ────────────────────────────────────────
print("\n" + "=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)

# Accuracy
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = acc_evaluator.evaluate(predictions)
print(f"\n✅ Accuracy       : {accuracy:.4f} ({accuracy*100:.2f}%)")

# Weighted F1
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
f1 = f1_evaluator.evaluate(predictions)
print(f"✅ Weighted F1    : {f1:.4f} ({f1*100:.2f}%)")

# Weighted Precision
prec_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision = prec_evaluator.evaluate(predictions)
print(f"✅ Precision      : {precision:.4f} ({precision*100:.2f}%)")

# Weighted Recall
rec_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall = rec_evaluator.evaluate(predictions)
print(f"✅ Recall         : {recall:.4f} ({recall*100:.2f}%)")

# ── Confusion Matrix ──────────────────────────────────
print("\n📋 Confusion Matrix (Predicted vs Actual):")
predictions.groupBy("label", "prediction") \
    .count() \
    .orderBy("label", "prediction") \
    .show()

# ── Feature Importance ────────────────────────────────
print("\n🌟 Feature Importance (Top Features):")
rf_model = model.stages[-1]
importances = rf_model.featureImportances
feature_importance = sorted(
    zip(feature_cols, importances.toArray()),
    key=lambda x: x[1],
    reverse=True
)
for feat, imp in feature_importance:
    bar = "█" * int(imp * 100)
    print(f"   {feat:<25} {imp:.4f}  {bar}")

# ── Save Model ────────────────────────────────────────
print("\n💾 Saving model to disk...")
model.write().overwrite().save("models/rf_severity_model")
print("   ✅ Model saved → models/rf_severity_model")

print("\n" + "=" * 55)
print("  ML PIPELINE COMPLETE!")
print("=" * 55)
print(f"""
Summary:
  Algorithm  : Random Forest Classifier (MLlib)
  Features   : {len(feature_cols)} input features
  Training   : 80% of 100K sample
  Test       : 20% of 100K sample
  Accuracy   : {accuracy*100:.2f}%
  F1 Score   : {f1*100:.2f}%
""")

spark.stop()
