# ============================================================
# Makefile — Real-Time Traffic Accident Severity Prediction
# ITCS 6190 — Cloud Computing | UNC Charlotte | Spring 2026
# ============================================================
# Usage:
#   make install       → Install all dependencies
#   make train         → Train XGBoost model on full dataset
#   make train-spark   → Train Random Forest using Spark MLlib
#   make dashboard     → Launch Streamlit dashboard
#   make sql           → Run 5 complex Spark SQL queries
#   make plots         → Generate complex query plots
#   make all           → Run full pipeline (install → train → dashboard)
#   make clean         → Remove cached files
# ============================================================

PYTHON = python3
PIP    = pip3
DATA   = data/US_Accidents_March23.csv
SAMPLE = data/sample.csv
MODEL  = models/rf_model.pkl

.PHONY: all install train train-spark dashboard sql plots eda clean help

# ── Default target ────────────────────────────────────────────────────────────
all: install train dashboard

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "============================================================"
	@echo "  Real-Time Traffic Accident Severity Prediction"
	@echo "  ITCS 6190 — Cloud Computing | UNC Charlotte"
	@echo "============================================================"
	@echo ""
	@echo "  make install       Install all Python dependencies"
	@echo "  make train         Train XGBoost model (7.7M records)"
	@echo "  make train-spark   Train Random Forest via Spark MLlib"
	@echo "  make dashboard     Launch Streamlit dashboard"
	@echo "  make sql           Run 5 complex Spark SQL queries"
	@echo "  make plots         Generate complex query visualizations"
	@echo "  make all           Full pipeline: install → train → dashboard"
	@echo "  make clean         Remove cache and temporary files"
	@echo ""

# ── Install dependencies ──────────────────────────────────────────────────────
install:
	@echo "📦 Installing dependencies..."
	$(PIP) install --break-system-packages \
		streamlit==1.43.2 \
		pandas \
		numpy \
		scikit-learn \
		xgboost \
		plotly \
		joblib \
		pyspark
	@echo "✅ Dependencies installed!"

# ── Train XGBoost model on full 7.7M dataset ──────────────────────────────────
train:
	@echo "🌲 Training XGBoost model on full dataset..."
	@echo "   Dataset : $(DATA)"
	@echo "   Output  : $(MODEL)"
	$(PYTHON) retrain_xgb.py
	@echo "✅ Model trained and saved to $(MODEL)"

# ── Train Random Forest using Spark MLlib ─────────────────────────────────────
train-spark:
	@echo "⚡ Training Random Forest via Spark MLlib..."
	@echo "   Dataset : $(SAMPLE)"
	$(PYTHON) ML_Model.py
	@echo "✅ Spark MLlib model saved to models/rf_severity_model"

# ── Run complex Spark SQL queries ─────────────────────────────────────────────
sql:
	@echo "🔍 Running 5 Complex Spark SQL Queries..."
	$(PYTHON) ComplexQueries.py
	@echo "✅ All 5 complex queries complete!"

# ── Generate complex query plots ──────────────────────────────────────────────
plots:
	@echo "📊 Generating Complex Query Plots..."
	$(PYTHON) ComplexPlots.py
	@echo "✅ Plots saved to plots/ folder"

# ── Launch Streamlit dashboard ────────────────────────────────────────────────
dashboard:
	@echo "🖥️  Launching Streamlit Dashboard..."
	@echo "   Open browser at: http://localhost:8501"
	streamlit run dashboard_final.py

# ── Clean cache ───────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ Clean complete!"

# ── Check setup ───────────────────────────────────────────────────────────────
check:
	@echo "🔎 Checking project setup..."
	@test -f $(SAMPLE) && echo "✅ sample.csv found" || echo "❌ sample.csv MISSING - add to data/"
	@test -f $(DATA) && echo "✅ US_Accidents_March23.csv found" || echo "⚠️  Full dataset not found (optional)"
	@test -f $(MODEL) && echo "✅ rf_model.pkl found" || echo "⚠️  Model not found - run: make train"
	@test -f dashboard_final.py && echo "✅ dashboard_final.py found" || echo "❌ dashboard_final.py MISSING"
	@test -d plots && echo "✅ plots/ folder found" || echo "⚠️  plots/ folder missing - run: make plots"
	@echo ""
	@echo "Run 'make all' to set up and launch the full pipeline."
