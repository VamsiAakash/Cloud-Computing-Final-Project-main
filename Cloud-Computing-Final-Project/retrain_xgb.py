import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading data...")
df = pd.read_csv('data/US_Accidents_March23.csv', low_memory=False)
print(f"Shape: {df.shape}")

if 'Start_Time' in df.columns:
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['Month'] = df['Start_Time'].dt.month
    df['Day_of_Week'] = df['Start_Time'].dt.dayofweek

le = LabelEncoder()
for col in ['Weather_Condition', 'State', 'Sunrise_Sunset']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

features = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)',
            'Pressure(in)', 'Distance(mi)', 'Hour', 'Day_of_Week', 'Month',
            'Junction', 'Crossing', 'Traffic_Signal',
            'Weather_Condition', 'State', 'Sunrise_Sunset']
features = [c for c in features if c in df.columns]
print(f"Using {len(features)} features")

df2 = df[features + ['Severity']].dropna()
for c in ['Junction', 'Crossing', 'Traffic_Signal']:
    if c in df2.columns:
        df2[c] = df2[c].astype(str).map({'True':1,'False':0,'1':1,'0':0}).fillna(0).astype(int)

X = df2[features]
y = df2['Severity'].astype(int) - 1  # XGBoost needs 0-indexed labels

print(f"Training on {len(X)} samples")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining XGBoost (2-3 minutes)...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*40}")
print(f"ACCURACY: {acc*100:.2f}%")
print(f"{'='*40}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['S1','S2','S3','S4']))

joblib.dump(model, 'models/rf_model.pkl')
print("\nModel saved to models/rf_model.pkl")
