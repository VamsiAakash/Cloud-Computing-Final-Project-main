import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib, os
print("Loading data...")
df = pd.read_csv('data/sample.csv', low_memory=False)
feature_candidates = ['Temperature(F)','Humidity(%)','Visibility(mi)','Wind_Speed(mph)','Pressure(in)','Distance(mi)','Junction','Crossing','Traffic_Signal']
features = [c for c in feature_candidates if c in df.columns]
df2 = df[features + ['Severity']].dropna()
for c in ['Junction','Crossing','Traffic_Signal']:
    if c in df2.columns:
        df2[c] = df2[c].astype(str).map({'True':1,'False':0,'1':1,'0':0}).fillna(0).astype(int)
X = df2[features]
y = df2['Severity'].astype(int)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42,n_jobs=-1)
model.fit(X_train,y_train)
acc = model.score(X_test,y_test)
print(f"Accuracy: {acc*100:.2f}%")
joblib.dump(model,'models/rf_model.pkl')
print("Model saved to models/rf_model.pkl")
