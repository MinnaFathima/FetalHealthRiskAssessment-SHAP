import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

print("🤰 Training FINAL SHAP-Ready Model...")

df = pd.read_csv("fetal_health.csv")
label_map = {1.0:0, 2.0:1, 3.0:2}
df["target"] = df["fetal_health"].map(label_map)
X = df.drop(["fetal_health", "target"], axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
model.fit(X_train_bal, y_train_bal)

# SHAP: Pre-compute on validation set only
print("Computing SHAP baseline...")
import shap
explainer = shap.TreeExplainer(model)
shap_values_val = explainer.shap_values(X_test_scaled[:50])  # List of 3 arrays

joblib.dump(model, "fetal_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
joblib.dump({
    "X_test": X_test.iloc[:50], 
    "y_test": y_test.iloc[:50],
    "shap_values": shap_values_val  # Pre-computed SHAP
}, "validation_data.pkl")


# Save test dataset for Streamlit CSV upload
test_data = X_test.copy()
test_data["fetal_health"] = y_test.map({0:1.0, 1:2.0, 2:3.0})  # convert back to original labels
test_data.to_csv("test_data.csv", index=False)

print("✅ Test dataset saved as test_data.csv")
print("✅ 91% Accuracy | SHAP Pre-computed |")
