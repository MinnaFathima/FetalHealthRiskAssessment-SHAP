import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

# LOAD DATA
df = pd.read_csv("fetal_health.csv")

label_map = {1.0:0,2.0:1,3.0:2}
df["target"] = df["fetal_health"].map(label_map)

X = df.drop(["fetal_health","target"], axis=1)
y = df["target"]

# SPLIT DATA
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

# LOAD MODEL + SCALER
model = joblib.load("fetal_model.pkl")
scaler = joblib.load("scaler.pkl")

X_test_scaled = scaler.transform(X_test)

# PREDICTIONS
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# ----------------------------
# CONFUSION MATRIX
# ----------------------------

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=["Normal","Suspect","Pathological"],
    yticklabels=["Normal","Suspect","Pathological"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test,y_pred))


# ----------------------------
# ROC CURVE
# ----------------------------

y_test_bin = label_binarize(y_test,classes=[0,1,2])

plt.figure()

for i in range(3):

    fpr,tpr,_ = roc_curve(y_test_bin[:,i],y_prob[:,i])
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f"Class {i} AUC={roc_auc:.2f}")

plt.plot([0,1],[0,1],"k--")

plt.title("ROC-AUC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# ----------------------------
# PRECISION RECALL CURVE
# ----------------------------

plt.figure()

for i in range(3):

    precision,recall,_ = precision_recall_curve(y_test_bin[:,i],y_prob[:,i])

    plt.plot(recall,precision,label=f"Class {i}")

plt.title("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()


# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,6))

plt.title("Feature Importance")

plt.bar(
    range(len(importances)),
    importances[indices]
)

plt.xticks(
    range(len(importances)),
    X.columns[indices],
    rotation=90
)

plt.show()


# ----------------------------
# SHAP EXPLANATION
# ----------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar"
)

# === ENHANCED MULTICLASS SHAP ===
classes = ["Normal", "Suspect", "Pathological"]
shap_values = explainer.shap_values(X_test_scaled)  # List of 3 arrays

# 1. Global beeswarm (distribution)
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title("SHAP Beeswarm (All Classes)")
plt.show()

# 2. Per-class bar plots
for i, cls_name in enumerate(classes):
    plt.figure()
    shap.summary_plot(shap_values[i], X_test, plot_type="bar", 
                    max_display=10, show=False)
    plt.title(f"Top SHAP Features → {cls_name}")
    plt.show()

print("✅ FULL SHAP ANALYSIS COMPLETE")
