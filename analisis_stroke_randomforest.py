import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

# 1. Load Data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# 2. Preprocessing

if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Tangani missing value pada 'bmi'
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encoding variabel kategorikal
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Split Data
X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']

# Normalisasi fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train-test sebelum SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Terapkan SMOTE pada data training
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Cek distribusi label sebelum dan sesudah SMOTE
print("Distribusi sebelum SMOTE:\n", y_train.value_counts())
print("Distribusi setelah SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# 5. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# 6. Evaluasi Model dengan Threshold Tuning
y_proba = rf.predict_proba(X_test)[:, 1]

# Default threshold 0.5
y_pred_default = (y_proba >= 0.5).astype(int)
print("\nClassification Report (Threshold = 0.5):\n")
print(classification_report(y_test, y_pred_default))

# Threshold yang diturunkan (misal 0.3)
threshold = 0.3
y_pred_threshold = (y_proba >= threshold).astype(int)
print(f"\nClassification Report (Threshold = {threshold}):\n")
print(classification_report(y_test, y_pred_threshold))

# ROC-AUC tetap dihitung dari probabilitas
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 7. Visualisasi

# Pastikan folder untuk menyimpan visualisasi tersedia
os.makedirs("visualisasi", exist_ok=True)

# a) Confusion Matrix (Threshold 0.3)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_threshold), annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (Threshold = {threshold})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("visualisasi/confusion_matrix_threshold_03.png")
plt.show()

# b) ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("visualisasi/roc_curve_random_forest.png")
plt.show()

# c) Feature Importance
importances = rf.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("visualisasi/feature_importance_top10.png")
plt.show()
