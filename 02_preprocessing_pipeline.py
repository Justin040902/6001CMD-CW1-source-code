"""
02_preprocessing_pipeline.py
Task 4 - Data Preprocessing Strategy + Model Training
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
    print("imbalanced-learn is available → Using SMOTE")
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not found → Using class_weight='balanced' instead")

# -------------------------------------------------
# 1. Load Data
# -------------------------------------------------
print("Loading dataset...")
adult = fetch_openml(data_id=1590, as_frame=True, parser="auto")
df = adult.frame

# Remove duplicates
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# -------------------------------------------------
# 2. Prepare X and y
# -------------------------------------------------
X = df.drop(columns=["class"])
y = df["class"].map({"<=50K": 0, ">50K": 1})

num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]

# -------------------------------------------------
# 3. Build Preprocessing Pipeline
# -------------------------------------------------
def log1p_capital(X_arr):
    """Apply log1p transformation to capital-gain and capital-loss"""
    X_arr = X_arr.copy()
    X_arr[:, 3] = np.log1p(X_arr[:, 3])  # capital-gain
    X_arr[:, 4] = np.log1p(X_arr[:, 4])  # capital-loss
    return X_arr

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(log1p_capital)),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# -------------------------------------------------
# 4. Full Pipeline (with or without SMOTE)
# -------------------------------------------------
if HAS_SMOTE:
    model = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
else:
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

# -------------------------------------------------
# 5. Train / Test Split + Evaluation
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTraining the model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------
# 6. Cross-Validation
# -------------------------------------------------
print("\nRunning 5-Fold Cross Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("\nPipeline completed successfully!")