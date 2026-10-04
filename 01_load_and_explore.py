"""
01_load_and_explore.py
Task 3 - Critical Analysis of Data Quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------
print("Loading Adult Income dataset from OpenML...")
adult = fetch_openml(data_id=1590, as_frame=True, parser="auto")
df = adult.frame

print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# -------------------------------------------------
# 2. Missing Values
# -------------------------------------------------
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])

# -------------------------------------------------
# 3. Duplicates
# -------------------------------------------------
print("\n" + "="*50)
print("DUPLICATES")
print("="*50)
print(f"Exact duplicate rows: {df.duplicated().sum()}")

# -------------------------------------------------
# 4. Class Distribution
# -------------------------------------------------
print("\n" + "="*50)
print("CLASS DISTRIBUTION")
print("="*50)
print(df["class"].value_counts())
print("\nProportions:")
print(df["class"].value_counts(normalize=True).round(4))

# -------------------------------------------------
# 5. Numerical Analysis
# -------------------------------------------------
num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

print("\n" + "="*50)
print("SKEWNESS")
print("="*50)
print(df[num_cols].skew().round(4))

print("\n" + "="*50)
print("OUTLIERS (IQR Method)")
print("="*50)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"{col:20s}: {outliers:5d} outliers ({outliers/len(df)*100:5.1f}%)")

print("\n" + "="*50)
print("CORRELATION MATRIX")
print("="*50)
print(df[num_cols].corr().round(3))

# -------------------------------------------------
# 6. Visualisations
# -------------------------------------------------
sns.set_theme(style="whitegrid")

# Class distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="class")
plt.title("Target Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()

# Missing values
plt.figure(figsize=(8, 5))
missing[missing > 0].plot(kind="bar", color="coral")
plt.title("Missing Values by Feature")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("missing_values.png", dpi=150)
plt.show()

# Histograms
df[num_cols].hist(bins=30, figsize=(12, 8), edgecolor="black")
plt.suptitle("Numerical Feature Distributions", y=1.02)
plt.tight_layout()
plt.savefig("numerical_histograms.png", dpi=150)
plt.show()

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i//3, i%3], color="steelblue")
    axes[i//3, i%3].set_title(col)
plt.suptitle("Boxplots - Outlier Detection")
plt.tight_layout()
plt.savefig("boxplots.png", dpi=150)
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

print("\nAll plots have been saved as PNG files.")
print("Exploration completed successfully!")