# Heart Disease Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]

df = pd.read_csv(url, names=columns, na_values="?")
df["target"] = (df["num"] > 0).astype(int)
df.drop(columns=["num"], inplace=True)

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nUnique values:\n", df.nunique())
print("\nStatistics:\n", df[["age","chol"]].agg(["mean","median","std"]).T)

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="age", bins=15, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df[["chol","trestbps"]])
plt.title("Cholesterol and Resting Blood Pressure")
plt.show()

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
