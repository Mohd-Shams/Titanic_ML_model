import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import os


df = pd.read_csv("Titanic-Dataset.csv")
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# ------------------------------
# Train-test split
# ------------------------------
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(df, df["Survived"]):
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]
df_test.to_csv('input1.csv',index=False)

labels = df_train["Survived"].copy()
features = df_train.drop("Survived", axis=1)

num_attribs = features.select_dtypes(include=[np.number]).columns.tolist()
cat_attribs = features.select_dtypes(include=["object"]).columns.tolist()


num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])
X_prepared = full_pipeline.fit_transform(features)
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}
results = {}
for name, model in classifiers.items():
    scores = cross_val_score(model, X_prepared, labels, cv=5, scoring="accuracy")
    results[name] = scores.mean()
    print(f"{name} Accuracy: {scores.mean():.4f}")

best_model_name = max(results, key=results.get)
best_model = classifiers[best_model_name]
best_model.fit(X_prepared, labels)

print(f"\n✅ Best Model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")

# ------------------------------
# Save best model & pipeline
# ------------------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(full_pipeline, "pipeline.pkl")
print("Model and pipeline saved successfully.")