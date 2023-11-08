
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("train.csv")


print(data)

X = data.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
y = data["Survived"]

# X['Age'].fillna(X['Age'].mean(), inplace=True)

data["Fare"].fillna(data["Fare"].median(), inplace=True)

age_means = data.groupby("Sex")["Age"].mean()


for sex in ["male", "female"]:
    X.loc[(X["Age"].isnull()) | (X["Age"].isna()) | (X["Age"] == 0), "Age"] = round(age_means[sex], 1)


data.describe()

X["Embarked"].fillna(X["Embarked"].mode()[0], inplace=True)

X = pd.get_dummies(X, columns=["Sex", "Embarked"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=11
)

# Modeli eğitimi
model = LogisticRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Model performansını değerlendir
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
