from statistics import LinearRegression
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("Collage.csv")


print(data)
print( data.describe())

#X = data.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
#y = data["Survived"]

## X['Age'].fillna(X['Age'].mean(), inplace=True)

#data["Fare"].fillna(data["Fare"].median(), inplace=True)

#age_means = data.groupby("Sex")["Age"].mean()


#for sex in ["male", "female"]:
#    X.loc[(X["Age"].isnull()) | (X["Age"].isna()) | (X["Age"] == 0), "Age"] = round(age_means[sex], 1)



