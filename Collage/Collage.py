import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


college = pd.read_csv("College.csv", index_col=0)
print(college.columns)
print("--------------------------------")
print(college.describe())

print(college)
