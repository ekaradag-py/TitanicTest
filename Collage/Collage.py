import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

college2 = pd.read_csv('college1.csv', index_col=0)
college3 = college.rename({'Unnamed: 0': 'College'}, axis=1)
college3 = college3.set_index('College')



print(college2)
print(college2.describe())
print("--------------------------------")
print(college3)
print(college3.describe())


