import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

#8 (b)
college = pd.read_csv("College.csv", index_col=0)
college['Private'] = college['Private'].replace({'Yes': 1, 'No': 0})
print(college.columns)
print("--------------------------------")
print(college.describe())

# 9 (d) matrix oluşturma
pd.plotting.scatter_matrix(college[["Top10perc", "Apps", "Enroll"]])

# 9 (e) yan yana kutu grafikleri oluşturma
college.boxplot(column="Outstate", by="Private")

# (f) 'Elite' değişkenini oluşturmak ve sonrasında yan yana kutu grafikleri oluşturma
college["Elite"] = pd.cut(college["Top10perc"], [0, 0.5, 1], labels=["No", "Yes"])
print(college["Elite"].value_counts())
college.boxplot(column="Outstate", by="Elite")

# Apps
print("Apps aralığı:", college['Apps'].min(), "-", college['Apps'].max())
# Accept
print("Accept aralığı:", college['Accept'].min(), "-", college['Accept'].max())

# Apps
print("Apps ortalama ve standart sapma:", college['Apps'].mean(), ",", college['Apps'].std())
# Accept
print("Accept ortalama ve standart sapma:", college['Accept'].mean(), ",", college['Accept'].std())

# 10. ile 85. gözlemler arasındaki veri alt kümesi
subset_college = college.iloc[10:86, :]

# Aralık
print("Subset aralığı:")
print(subset_college.max() - subset_college.min())

# Yalnızca niceliksel özellikler içeren alt küme oluşturun
numeric_subset = subset_college.select_dtypes(include=['int64', 'float64'])

# Ortalama
print("Subset ortalama:")
print(numeric_subset.mean())

# Standart sapma
print("Subset standart sapma:")
print(numeric_subset.std())


# (g) 'hist()' yöntemini kullanarak histogramlar oluşturmak için:
fig, axes = plt.subplots(2, 2)
college.hist(column="Apps", bins=20, ax=axes[0, 0])
college.hist(column="Enroll", bins=20, ax=axes[0, 1])
college.hist(column="F.Undergrad", bins=20, ax=axes[1, 0])
college.hist(column="P.Undergrad", bins=20, ax=axes[1, 1])

plt.show()