import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

college = pd.read_csv("College.csv", index_col=0)
college["Private"] = college["Private"].replace({"Yes": 1, "No": 0})

print(college.columns)
print("--------------------------------")
print(college.describe())

X = college[["Private", "Apps", "Accept", "S.F.Ratio"]]
y = college["Grad.Rate"]

# Eksik değerleri doldurma
college = college.fillna(college.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Tahmin yap
y_pred = regressor.predict(X_test)

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #karekökü
