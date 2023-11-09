import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("Boston.csv", index_col=0)


print(data.head())
print(data.describe())
print(data.info())


X = data[
    [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "black",
        "lstat",
    ]
]
y = data["medv"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)

# Model performans
print("Ortalama Mutlak Hata:", mean_absolute_error(y_test, y_pred))
print("Ortalama Kare Hata:", mean_squared_error(y_test, y_pred))
print("Kök Ortalama Kare Hata:", np.sqrt(mean_squared_error(y_test, y_pred)))
