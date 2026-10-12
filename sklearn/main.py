import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv("processed.csv")

print(df.head())

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test  = X.iloc[split:]

y_train = y.iloc[:split]
y_test  = y.iloc[split:]

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.intercept_)
print(model.coef_)

print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
