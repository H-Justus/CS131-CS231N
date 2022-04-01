# -*- coding: utf-8 -*-
# @Time    : 2022/2/20 2:10
# @Author  : Justus
# @FileName: linear regression.py
# @Software: PyCharm
"""
数据集:UCI Machine Learning Repository:Combined Cycle Power Plant Data Set
Data Set Information:

The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011),
when the power plant was set to work with full load.
Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH)
and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant.
A combined cycle power plant (CCPP) is composed of gas turbines (GT), steam turbines (ST)
and heat recovery steam generators. In a CCPP, the electricity is generated by gas and steam turbines,
which are combined in one cycle, and is transferred from one turbine to another.
While the Vacuum is colected from and has effect on the Steam Turbine,
he other three of the ambient variables effect the GT performance.
For comparability with our baseline studies, and to allow 5x2 fold statistical tests be carried out,
we provide the data shuffled five times. For each shuffling 2-fold CV is carried out
and the resulting 10 measurements are used for statistical testing.
We provide the data both in .ods and in .xlsx formats.


Attribute Information:

Features consist of hourly average ambient variables
- Temperature (T) in the range 1.81°C and 37.11°C,
- Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
- Relative Humidity (RH) in the range 25.56% to 100.16%
- Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
- Net hourly electrical energy output (EP) 420.26-495.76 MW
The averages are taken from various sensors located around the plant that record the ambient variables every second.
The variables are given without normalization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

data = pd.read_csv(r'.\ccpp.csv')
# print(data.head())
# print(data.tail())
# print(data.shape)  # (9568, 5)
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# print(X_train.shape)  # (7176, 4)
# print(y_train.shape)  # (7176, 1)
# print(X_test.shape)  # (2392, 4)
# print(y_test.shape)  # (2392, 1)
# scikit-learn的线性回归算法使用的是最小二乘法来实现的
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.intercept_)
print(lin_reg.coef_)

# 模型拟合测试集
y_pred = lin_reg.predict(X_test)
# 用scikit-learn计算MSE
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

X = data[['AT', 'V', 'AP']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# 模型拟合测试集
y_pred = lin_reg.predict(X_test)
# 用scikit-learn计算MSE
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
predicted = cross_val_predict(lin_reg, X, y, cv=10)
# 用scikit-learn计算MSE
print("MSE:", metrics.mean_squared_error(y, predicted))
# 用scikit-learn计算RMSE
print("RMSE:", np.sqrt(metrics.mean_squared_error(y, predicted)))

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors='k')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()