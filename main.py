import string
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn import metrics
import statsmodels.api as sm
from sklearn.ensemble import *
import numpy as np
import xgboost as xgb

df = pandas.read_excel("city_stats.xlsx")

#    inputs
input_start_col = "B"  # from this column
input_end_col = "B"  # to this column
output_col = "E"  # y Column
predicting = True  # whether the model is trained with all the data
#

in_num1 = 0
in_num2 = 0
out_num = 0
for c in input_start_col:
    if c in string.ascii_letters:
        in_num1 = in_num1 * 26 + (ord(c.upper()) - ord('A')) + 1
for c in input_end_col:
    if c in string.ascii_letters:
        in_num2 = in_num2 * 26 + (ord(c.upper()) - ord('A')) + 1
for c in output_col:
    if c in string.ascii_letters:
        out_num = out_num * 26 + (ord(c.upper()) - ord('A')) + 1
in_num1 -= 1
out_num -= 1

X = df.iloc[:, in_num1:in_num2].values.tolist()
y = df.iloc[:, out_num:out_num + 1].values.tolist()

if predicting:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = xgb.XGBRegressor()

if predicting:
    regressor.fit(X_train, np.ravel(y_train))
    est = sm.OLS(np.ravel(y_train), X_train)
    est2 = est.fit()
    pvalues = est2.pvalues
    y_pred = regressor.predict(X_test)

    b_regressor = BaggingRegressor(regressor, n_estimators=20, max_features=len(X[0]),
                                   max_samples=.5)
    b_regressor.fit(X_train, np.ravel(y_train))
else:
    regressor.fit(X, np.ravel(y))
    est = sm.OLS(np.ravel(y), X)
    est2 = est.fit()
    pvalues = est2.pvalues
    y_pred = regressor.predict(X)

    b_regressor = BaggingRegressor(regressor, n_estimators=20, max_features=len(X[0]),
                                   max_samples=.5)
    b_regressor.fit(X, np.ravel(y))

while True:
    test_input = [float(x) for x in input("Input: ").split()]
    print("Input:")
    print(test_input)
    print("Output:")
    print(b_regressor.predict([test_input]))
