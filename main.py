import string
import pandas
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.ensemble import *
import numpy as np
from statistics import median

df = pandas.read_excel("city_stats.xlsx")

#    inputs
input_start_col = "E"  # from this column
input_end_col = "J"  # to this column
output_col = "O"  # y Column
predicting = False  # whether the model is trained with all the data
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

regressor = RandomForestRegressor(max_depth=3, min_samples_split=4, min_samples_leaf=2)

if predicting:
    regressor.fit(X_train, np.ravel(y_train))
    est = sm.OLS(np.ravel(y_train), X_train)
    est2 = est.fit()
    pvalues = est2.pvalues
    y_pred = regressor.predict(X_test)

    b_regressor = BaggingRegressor(regressor, n_estimators=100, max_features=len(X[0]),
                                   max_samples=.5)
    b_regressor.fit(X_train, np.ravel(y_train))
else:
    regressor.fit(X, np.ravel(y))
    est = sm.OLS(np.ravel(y), X)
    est2 = est.fit()
    pvalues = est2.pvalues
    y_pred = regressor.predict(X)

    b_regressor = BaggingRegressor(regressor, n_estimators=100, max_features=len(X[0]),
                                   max_samples=.5)
    b_regressor.fit(X, np.ravel(y))

X = [list(x) for x in zip(*X)]
atr_chg = []
for col in range(2, len(X)):
    atr_chg.append(median(X[col])/10)
X = [list(x) for x in zip(*X)]
print(atr_chg)
column_names_all = df.iloc[:, in_num1:in_num2].columns.values.tolist()
column_names = df.iloc[:, in_num1+2:in_num2].columns.values.tolist()

while True:
    print("")
    chars_to_remove = list(',$%')
    test_input = input("Enter " + ", ".join(column_names_all) + ": ").translate({ord(x): '' for x in chars_to_remove})
    test_input = [float(x) for x in test_input.split()]
    print("Input:")
    print(test_input)
    # print("Output:")
    # print(b_regressor.predict([test_input]))

    pred_change = []
    for index, value in enumerate(test_input[2:]):
        test_chg_max = test_input.copy()
        test_chg_min = test_input.copy()
        test_chg_max[index + 2] += atr_chg[index]
        test_chg_min[index + 2] -= atr_chg[index]
        pred_change.append([b_regressor.predict([test_chg_max])[0], b_regressor.predict([test_chg_min])[0]])
    improvement = [0]
    for p_index, col in enumerate(pred_change):
        for c_index, pred in enumerate(col):
            if pred > improvement[0]:
                improvement = [pred, column_names[p_index], ["increasing", "decreasing"][c_index], atr_chg[p_index]]
    print(f"Focusing on {improvement[2]} {improvement[1]} would be most effective.")
