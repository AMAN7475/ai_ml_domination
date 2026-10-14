# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

dataset = pd.read_csv(r"/Users/aman/Documents/ai_ml_domination/ml_project_3/startup_profit_dataset.csv")

# ============================================================
# STEP 3: SEPARATE INPUT AND OUTPUT
# ============================================================

# X = Years of Experience
# y = Salary

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

x = pd.get_dummies(x,dtype=int)

# ============================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    train_size=0.8,
    random_state=0
)

# ============================================================
# STEP 5: CREATE AND TRAIN MULTI-LINEAR REGRESSION MODEL
# ============================================================

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

# ============================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================

y_predict = regressor.predict(x_test)

# ============================================================
# STEP 7: PRINT INTERCEPT AND SLOPE
# ============================================================

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

X = np.append(arr=np.full((36,1), 51761).astype(int), values=x, axis=1)

# ============================================================
# STEP 8: PERFORM ORDINARY LEAST SQUARES (OLS) REGRESSION
# ============================================================

import statsmodels.api as sm 

X_opt = X[:,[0,1,2,3,4,5]]
#Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()






