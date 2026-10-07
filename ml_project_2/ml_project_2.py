# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================

# Read the CSV file using Pandas.
# The dataset contains Years of Experience and Salary.
dataset = pd.read_csv(
    r"/Users/aman/Documents/ai_ml_domination/ml_project_2/data_2.csv"
)

# ============================================================
# STEP 3: SEPARATE INDEPENDENT AND DEPENDENT VARIABLES
# ============================================================

# X = Independent variable / Input
# In our case: Years of Experience
#
# y = Dependent variable / Output
# In our case: Salary

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# ============================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

from sklearn.model_selection import train_test_split

# We divide the dataset into:
#
# 80% → Training data
#       Used to teach the model
#
# 20% → Testing data
#       Used later to check how well the model performs
#
# random_state=0 makes sure we get the same split
# every time we run the code.

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    train_size=0.8,
    random_state=0
)

# ============================================================
# STEP 5: CREATE THE LINEAR REGRESSION MODEL
# ============================================================

from sklearn.linear_model import LinearRegression

# Create an empty Linear Regression model.
#
# At this point the model has NOT learned anything yet.

regressor = LinearRegression()

# ============================================================
# STEP 6: TRAIN THE MODEL
# ============================================================

# fit() teaches the model using the training data.
#
# The model looks at:
#     x_train → Years of Experience
#     y_train → Actual Salary
#
# and tries to find the best-fitting straight line:
#     y = mx + c
#
# where:
#     m = coefficient / slope
#     c = intercept

regressor.fit(x_train, y_train)

# ============================================================
# STEP 7: SEE WHAT THE MODEL LEARNED
# ============================================================

# coef_ contains the coefficient (slope) learned by the model.
#
# In our example, it tells us approximately how much
# the predicted salary changes for every 1 additional
# year of experience.

print(f"Coefficient (m): {regressor.coef_}")

# intercept_ contains the intercept learned by the model.
#
# It represents the predicted salary when
# Years of Experience = 0.

print(f"Intercept (c): {regressor.intercept_}")

# Therefore, our model's equation is:
#
#     Salary = Coefficient × Experience + Intercept
#
# Example:
#
#     Salary = 9500 × Experience + 25000
#
# The actual numbers will come from our dataset.
