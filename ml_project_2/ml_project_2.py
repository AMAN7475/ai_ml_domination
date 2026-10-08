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
dataset = pd.read_csv(r"/Users/aman/Documents/ai_ml_domination/ml_project_2/data_2.csv")

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


bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

# ============================================================
# STEP 8: MAKE PREDICTIONS
# ============================================================

# Now that the model has learned the relationship between
# Experience and Salary, we can use it to predict salaries.
#
# x_test contains experience values that the model
# did NOT use during training.

y_predict = regressor.predict(x_test)


# ============================================================
# STEP 8.1: CALCULATE SST, SSR AND SSE
# ============================================================

# Mean of the actual salary values in the test dataset.
y_mean = y_test.mean()

# ------------------------------------------------------------
# SST = Total Sum of Squares
# ------------------------------------------------------------
# Measures the total variation in actual salary
# around the average salary.
#
# Formula:
# SST = Σ(y_actual - y_mean)^2

SST = np.sum((y_test - y_mean) ** 2)


# ------------------------------------------------------------
# SSR = Regression Sum of Squares
# ------------------------------------------------------------
# Measures the variation explained by the
# Linear Regression model.
#
# Formula:
# SSR = Σ(y_predicted - y_mean)^2

SSR = np.sum((y_predict - y_mean) ** 2)

# ============================================================
# STEP 9: COMPARE ACTUAL VS PREDICTED SALARY
# ============================================================

# Create a DataFrame to compare:
#
# Actual    → Real salary from the dataset
# Predicted → Salary predicted by our model

comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_predict
})

print("\nActual vs Predicted Salary:")
print(comparison)


# ============================================================
# STEP 10: PREDICT SALARY FOR 12 YEARS OF EXPERIENCE
# ============================================================

# Linear Regression equation:
#     y = mx + c
# where:
#     m = regressor.coef_
#     x = 12 years of experience
#     c = regressor.intercept_
#
# IMPORTANT:
# coef_ is the slope (m)
# intercept_ is the intercept (c)

y_12year = regressor.coef_ * 12 + regressor.intercept_

print(f"\nPredicted salary for 12 years of experience: {y_12year}")

# ============================================================
# STEP 11: BASIC STATISTICAL ANALYSIS
# ============================================================

# Mean = Average value

print("\nMean:")
print(dataset.mean(numeric_only=True))


# Calculate the average Salary specifically.

print("\nAverage Salary:")
print(dataset['Salary'].mean())

# Median = Middle value after sorting the data.

print("\nMedian:")
print(dataset.median(numeric_only=True))


# Mode = Most frequently occurring value.

print("\nSalary Mode:")
print(dataset['Salary'].mode())


# describe() gives a quick statistical summary:
#
# count → Number of values
# mean  → Average
# std   → Standard deviation
# min   → Minimum
# 25%   → First quartile
# 50%   → Median
# 75%   → Third quartile
# max   → Maximum

print("\nStatistical Summary:")
print(dataset.describe())


# Variance tells us how spread out the values are
# from their average.

print("\nVariance:")
print(dataset.var(numeric_only=True))


# Standard deviation also tells us how spread out
# the values are.
#
# Unlike variance, standard deviation is expressed
# in the same units as the original data.

print("\nStandard Deviation:")
print(dataset.std(numeric_only=True))


# Correlation tells us how strongly two numerical
# variables are related.
#
# Correlation ranges from -1 to +1:
#
# +1 → Strong positive relationship
#  0 → No linear relationship
# -1 → Strong negative relationship

print("\nCorrelation:")
print(dataset.corr(numeric_only=True))


# Print the complete dataset.

print("\nComplete Dataset:")
print(dataset)


# ============================================================
# STEP 12: VISUALIZE TRAINING DATA
# ============================================================

# Scatter plot:
# Each red dot represents an actual employee from the training dataset.
plt.scatter(x_train,y_train,color='red')

# Draw the regression line learned by the model.
#
# regressor.predict(x_train)
# gives the salary predicted by the model
# for every training experience value.

plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# ============================================================
# STEP 13: VISUALIZE TEST DATA
# ============================================================
# Red dots = actual salary values from the test dataset.
plt.scatter(x_test,y_test,color='red')

# Blue line = predictions made by our trained
# Linear Regression model.

plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()