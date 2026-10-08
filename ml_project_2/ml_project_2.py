# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

dataset = pd.read_csv(
    r"/Users/aman/Documents/ai_ml_domination/ml_project_2/data_2.csv"
)


# ============================================================
# STEP 3: SEPARATE INPUT AND OUTPUT
# ============================================================

# X = Years of Experience
# y = Salary

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


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
# STEP 5: CREATE AND TRAIN LINEAR REGRESSION MODEL
# ============================================================

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)


# ============================================================
# STEP 6: MODEL COEFFICIENT AND INTERCEPT
# ============================================================

print(f"Coefficient (m): {regressor.coef_}")
print(f"Intercept (c): {regressor.intercept_}")

# Model equation:
# Salary = m × Experience + c


# ============================================================
# STEP 7: MODEL SCORE
# ============================================================

train_score = regressor.score(x_train, y_train)
test_score = regressor.score(x_test, y_test)

print(f"\nTraining Score: {train_score}")
print(f"Testing Score: {test_score}")


# ============================================================
# STEP 8: MAKE PREDICTIONS
# ============================================================

y_predict = regressor.predict(x_test)


# ============================================================
# STEP 9: CALCULATE SST, SSR AND SSE
# ============================================================

y_mean = y_test.mean()

# Total variation
SST = np.sum((y_test - y_mean) ** 2)

# Variation explained by the model
SSR = np.sum((y_predict - y_mean) ** 2)

# Prediction error
SSE = np.sum((y_test - y_predict) ** 2)

print("\nSum of Squares:")
print(f"SST: {SST}")
print(f"SSR: {SSR}")
print(f"SSE: {SSE}")

# Check: SST = SSR + SSE
print("\nChecking SST = SSR + SSE:")
print(f"SST:       {SST}")
print(f"SSR + SSE: {SSR + SSE}")


# ============================================================
# STEP 10: COMPARE ACTUAL VS PREDICTED
# ============================================================

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_predict
})

print("\nActual vs Predicted Salary:")
print(comparison)


# ============================================================
# STEP 11: PREDICT SALARY FOR 12 YEARS OF EXPERIENCE
# ============================================================

y_12year = regressor.coef_ * 12 + regressor.intercept_

print(f"\nPredicted salary for 12 years of experience: {y_12year}")


# ============================================================
# STEP 12: BASIC STATISTICAL ANALYSIS
# ============================================================

print("\nMean:")
print(dataset.mean(numeric_only=True))

print("\nAverage Salary:")
print(dataset["Salary"].mean())

print("\nMedian:")
print(dataset.median(numeric_only=True))

print("\nSalary Mode:")
print(dataset["Salary"].mode())

print("\nStatistical Summary:")
print(dataset.describe())

print("\nVariance:")
print(dataset.var(numeric_only=True))

print("\nStandard Deviation:")
print(dataset.std(numeric_only=True))

print("\nCorrelation:")
print(dataset.corr(numeric_only=True))

print("\nComplete Dataset:")
print(dataset)


# ============================================================
# STEP 13: VISUALIZE TRAINING DATA
# ============================================================

plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")

plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# ============================================================
# STEP 14: VISUALIZE TEST DATA
# ============================================================

plt.scatter(x_test, y_test, color="red")
plt.plot(x_test, regressor.predict(x_test), color="blue")

plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()