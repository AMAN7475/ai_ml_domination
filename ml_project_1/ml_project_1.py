#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------------------
#import the dataset and divide into dependent and independent
dataset = pd.read_csv(r"/Users/aman/Documents/ai_ml_domination/ml_project_1/data_1.csv")

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#---------------------------------------------------
# Handling missing values in numerical columns (Age and Salary)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])


# Encoding the categorical State column into numerical values
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x. fit_transform(x[:,0])


# Encoding the target variable (Purchased) into numerical values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
