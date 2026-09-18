'''Problem Statement: Sales Data Statistical Analysis

A company wants to analyze its product sales data to understand sales performance, variation in units sold and differences between product categories.

The objective is to create a synthetic sales dataset containing product information, categories, units sold and sale dates, and then perform statistical analysis on the data using Python.

The analysis should:
1. Create and store a synthetic sales dataset using Pandas and NumPy.
2. Calculate descriptive statistics such as mean, median, mode, variance and standard deviation for units sold.
3. Analyze sales performance by product category using total sales, average sales and standard deviation.
4. Calculate a 95% confidence interval for the average number of units sold.
5. Perform a one-sample t-test to determine whether the average number of units sold is significantly different from 20.
6. Visualize the distribution and variation of sales using histograms, boxplots, and bar charts.

The final analysis should help demonstrate how statistical techniques and data visualization can be used to understand sales data and draw conclusions from a sample dataset.
'''

#----------------------------------------------------------------------------------------
#Step-1 Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

#----------------------------------------------------------------------------------------
#Step:2 Create the Dataset

# Set the random seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset
data = {
    'product_id': range(1,21),
    'product_name': [f'Product ({i})' for i in range(1, 21)],
    'category': np.random.choice(['Electronics','Clothing', 'Home', 'Sports'], 20),
    'units_sold': np.random.poisson(lam=20, size=20),# Poisson distribution for sales
    'sale_date': pd.date_range(start='2023-01-01', periods=20, freq='D')
}

sales_data = pd.DataFrame(data)

# Display the first few rows of the dataset
print("Sales Data: ")
print(sales_data)

#Save the Dataframe as CSV file
sales_data.to_csv('salesdata.csv', index=False)

#Path location
os.getcwd()



