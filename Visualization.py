import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file

# Select Your CSV file that you want to Visualize

# df = pd.read_csv('diabetes_prediction_dataset.csv')           #This is the RAW Data

# df = pd.read_csv('preprocessed_diabetes_dataset.csv')      #This is the Preprocessed Data


print("Column Names:", df.columns)

# Get all column names except for the index
columns = df.columns.tolist()

# Generate scatter plots for all combinations of columns
for i in range(len(columns)) :
    for j in range(i + 1, len(columns)) :
        x_col = columns[i]
        y_col = columns[j]

        # Extract data from specified columns
        x = df[x_col]
        y = df[y_col]

        # Create a scatter plot
        plt.scatter(x, y)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title('Scatter Plot: {} vs {}'.format(x_col, y_col))
        plt.show()
