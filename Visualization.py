import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('preprocessed_diabetes_dataset.csv')      #This is the Preprocessed Data

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

        # Add statistical summaries
        correlation_coefficient = df[[x_col, y_col]].corr().iloc[0, 1]

        # Print statistical summaries
        print('Scatter Plot: {} vs {}'.format(x_col, y_col))
        print('Correlation: {:.2f}'.format(correlation_coefficient))
        print('Mean {}: {:.2f}, Std {}: {:.2f}'.format(x_col, x.mean(), x_col, x.std()))
        print('Mean {}: {:.2f}, Std {}: {:.2f}'.format(y_col, y.mean(), y_col, y.std()))

        plt.show()
