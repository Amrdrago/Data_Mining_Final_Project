import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('preprocessed_diabetes_dataset.csv')

# Take a 10% sample
sample_data = data.sample(frac=0.1, random_state=42)

# Calculate the correlation matrix for the sample
corr_matrix = sample_data.corr()
print(corr_matrix.to_string())
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap (Sample)')
plt.show()

