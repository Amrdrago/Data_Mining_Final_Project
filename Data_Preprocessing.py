import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("diabetes_prediction_dataset.csv")
data = data[data['smoking_history'] != 'No Info']
# Checking missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Step 2: Encode categorical variables
categorical_cols = ['gender']  # Replace with actual categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Step 3: Normalize or scale numerical features
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']  # Replace with actual numerical columns
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_encoded[numerical_cols])
print(data.head(10).to_string())

# Replace the original numerical columns with scaled ones
data_encoded[numerical_cols] = data_scaled

data_encoded.drop('gender_Other', axis=1, inplace=True)

# Save preprocessed data to a new CSV file
data_encoded.to_csv("preprocessed_diabetes_dataset.csv", index=False)