import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Read the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Step 2: Remove outliers using Interquartile Range (IQR) method
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_cleaned = data.copy()
for col in numerical_cols:
    data_cleaned = data_cleaned[(data_cleaned[col] >= lower_bound[col]) & (data_cleaned[col] <= upper_bound[col])]

# Step 3: Encode categorical variables
data_cleaned = data_cleaned[data_cleaned['smoking_history'] != 'No Info']
categorical_cols = ['gender', 'smoking_history']
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_cols, dtype=int)

# Step 4: Normalize or scale numerical features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_encoded[numerical_cols])

# Replace the original numerical columns with scaled ones
data_encoded[numerical_cols] = data_scaled

# Remove unnecessary columns


columns_to_drop = ['gender_Other','smoking_history_not current', 'smoking_history_never', 'smoking_history_former', 'smoking_history_ever', 'smoking_history_current', 'gender_Female', 'gender_Male']  # Replace 'Column1', 'Column2' with the columns you want to drop
data_encoded.drop(columns_to_drop, axis=1, inplace=True)

# Save preprocessed data to a new CSV file
data_encoded.to_csv("preprocessed_diabetes_dataset.csv", index=False)
