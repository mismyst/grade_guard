import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib  # For loading the saved model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the saved model
model = joblib.load("best_svm_model.pkl")

# Load the new testing dataset
df = pd.read_csv("new_student_dataset.csv")

# Strip whitespace from column names (if necessary)
df.columns = df.columns.str.strip()

# Check for NaN values and handle them (this may be redundant if your dataset is complete)
df.dropna(subset=["english.grade", "math.grade", "sciences.grade", "language.grade"], inplace=True)

# Create 'overall_grade' by summing up the grade columns
df["overall_grade"] = df["english.grade"] + df["math.grade"] + df["sciences.grade"] + df["language.grade"]

# Define the threshold for "top" students (using the 75th percentile)
threshold = df["overall_grade"].quantile(0.75)
df["top_low_studying"] = np.where(df["overall_grade"] >= threshold, 1, 0)  # 1 = Top, 0 = Low

# Define categorical and numerical columns
categorical_features = ["nationality", "city", "gender", "ethnic.group"]
numerical_features = ["english.grade", "math.grade", "sciences.grade", "language.grade"]

# Prepare the feature matrix (exclude target and irrelevant columns)
X_test = df.drop(columns=["top_low_studying", "overall_grade", "id", "name"], errors='ignore')
y_test = df["top_low_studying"]  # True labels for evaluation

# Make predictions using the loaded model
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))
print("Confusion Matrix on Test Data:\n", confusion_matrix(y_test, y_pred))
print("Classification Report on Test Data:\n", classification_report(y_test, y_pred))

# Add predictions to the dataframe and save it
df["predicted_top_low"] = y_pred
df.to_csv("new_student_dataset_with_predictions.csv", index=False)
print("Predictions saved in new_student_dataset_with_predictions.csv")
