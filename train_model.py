# Required libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # For saving the model

# Load the synthetic dataset
df = pd.read_csv("synthetic_student_dataset.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check for NaN values and handle them (dropping rows with NaN in grade columns)
df.dropna(subset=["english.grade", "math.grade", "sciences.grade", "language.grade"], inplace=True)

# Create 'overall_grade' by summing up the grade columns
df["overall_grade"] = df["english.grade"] + df["math.grade"] + df["sciences.grade"] + df["language.grade"]

# Adjust the threshold based on your distribution (using the 75th percentile)
threshold = df["overall_grade"].quantile(0.75)
df["top_low_studying"] = np.where(df["overall_grade"] >= threshold, 1, 0)  # 1 = Top, 0 = Low

# Define categorical and numerical columns
categorical_features = ["nationality", "city", "gender", "ethnic.group"]
numerical_features = ["english.grade", "math.grade", "sciences.grade", "language.grade"]

# Preprocess the data: handling missing values and encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean for numerical
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  # Fill missing values for categorical
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Define the target variable and feature matrix
X = df.drop(columns=["top_low_studying", "overall_grade", "id", "name"], errors='ignore')
y = df["top_low_studying"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that includes preprocessing and the classifier (SVC)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Hyperparameter tuning
param_grid = {
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto']
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Save the trained model and the preprocessor for future use
joblib.dump(grid_search.best_estimator_, "best_svm_model.pkl")

# Evaluate the model on the testing data
y_pred = grid_search.predict(X_test)

# Print evaluation metrics
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
