# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:40:24 2024

@author: Administrator
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/Administrator/Desktop/Research Data/Dataset/DrivingDataset_19_Drivers (1).csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())
print(df.info())
print(df.describe())

# Convert 'AuthorisedClass' to numeric values
df['AuthorisedClass'] = df['AuthorisedClass'].map({'No': 0, 'Yes': 1})

# Convert 'F_M' to numeric values
df['F_M'] = df['F_M'].map({'F': 0, 'M': 1})

# Handling missing values by filling with the mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Define features and target variable
X = df.drop(columns=['AuthorisedClass'])
y = df['AuthorisedClass']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Use Sequential Feature Selector for feature selection
sfs = SFS(gnb,
          k_features=10,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(X_train, y_train)

# Get the selected features
selected_features = list(sfs.k_feature_names_)
print(f"Selected features: {selected_features}")

# Transform the datasets
X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)

# Train the model with the selected features
gnb.fit(X_train_sfs, y_train)

# Make predictions
y_pred = gnb.predict(X_test_sfs)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy after SFS: {accuracy:.2f}')

# Print the classification report
print(classification_report(y_test, y_pred))

# Plot feature importance
feature_importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': sfs.k_score_
})

# Sort the dataframe by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importance
print("Feature importance based on SFS scores:")
print(feature_importance_df)

# Plot the feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
#plt.title('Feature Importance based on SFS Scores')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
