# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:40:24 2024

@author: Administrator
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load the dataset
file_path = 'C:/Users/Administrator/Desktop/Research Data/Dataset/DrivingDataset_19_Drivers (1).csv'
df = pd.read_csv(file_path)

# Randomize the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

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

# Normalize the features
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Convert target variable to categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the ANN model with regularization and dropout
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=4, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy with ANN: {accuracy:.2f}')

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_test_classes = y_test.argmax(axis=1)

# Print the classification report
print(classification_report(y_test_classes, y_pred_classes))
