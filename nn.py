import os
import numpy as np
import pandas as pd
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Load data
df = pd.read_csv("creditcard_2023.csv")

# Preprocessing
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

# Shuffle data
df = shuffle(df)

# Split features and target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the model
model_path = 'neural_network_model.h5'
try:
    model.save(model_path)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error occurred while saving the model: {str(e)}")
