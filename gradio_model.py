import numpy as np 
import pandas as pd
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import gradio_model as gr

# Load the trained model
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

def predict_fraud(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file.name)
    
    # Preprocess the data
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    
    x = df.drop(['Class'], axis=1)
    stn_scaler = StandardScaler()
    x_scaled = stn_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled, columns=x.columns)
    
    # Make predictions
    predictions = clf.predict(X)
    
    # Return the predictions
    return ["Fraudulent" if pred == 1 else "Not Fraudulent" for pred in predictions]

# Create the Gradio interface
iface = gr.Interface(fn=predict_fraud,
                     inputs="file",
                     outputs="text",
                     title="Credit Card Fraud Detection",
                     description="Upload a CSV file containing credit card transactions data to detect fraudulent transactions.")

iface.launch()

