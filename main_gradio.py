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
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

# Initialize global variables to store cumulative predictions
cumulative_predictions = []
cumulative_count = 0

def predict_fraud(csv_file):
    global cumulative_predictions, cumulative_count
    
    # Load the CSV file
    print("loading the csv file")
    df = pd.read_csv(csv_file.name)
    
    # Preprocess the data
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    
    x = df.drop(['Class'], axis=1)
    print("Scaling data...")
    stn_scaler = StandardScaler()
    x_scaled = stn_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled, columns=x.columns)
    
    # Make predictions
    print("Making predictions...")
    predictions = clf.predict(X)
    
    # Update cumulative predictions
    cumulative_predictions.extend(predictions)
    cumulative_count += len(predictions)
    
    # Generate visualization
    print("Generating visualization...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x=cumulative_predictions)
    plt.title('Cumulative Distribution of Predictions')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Fraudulent', 'Not Fraudulent'])
    plt.tight_layout()
    plt.savefig("cumulative_prediction_distribution.png")
    
    # Return the predictions and visualization
    print("Returning predictions...")
    return ["Fraudulent" if pred == 1 else "Not Fraudulent" for pred in predictions], "cumulative_prediction_distribution.png"

# Create the Gradio interface
iface = gr.Interface(fn=predict_fraud,
                     inputs="file",
                     outputs=["text", "image"],
                     title="Credit Card Fraud Detection",
                     allow_flagging="manual",
                     description="Upload a CSV file containing credit card transactions data to detect fraudulent transactions. The output includes prediction labels and a cumulative visualization of prediction distribution.")

iface.launch(share=True)


