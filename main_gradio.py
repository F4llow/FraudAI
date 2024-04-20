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
prediction_counts = []

def predict_fraud(csv_file):
    global cumulative_predictions, cumulative_count, prediction_counts
    
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
    prediction_counts.append(cumulative_count)
    
    # Generate visualization
    print("Generating visualization...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(prediction_counts) + 1), y=prediction_counts)
    plt.title('Cumulative Predictions Over Time')
    plt.xlabel('Submission Count')
    plt.ylabel('Cumulative Prediction Count')
    plt.tight_layout()
    plt.savefig("cumulative_predictions_over_time.png")
    
    # Generate count plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=cumulative_predictions)
    plt.title('Cumulative Distribution of Predictions')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Not Fraudulent', 'Fraudulent'])
    plt.tight_layout()
    plt.savefig("cumulative_prediction_distribution.png")
    
    # Return the predictions and visualizations
    print("Returning predictions...")
    return ["Fraudulent" if pred == 1 else "Not Fraudulent" for pred in predictions], "cumulative_predictions_over_time.png", "cumulative_prediction_distribution.png"

# Create the Gradio interface
iface = gr.Interface(fn=predict_fraud,
                     inputs="file",
                     outputs=["text", "image", "image"],
                     title="Credit Card Fraud Detection: Prediction Labels, Cumulative Predictions Over Time, and Cumulative Distribution of Predictions",
                     allow_flagging="manual",
                     description="Upload a CSV file containing credit card transactions data to detect fraudulent transactions. The outputs include prediction labels, a line graph of cumulative predictions over time, and a count plot of the cumulative distribution of predictions.")


iface.launch(share=True)

