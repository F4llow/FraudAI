import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
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

def predict_fraud(uploaded_file):
    global cumulative_predictions, cumulative_count, prediction_counts

    if uploaded_file is None:
        # Load the default CSV file if no file is uploaded
        df = pd.read_csv('sample.csv')
    else:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)

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

    # Generate visualization for cumulative predictions over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(prediction_counts) + 1), y=prediction_counts)
    plt.title('Cumulative Predictions Over Time')
    plt.xlabel('Submission Count')
    plt.ylabel('Cumulative Prediction Count')
    plt.tight_layout()
    plt.savefig("cumulative_predictions_over_time.png")
    plt.close()

    # Generate count plot for cumulative prediction distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=cumulative_predictions)
    plt.title('Cumulative Distribution of Predictions')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Not Fraudulent', 'Fraudulent'])
    plt.tight_layout()
    plt.savefig("cumulative_prediction_distribution.png")
    plt.close()

    # Return the predictions and visualizations
    print("Returning predictions...")
    return ["Fraudulent" if pred == 1 else "Not Fraudulent" for pred in predictions], "cumulative_predictions_over_time.png", "cumulative_prediction_distribution.png"

# Create the Gradio interface
iface = gr.Interface(fn=predict_fraud,
                     inputs=gr.File(label="Upload a CSV File"),
                     outputs=["text", "image", "image"],
                     title="Credit Card Fraud Detection: Prediction Labels, Cumulative Predictions Over Time, and Cumulative Distribution of Predictions",
                     allow_flagging="manual",
                     description="Upload a CSV file containing credit card transactions data to detect fraudulent transactions. If no file is uploaded, the default 'sample.csv' file will be used.")

iface.launch(share=True)

