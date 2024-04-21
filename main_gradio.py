import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from keras.models import load_model

# PHILIP ONLY, DO NOT USE THE PHILIP FLAG UNLESS YOU ARE PHILIP
# If the philip flag is passed, use the settings to deploy to racknerd.
remote = len(sys.argv) > 1 and sys.argv[1] == "--philip"

description = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-X3EM9MXYMM"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-X3EM9MXYMM');
</script>

<hr>
In the span of 12 hours, we brainstormed, designed, and implemented a full stack web application solving the problem of fraud detection in credit card transactions.

[Read about this project on Devpost!](https://devpost.com/software/capitalsavvy)

Upload a CSV file containing credit card transactions data to detect fraudulent transactions. If no file is uploaded, the default 'sample.csv' file will be used.
"""

# Load the trained model
clf = load_model("neural_network_model.h5")

# Initialize global variables to store cumulative predictions
cumulative_predictions = []
cumulative_count = 0
prediction_counts = []

def predict_fraud(uploaded_file):
    global cumulative_predictions, cumulative_count, prediction_counts
    if uploaded_file is None:
        df = pd.read_csv('sample.csv')
    else:
        df = pd.read_csv(uploaded_file)

    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)

    x = df.drop(['Class'], axis=1)
    stn_scaler = StandardScaler()
    x_scaled = stn_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled, columns=x.columns)

    predictions = clf.predict(X)

    cumulative_predictions.extend(predictions)
    cumulative_count += len(predictions)
    prediction_counts.append(cumulative_count)

    plt.figure(figsize=(10, 6))
    sns.countplot(x=np.concatenate(cumulative_predictions))
    plt.title('Cumulative Distribution of Predictions')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Fraudulent', 'Not Fraudulent'])
    plt.tight_layout()
    plt.savefig("cumulative_prediction_distribution.png")
    plt.close()

    # Plotting 
    plt.figure(figsize=(10, 6)) 
    data = [2, 50] 
    plt.pie(data, labels = ['Fraud', 'Not Fraud'], colors=['red', 'green']) 
    plt.legend() 
    plt.grid(True) 
    plt.tight_layout() 
    plt.savefig("fraud.png") 
    plt.show()




    # HTML content to display an image
    html_content = '<img src="fradpic.png" alt="Fraud Detection Visualization" style="width:100%;">'
    
    return ["Fraudulent" if pred >= .5 else "Not Fraudulent" for pred in predictions], "cumulative_prediction_distribution.png", "fraud.png", html_content

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_fraud,
    inputs=[gr.File(label="Upload a CSV File")],
    outputs=[
        gr.Text(label="Predictions"),
        gr.Image(label="Graph"),
        gr.Image(label="Graph"),
        gr.HTML(label="Fraud Detection Image")
    ],
    title="Capital Savvy",
    allow_flagging="manual",
    css="""
    body::after {
        content: "";
        position: fixed;
        right: 10px;
        bottom: 10px;
        width: 120px;
        height: 120px;
        background: url('fradpic.png') no-repeat center center;
        background-size: contain;
    }
    footer {
        visibility: hidden;  /* Hide the footer */
    }
    """,
    js="async () => {\
	  var imgTag = document.getElementById('imghere');\
	  \
	  imgTag.innerHTML= \"<img src=\'https://i.imgur.com/Eg2ihWI.png\'></img>\";\
	}",
    description=description
)

# Launch settings based on environment
if remote:
    iface.launch(share=False,
                 debug=False,
                 server_port=443,
                 ssl_certfile="../certs/fullchain.pem",
                 ssl_keyfile="../certs/privkey.pem",
                 server_name="capitalsavvy.app")
else:
    iface.launch(share=not remote)

