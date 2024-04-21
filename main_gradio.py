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

We implemented a flagging feature which keeps track on our server of any discoveries of fraudulent transaction activity. This feature can be extended in a number of ways, for example, automatic notification of credit card holders.

<hr>
"""

# Load the trained model
clf = load_model("neural_network_model.h5")

# Initialize global variables to store cumulative predictions
cumulative_predictions = []
cumulative_count = 0
prediction_counts = []

def predict_fraud(uploaded_file):
	global cumulative_predictions, cumulative_count, prediction_counts
	#df = pd.read_csv('sample.csv')
	"""
	if uploaded_file is None:
	df = pd.read_csv('sample.csv')
	else:
	df = pd.read_csv(uploaded_file)

	"""
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
	data = [2, 55] 
	plt.pie(data, labels = ['Fraud', 'Not Fraud'], colors=['red', 'green']) 
	plt.legend() 
	plt.grid(True) 
	plt.tight_layout() 
	plt.savefig("fraud.png") 
	plt.show()

	plt.figure(figsize=(10, 6))
	labels = ['Fraud', 'Not Fraud']
	plt.scatter(labels, data, color='red')
	plt.xlabel('Category')
	plt.ylabel('Count')
	plt.title('Fraud vs Not Fraud')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("fraud_scatter.png")
	plt.show()

	# 1. Boxplot
	plt.figure(figsize=(10, 6))
	sns.boxplot(x='category', y='value', data=data)
	plt.title('Boxplot of Values by Category')
	plt.xlabel('Category')
	plt.ylabel('Value')
	plt.tight_layout()
	plt.savefig("boxplot.png")
	plt.close()

	# 2. Scatter plot
	plt.figure(figsize=(10, 6))
	x = np.random.rand(100)
	y = np.random.rand(100)
	plt.scatter(x, y, color='blue')
	plt.title('Scatter Plot')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.tight_layout()
	plt.savefig("scatter_plot.png")
	plt.close()

	# 3. Bar plot
	plt.figure(figsize=(10, 6))
	x = ['A', 'B', 'C', 'D', 'E']
	y = [10, 20, 15, 25, 30]
	plt.bar(x, y, color='orange')
	plt.title('Bar Plot')
	plt.xlabel('Categories')
	plt.ylabel('Values')
	plt.tight_layout()
	plt.savefig("bar_plot.png")
	plt.close()

	# 4. Line plot
	plt.figure(figsize=(10, 6))
	x = np.arange(0, 10, 0.1)
	y = np.sin(x)
	plt.plot(x, y, color='purple')
	plt.title('Sine Wave')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.tight_layout()
	plt.savefig("line_plot.png")
	plt.close()

	# 5. Histogram
	plt.figure(figsize=(10, 6))
	data = np.random.normal(loc=0, scale=1, size=1000)
	plt.hist(data, bins=30, color='gray', alpha=0.7)
	plt.title('Histogram of Random Data')
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig("histogram.png")
	plt.close()


	return ["Fraudulent" if pred >= .5 else "Not Fraudulent" for pred in predictions], "cumulative_prediction_distribution.png", "fraud.png", "fraud_scatter.png", "boxplot.png", "scatter_plot.png", "bar_plot.png", "line_plot.png"


# Create the Gradio interface
iface = gr.Interface(
	fn=predict_fraud,

	inputs=gr.File(label="Upload a CSV File", value='sample.csv'),
	outputs=[
	gr.Text(label="Predictions"),
	gr.Image(label="Graph"),
	gr.Image(label="Graph"),
	gr.Image(label="Graph")
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
	let result = await (async () => {\
	var body = document.getElementsByTagName('body')[0];\
	var elem = document.createElement('img');\
	elem.src = 'https://i.imgur.com/Eg2ihWI.png';\
	elem.style.width = '50%';\
	elem.style.height = '50%';\
	elem.style.margin= 'auto auto auto auto';\
	body.appendChild(document.createElement('br'));\
	body.appendChild(document.createElement('br'));\
	body.appendChild(document.createElement('br'));\
	body.appendChild(elem);\
	})();\
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

