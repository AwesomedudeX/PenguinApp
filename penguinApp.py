import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC as svc
from sklearn.linear_model import LogisticRegression as lgreg
from sklearn.ensemble import RandomForestClassifier as rfc

# Load the DataFrame
df = pd.read_csv('penguin.csv')

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create f and t variables
f = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
t = df['label']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = tts(f, t, test_size = 0.33, random_state = 42)

def predict(pm, island, bl, bd, fl, bmg, sex):
  pred = pm.predict([[island, bl, bd, fl, bmg, sex]])
  pred = pred[0]

  if pred == 0:
  	st.write(f"This is a Adelie penguin.")
  elif pred == 1:
  	st.write(f"This is a Chinstrap penguin.")
  else:
  	st.write(f"This is a Gentoo penguin.")

st.title("Penguin Classifier")
bl = st.slider("Bill Length (mm)", int(df['bill_length_mm'].min()), int(df['bill_length_mm'].max()))
bd = st.slider("Bill Depth (mm)", int(df['bill_depth_mm'].min()), int(df['bill_depth_mm'].max()))
fl = st.slider("Flipper Length (mm)", int(df['flipper_length_mm'].min()), int(df['flipper_length_mm'].max()))
bmg = st.slider("Body Mass (g)", int(df['body_mass_g'].min()), int(df['body_mass_g'].max()))

sex = st.selectbox("Sex:", ("Male", "Female"))
island = st.selectbox("Island", (0, 1, 2))

if sex == "Male":
	sex = 0
elif sex == "Female":
	sex = 1

clfs = [
	"Support Vector (Slower, but more accurate)",
	"Logistic Regression (Balanced)",
	"Random Forest Classifier (Faster, but less accurate)"
]

st.sidebar.title("Classifier:")

for i in clfs:
	st.sidebar.write(i)

clf = st.sidebar.selectbox("", ("Support Vector", "Logistic Regression", "Random Forest Classifier"))

if st.button("Predict"):
	if clf == "Support Vector":

		# Build a SVC model using the 'sklearn' module.
		svcm = svc(kernel = 'linear').fit(xtrain, ytrain)
		score = svcm.score(xtrain, ytrain)
		predict(svcm, island, bl, bd, fl, bmg, sex)
		st.write(f"Model accuracy score: {score}")

	elif clf == "Logistic Regression":

		# Build a LogisticRegression model using the 'sklearn' module.
		lr = lgreg().fit(xtrain, ytrain)
		score = lr.score(xtrain, ytrain)
		predict(lr, island, bl, bd, fl, bmg, sex)
		st.write(f"Model accuracy score: {score}")

	elif clf == "Random Forest Classifier":

		# Build a RandomForestClassifier model using the 'sklearn' module.
		rfcm = rfc(n_jobs = -1).fit(xtrain, ytrain)
		score = rfcm.score(xtrain, ytrain)
		predict(rfcm, island, bl, bd, fl, bmg, sex)
		st.write(f"Model accuracy score: {score}")

