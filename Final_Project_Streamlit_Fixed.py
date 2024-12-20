import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
import io

# Title and description
st.title("Explainable AI for Healthcare Prediction Models: Diabetes Prediction")
st.write("This app predicts diabetes risk using Explainable AI techniques while providing interpretable insights.")

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
data = pd.read_csv(data_path)

# Show dataset overview
st.subheader("Dataset Preview")
st.write(data.head())

# Show dataset information
st.subheader("Dataset Information")
buffer = io.StringIO()
data.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

# Check for missing values
st.subheader("Missing Values")
st.write(data.isnull().sum())

# Statistical summary
st.subheader("Statistical Summary")
st.write(data.describe())

# Splitting the dataset
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model as an example
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model performance metrics
st.subheader("Model Performance")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# SHAP Feature Importance for Classification
st.subheader("Feature Importance using SHAP")

# Use TreeExplainer for tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Debugging: Check the structure of SHAP values
st.subheader("Debugging: SHAP Values Shape and Type")
st.write(f"SHAP values type: {type(shap_values)}")
st.write(f"SHAP values shape: {np.array(shap_values).shape}")

# Debugging: Display a slice of SHAP values
if isinstance(shap_values, list) and len(shap_values) > 1:
    st.write("SHAP values (positive class - class 1):")
    st.write(shap_values[1][:5])  # Display the first 5 rows of the positive class
else:
    st.write("SHAP values (binary classification):")
    st.write(shap_values[:5])  # Display the first 5 rows

# Visualize SHAP values based on classification type
if isinstance(shap_values, list) and len(shap_values) > 1:
    # For multi-class classification
    st.write("Visualizing SHAP values for the positive class (1)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
    st.pyplot(fig)
elif isinstance(shap_values, np.ndarray):
    # For binary classification
    st.write("Visualizing SHAP values for binary classification")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)
else:
    st.error("Error in SHAP value shape. Ensure compatibility with the model and dataset.")

# Alternative Visualization: Beeswarm plot
st.subheader("Alternative SHAP Visualization: Beeswarm Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X_test, show=False)
st.pyplot(fig)
