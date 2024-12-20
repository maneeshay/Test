
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

# Validate the shape of SHAP values and X_test
if isinstance(shap_values, list) and len(shap_values) > 1:
    # Visualize SHAP values for the positive class (class 1)
    st.write("Visualizing SHAP values for the positive class (1)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
    st.pyplot(fig)
elif isinstance(shap_values, np.ndarray):
    # For binary classification with single set of SHAP values
    st.write("Visualizing SHAP values for binary classification")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)
else:
    st.error("Error in SHAP value shape. Ensure compatibility with the model and dataset.")
