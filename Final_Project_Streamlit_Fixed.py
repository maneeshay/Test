import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
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

# Reshape SHAP values to handle 3D array for binary classification
try:
    if isinstance(shap_values, list):
        # For multi-class classification, display SHAP values for the positive class
        st.write("SHAP values for positive class (class 1):")
        shap_values_class1 = shap_values[1]  # Extract SHAP values for class 1
        st.write(pd.DataFrame(shap_values_class1[:5], columns=X_test.columns))
    else:
        # For binary classification, handle 3D array
        shap_values_2d = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
        st.write("First 5 SHAP rows for binary classification:")
        st.write(pd.DataFrame(shap_values_2d[:5], columns=X_test.columns))
except Exception as e:
    st.error(f"Error displaying SHAP values: {e}")

# Visualize SHAP values
try:
    if isinstance(shap_values, list):
        # Visualize SHAP values for positive class
        st.write("Visualizing SHAP values for the positive class (1):")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
        st.pyplot(fig)
    else:
        # Visualize binary classification SHAP values
        st.write("Visualizing SHAP values for binary classification:")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values_2d, X_test, plot_type="bar", show=False)
        st.pyplot(fig)
except Exception as e:
    st.error(f"Error visualizing SHAP values: {e}")

# Alternative SHAP Visualization: Beeswarm plot
st.subheader("Alternative SHAP Visualization: Beeswarm Plot")
try:
    fig, ax = plt.subplots()
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, show=False)
    else:
        shap.summary_plot(shap_values_2d, X_test, show=False)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error displaying beeswarm plot: {e}")
