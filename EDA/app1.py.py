#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    return joblib.load("model.pkl")

# Load Model
model = load_model()

# Get feature names from the model
feature_names = model.feature_names_in_

# Streamlit UI
st.title("Cybersecurity Attack Prediction App")
st.sidebar.header("Enter Feature Values")

# File Upload for Bulk Predictions
st.subheader("Upload CSV for Bulk Predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

df_sample = None
if uploaded_file:
    df_sample = pd.read_csv(uploaded_file)
    missing_cols = [col for col in feature_names if col not in df_sample.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
    else:
        predictions = model.predict(df_sample[feature_names])
        df_sample["Predicted Attack Type"] = predictions
        st.write(df_sample)
        csv = df_sample.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# Sidebar Inputs with Real Values
def get_default_value(feature):
    if df_sample is not None and feature in df_sample.columns:
        return float(df_sample.iloc[0][feature])  # Use first row's value
    return 0.0  # Default value if no data available

user_input = []
for feature in feature_names:
    value = st.sidebar.number_input(feature, value=get_default_value(feature))
    user_input.append(value)

# Convert to DataFrame
input_df = pd.DataFrame([user_input], columns=feature_names)

# Predict attack type
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("Predicted Attack Type:")
    st.success(f"{prediction}")

# Feature Importance
st.subheader("Feature Importance")
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
fig, ax = plt.subplots()
sns.barplot(x=feature_importances['Importance'], y=feature_importances['Feature'], ax=ax)
st.pyplot(fig)


# In[ ]:




