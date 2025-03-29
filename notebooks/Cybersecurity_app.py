#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pandas as pd
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Data
def load_data():
    file_path = "../data/app_input.csv"
    df = pd.read_csv(file_path)
    return df

# Preprocess Data
def preprocess_data(df):
    selected_features = ["Protocol", "Packet Length", "Packet Type", "Traffic Type", "Malware Indicators", "Anomaly Scores"]
    target_column = "Attack Type"
    
    df = df[selected_features + [target_column]].dropna()
    
    label_encoder = {}
    for col in ["Protocol", "Packet Type", "Traffic Type", "Malware Indicators"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoder[col] = le
    
    X = df[selected_features]
    y = df[target_column]
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoder, le_target

# Train Model
def train_model():
    df = load_data()
    X, y, scaler, label_encoder, le_target = preprocess_data(df)
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(le_target, "le_target.pkl")

# Load Model
def load_model():
    if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
        train_model()
    
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    le_target = joblib.load("le_target.pkl")
    return model, scaler, label_encoder, le_target

def predict_attack_type(df, model, scaler, label_encoder):
    selected_features = ["Protocol", "Packet Length", "Packet Type", "Traffic Type", "Malware Indicators", "Anomaly Scores"]
    df = df[selected_features].dropna()
    
    for col in ["Protocol", "Packet Type", "Traffic Type", "Malware Indicators"]:
        df[col] = label_encoder[col].transform(df[col])
    
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    return predictions

# Load trained model (or train if missing)
model, scaler, label_encoder, le_target = load_model()

# Streamlit UI
st.title("🔍 Cyber Attack Detection App")
st.subheader("Upload a CSV file to predict attack types")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded file:")
    st.dataframe(df.head())
    
    if st.button("Predict Attack Type"):
        predictions = predict_attack_type(df, model, scaler, label_encoder)
        df["Predicted Action"] = le_target.inverse_transform(predictions)
        
        output_file = "predicted_attacks.csv"
        df.to_csv(output_file, index=False)
        
        st.success("Predictions completed!")
        st.dataframe(df)
        
        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_attacks.csv",
            mime="text/csv"
        )

# Sidebar for Insights
st.sidebar.header("🛡️ Attack Insights & Precautions")
st.sidebar.write("Select an attack type to view details")
st.sidebar.write("More insights will be dynamically added based on predictions")


# In[ ]:




