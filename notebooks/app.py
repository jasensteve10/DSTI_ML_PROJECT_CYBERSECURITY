import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests

from io import BytesIO

# Load Data
def load_data(uploaded_file):
    if uploaded_file is None:
        st.error("No file uploaded. Please upload a CSV file.")
        return None
    df = pd.read_csv(uploaded_file)
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


def load_model():

    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    le_target = joblib.load('model/le_target.pkl')

    st.warning("⚠️ Model files not found. Please upload a dataset and train the model first.")
    return model, scaler, label_encoder, le_target  # Prevent app crash

def predict_attack_type(df, model, scaler, label_encoder):
    selected_features = ["Protocol", "Packet Length", "Packet Type", "Traffic Type", "Malware Indicators", "Anomaly Scores"]
    df = df[selected_features].dropna()

    for col in ["Protocol", "Packet Type", "Traffic Type", "Malware Indicators"]:
        df[col] = label_encoder[col].transform(df[col])

    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    return predictions

# Streamlit UI
st.title("🔍 Cyber Attack Detection App")
st.subheader("Upload a CSV file to predict attack types")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded file:")
    # AgGrid(df)
    st.dataframe(df.head())

    # Load trained model
    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    le_target = joblib.load('model/le_target.pkl')

    if model is not None:
        if st.button("Predict Attack Type"):
            predictions = predict_attack_type(df, model, scaler, label_encoder)
            df["Predicted Action"] = le_target.inverse_transform(predictions)

            st.success("Predictions completed!")
            st.dataframe(df)
            # AgGrid(df)

            st.download_button(
                label="Download Predictions",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="predicted_attacks.csv",
                mime="text/csv"
            )

# Sidebar for Insights
st.sidebar.header("🛡️ Attack Insights & Precautions")
# Sidebar for insights and information
st.sidebar.image("https://t4.ftcdn.net/jpg/05/96/89/71/360_F_596897127_EZfIxmLrtfqUW0IFXgIh3qzHN3hxs0TP.jpg", use_container_width =True)  # Replace with your image URL or local file path

st.sidebar.write("""
**Cyber Attack Detection App** uses machine learning to predict cyber threats in network traffic.

### Features:
- **Upload CSV**: Analyze your network traffic data.
- **Predictions**: Classify attack types like malware and DDoS.
- **Download Results**: Get predictions in a CSV file.

### How It Works:
1. Upload a CSV.
3. Predict and download results.

### Notes:
- **Anomaly Scores**: Identify suspicious activity.
- **Data Quality**: Clean data for best results.

Thank you for using **Cyber Attack Detection App**—your network security tool.
""")


# st.sidebar.write("Select an attack type to view details")
# st.sidebar.write("More insights will be dynamically added based on predictions")
