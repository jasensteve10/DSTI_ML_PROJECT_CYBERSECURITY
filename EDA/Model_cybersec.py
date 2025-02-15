#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipaddress
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[88]:


df=pd.read_csv("C:/Users/Utilisateur/Downloads/Project 1/Project 1/cybersecurity_attacks.csv")


# In[89]:


df.columns.tolist()


# In[90]:


df = df.drop(columns=['Timestamp','Source IP Address',
 'Destination IP Address',
 'Payload Data',
 'Attack Signature',
 'Action Taken',
 'Severity Level',
 'User Information',
 'Device Information',
 'Network Segment',
 'Geo-location Data',
 'Proxy Information',
 'Firewall Logs',
 'IDS/IPS Alerts',
 'Log Source'])


# In[91]:


df['Malware Indicators'] = df['Malware Indicators'].fillna('IOC not detected')
df['Alerts/Warnings'] = df['Alerts/Warnings'].fillna('Not triggered')


# In[92]:


df.head()


# In[93]:


# Label Encoding for Categorical Features
label_encoders = {}  # Store encoders for inverse transformation if needed

categorical_cols = ['Protocol', 'Packet Type', 'Traffic Type', 'Malware Indicators', 'Alerts/Warnings', 'Attack Type']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for later use



# In[94]:


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[95]:


# Assume X and y are your features and multi-class labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report (includes precision, recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-Validation Accuracy
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Note: For multi-class ROC-AUC, you can compute one-vs-rest scores
# This requires probability predictions and binarizing the output
y_prob = rf_model.predict_proba(X_test)
# Compute AUC for each class if needed (requires additional handling)
# Check Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
print(feature_importances.sort_values(by='Importance', ascending=False))



# In[96]:


# Train XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Accuracy
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))


# In[110]:


# Save Model & Scaler
import joblib
joblib.dump(rf_model, "model.pkl")


# In[112]:


import streamlit as st


# In[114]:


# Load Model and Scaler
model = joblib.load("model.pkl")

# Title
st.title("ML Model Attack Prediction App")

# Sidebar Input for Manual Prediction
st.sidebar.header("Enter Feature Values")

feature1 = st.sidebar.number_input("Feature 1", value=0.0)
feature2 = st.sidebar.number_input("Feature 2", value=0.0)

# Predict Button
if st.sidebar.button("Predict"):
    data_scaled = scaler.transform([[feature1, feature2]])
    prediction = model.predict(data_scaled)[0]
    st.sidebar.success(f"Prediction: {prediction}")

# File Upload for CSV
st.subheader("Upload CSV for Bulk Predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X_input = scaler.transform(df)
    predictions = model.predict(X_input)
    df["prediction"] = predictions

    # Show predictions
    st.write(df)

    # Download predictions
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")


# In[ ]:





# In[ ]:





# In[ ]:




