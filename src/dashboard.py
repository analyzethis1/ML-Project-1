# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
import joblib
import plotly.express as px

# ---- Define the Custom Attention Layer ----
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)  # Attention scores
        e = K.squeeze(e, axis=-1)  # Remove last dimension
        alpha = K.softmax(e)  # Softmax over time dimension
        alpha_expanded = K.expand_dims(alpha, axis=-1)  # Expand dims for multiplication
        context_vector = inputs * alpha_expanded  # Weighted sum
        context_vector = K.sum(context_vector, axis=1)
        return context_vector

# ---- Function to Load Models and Scaler ----
@st.cache_resource
def load_models():
    # Define Mean Squared Error loss function explicitly
    def mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    # Load LSTM model with custom objects
    lstm_model = tf.keras.models.load_model(
        "/content/drive/MyDrive/ML-Project-1/lstm_model.h5",
        custom_objects={
            "AttentionLayer": AttentionLayer,  # Register the AttentionLayer
            "mse": mse  # Register mse loss function
        }
    )
    
    # Load RL agent
    rl_model = PPO.load("/content/drive/MyDrive/ML-Project-1/ppo_maintenance_agent.zip")
    
    # Load scaler
    scaler = joblib.load("/content/drive/MyDrive/ML-Project-1/scaler.pkl")
    
    return lstm_model, rl_model, scaler

# Load models and scaler
lstm_model, rl_model, scaler = load_models()

# ---- ✅ DASHBOARD TITLE ----
st.title("Predictive Maintenance Dashboard")

st.markdown("""
This dashboard helps predict when equipment might fail and suggests the best times for maintenance.  
It analyzes sensor data to estimate how much life is left in a machine and provides smart recommendation to reduce unexpected breakdowns.  

**Created by Chris Karim for demonstration purposes only.**  

Upload a CSV file with sensor data to see predictions and maintenance suggestions.""")

# File uploader for sensor data CSV file
uploaded_file = st.file_uploader("Upload Sensor Data (CSV)", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV file
    new_data = pd.read_csv(uploaded_file)
    
    # ---- ✅ Ensure all required feature columns exist ----
    feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]

    # Debugging: Print detected columns
    st.write("### Debugging Info: Available Columns in Uploaded Data")
    st.write(new_data.columns.tolist())  # Print columns for debugging

    # Identify missing columns
    missing_cols = [col for col in feature_cols if col not in new_data.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded CSV: {missing_cols}")
        st.stop()  # Prevent further execution

    # Normalize the sensor data using the trained scaler
    new_data[feature_cols] = scaler.transform(new_data[feature_cols])
    
    st.write("### Uploaded Data Preview", new_data.head())
    
    # Function to create sequences (using a sliding window of 30 cycles)
    def create_sequences(data, window_size=30):
        sequences = []
        for i in range(0, len(data) - window_size + 1):
            seq = data.iloc[i:i+window_size][feature_cols].values
            sequences.append(seq)
        return np.array(sequences)
    
    # Create sequences from the preprocessed data
    sequences = create_sequences(new_data, window_size=30)
    st.write(f"Created {sequences.shape[0]} sequences, each of shape ({30}, {len(feature_cols)})")
    
    if st.button("Analyze"):
        # Predict RUL for each sequence using the LSTM model
        rul_preds = lstm_model.predict(sequences).flatten()
        
        # Convert predicted RUL into a risk score (lower RUL implies higher risk)
        max_rul_reference = 130  # Adjust this reference based on your training data
        risk_scores = [max(0.0, min(1.0, 1 - (rul / max_rul_reference))) for rul in rul_preds]
        
        # Generate maintenance recommendations using the RL agent.
        # Here, the RL agent's state is a single value (the risk score).
        actions = []
        for risk in risk_scores:
            state = np.array([risk], dtype=np.float32)  # State shape is (1,)
            action, _ = rl_model.predict(state, deterministic=True)
            actions.append(action)
        
        # Plot predicted risk scores over time
        st.subheader("Predicted Failure Risk Over Time")
        fig = px.line(
            x=np.arange(len(risk_scores)),
            y=risk_scores,
            labels={"x": "Time Step", "y": "Failure Risk Score"},
            title="Failure Risk Score (0: low risk, 1: high risk)"
        )
        st.plotly_chart(fig)
        
        # Display maintenance recommendations
        st.subheader("Maintenance Recommendations")
        action_df = pd.DataFrame({
            "Time Step": np.arange(len(actions)),
            "Action": ["Repair 🔧" if a == 1 else "No Action ✅" for a in actions]
        })
        st.dataframe(action_df)
        
        st.write(f"Total Repairs Recommended: {sum(actions)}")
