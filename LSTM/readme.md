# ML-Project-1: LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/StableBaselines3-Latest-green)](https://stable-baselines3.readthedocs.io/)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Overview

This LSTM module is part of a larger ML-Project-1 framework that aims to revolutionize building maintenance operations. The primary goal is to connect this model to Building Management Systems (BMS) for real-time equipment monitoring, failure prediction, and maintenance schedule optimization to minimize costs and prevent downtime.

## ✨ Features

- **🔮 Failure Prediction**: LSTM-based model for accurate prediction of equipment failures based on sensor readings
- **⚙️ Maintenance Optimization**: Reinforcement Learning (PPO) agent that determines the optimal timing for maintenance actions
- **💰 Cost-Effective Decisions**: Balances repair costs against potential downtime costs for optimal resource allocation
- **🏢 BMS Integration**: Designed to connect with Building Management Systems (BMS) to monitor real-time sensor data
- **📈 Scalable Pipeline**: Preprocessing, model training, and evaluation workflows for continuous improvement

## 🏗️ Architecture

The system consists of two main components:

1. **🧠 LSTM Predictive Model**: Processes time-series sensor data to predict the probability of equipment failure
2. **🎮 RL Maintenance Agent**: Uses the failure probabilities to make cost-optimized maintenance decisions

## 📊 Data

The model was initially trained on the [NASA Turbofan Engine Degradation Simulation Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) on Kaggle, which provides a rich source of degradation patterns. While developed using this dataset, the system is designed to be adaptable to building sensor data from BMS systems.


## 📁 Project Structure

```
ML-Project-1/
├── LSTM/                       # LSTM-based predictive maintenance module
│   ├── Dockerfile              # Containerization for LSTM model
│   ├── requirements.txt        # LSTM-specific dependencies
│   ├── main.py                 # LSTM model training and RL agent
└── README.md                   # LSTM module documentation

```


## 📋 Requirements

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
joblib
gymnasium
stable-baselines3
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ML-Project-1.git
cd ML-Project-1/LSTM

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## 🐳 Docker Usage

The project includes a Dockerfile for easy deployment and isolation. To use Docker:

```bash
# Build the Docker image
docker build -t ml-project-lstm .

# Run the container with your data mounted
docker run -v /path/to/your/data:/data ml-project-lstm
```


## 🛠️ Usage

### Training the Models

```python
# Run the training script
python main.py --data_path path/to/your/sensor_data.txt
```

### Using the Model for Predictions

```python
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load models
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")
rl_agent = joblib.load("rl_maintenance_agent")

# Function to create sequences from new data
def create_sequences(data, sensor_cols, window_size=30, step_size=1):
    sequences = []
    for engine_id in data["id"].unique():
        engine_data = data[data["id"] == engine_id].reset_index(drop=True)
        for i in range(0, len(engine_data) - window_size, step_size):
            seq = engine_data.iloc[i:i+window_size][sensor_cols].values
            sequences.append(seq)
    return np.array(sequences)

# Preprocess new data
def process_new_data(new_data, scaler, sensor_cols):
    new_data[sensor_cols] = scaler.transform(new_data[sensor_cols])
    return new_data

# Generate maintenance recommendations
def recommend_actions(new_sensor_data, model, rl_agent, scaler, sensor_cols):
    processed_data = process_new_data(new_sensor_data, scaler, sensor_cols)
    sequences = create_sequences(processed_data, sensor_cols)
    failure_probs = model.predict(sequences).flatten()
    
    from gymnasium import spaces
    import numpy as np
    import gym
    
    class MaintenanceEnv(gym.Env):
        def __init__(self, failure_probs, repair_cost=1000, downtime_cost=5000):
            super().__init__()
            self.failure_probs = failure_probs
            self.repair_cost = repair_cost
            self.downtime_cost = downtime_cost
            self.current_step = 0
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 100.0]), dtype=np.float32)

        def step(self, action):
            done = self.current_step >= len(self.failure_probs) - 1
            if done:
                return self.state, 0.0, done, False, {}
            reward = -self.repair_cost if action == 1 else 0.0
            if action == 0 and self.failure_probs[self.current_step] > 0.5:
                reward -= self.downtime_cost
            self.state = np.array([self.failure_probs[self.current_step], 
                                  self.state[1] + 1 if action == 0 else 0.0], dtype=np.float32)
            self.current_step += 1
            return self.state, reward, done, False, {}

        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.state = np.array([0.0, 0.0], dtype=np.float32)
            return self.state, {}
    
    env = MaintenanceEnv(failure_probs)
    obs, _ = env.reset()
    actions = []
    
    for _ in range(len(failure_probs)):
        action, _ = rl_agent.predict(obs)
        actions.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    
    return actions
```

## 📈 Model Details

### 🧠 LSTM Model

- **Architecture**: Two-layer LSTM network (64 units followed by 32 units)
- **Input**: Sequences of sensor readings (window size: 30 timesteps)
- **Output**: Probability of failure
- **Training**: Binary cross-entropy loss with Adam optimizer

### 🎮 Reinforcement Learning Agent

- **Algorithm**: Proximal Policy Optimization (PPO)
- **State Space**: Current failure probability and time since last maintenance
- **Action Space**: Binary (perform maintenance or not)
- **Reward Function**: Negative rewards for maintenance actions (-repair_cost) and failures (-downtime_cost)

## ⚖️ Performance Optimization

The model parameters can be tuned to balance between:

- ❌ False positives (unnecessary maintenance)
- ❌ False negatives (missed failures leading to downtime)
- 💸 Repair costs vs. downtime costs

## 🔄 Integration with BMS

To integrate with your Building Management System:
1. Set up a data pipeline from your BMS to extract sensor readings in a compatible format
2. Configure the preprocessing pipeline to match your sensor data structure
3. Deploy the LSTM and RL models to make real-time predictions and recommendations
4. Connect the output recommendations to your maintenance scheduling system

## 📜 License

[MIT License](LICENSE)

## 💻 Contributing

Interested in contributing to ML-Project-1? Please contact the repository owner for collaboration opportunities.

## 🛟 Help & Questions

For any questions or inquiries about this project, please feel free to contact me!
