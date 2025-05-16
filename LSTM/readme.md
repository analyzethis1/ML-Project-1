# PropertyPulse LSTM: Predictive Maintenance Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/StableBaselines3-Latest-green)](https://stable-baselines3.readthedocs.io/)

## Overview

PropertyPulse LSTM is a sophisticated predictive maintenance solution that combines Long Short-Term Memory (LSTM) neural networks with Reinforcement Learning (RL) to optimize building maintenance schedules. The system analyzes time series data from sensors to predict potential equipment failures and recommends optimal maintenance schedules to minimize costs and prevent downtime.

## Features

- **Failure Prediction**: LSTM-based model for accurate prediction of equipment failures based on sensor readings
- **Maintenance Optimization**: Reinforcement Learning (PPO) agent that determines the optimal timing for maintenance actions
- **Cost-Effective Decisions**: Balances repair costs against potential downtime costs for optimal resource allocation
- **BMS Integration**: Designed to connect with Building Management Systems (BMS) to monitor real-time sensor data
- **Scalable Pipeline**: Preprocessing, model training, and evaluation workflows for continuous improvement

## Architecture

The system consists of two main components:

1. **LSTM Predictive Model**: Processes time-series sensor data to predict the probability of equipment failure
2. **RL Maintenance Agent**: Uses the failure probabilities to make cost-optimized maintenance decisions

## Data

The model was initially trained on the [NASA Turbofan Engine Degradation Simulation Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) on Kaggle, which provides a rich source of degradation patterns. While developed using this dataset, the system is designed to be adaptable to building sensor data from BMS systems.

## Requirements

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

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PropertyPulse.git
cd PropertyPulse/LSTM

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Models

```python
# Run the training script
python train_model.py --data_path path/to/your/sensor_data.txt
```

### Making Predictions

```python
import joblib
from tensorflow.keras.models import load_model
from property_pulse.utils import preprocess_data, create_sequences

# Load models
lstm_model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")
rl_agent = joblib.load("rl_maintenance_agent.pkl")

# Preprocess your new data
processed_data = preprocess_data(your_sensor_data, scaler)
sequences, _ = create_sequences(processed_data)

# Get failure probabilities
failure_probs = lstm_model.predict(sequences).flatten()

# Get maintenance recommendations
from property_pulse.environment import MaintenanceEnv
env = MaintenanceEnv(failure_probs)
obs, _ = env.reset()
maintenance_schedule = []

for _ in range(len(failure_probs)):
    action, _ = rl_agent.predict(obs)
    maintenance_schedule.append(action)
    obs, _, done, _, _ = env.step(action)
    if done:
        break

# maintenance_schedule now contains optimal maintenance timings
# 1 = perform maintenance, 0 = no maintenance needed
```

## Model Details

### LSTM Model

- **Architecture**: Two-layer LSTM network (64 units followed by 32 units)
- **Input**: Sequences of sensor readings (window size: 30 timesteps)
- **Output**: Probability of failure
- **Training**: Binary cross-entropy loss with Adam optimizer

### Reinforcement Learning Agent

- **Algorithm**: Proximal Policy Optimization (PPO)
- **State Space**: Current failure probability and time since last maintenance
- **Action Space**: Binary (perform maintenance or not)
- **Reward Function**: Negative rewards for maintenance actions (-repair_cost) and failures (-downtime_cost)

## Performance Optimization

The model parameters can be tuned to balance between:

- False positives (unnecessary maintenance)
- False negatives (missed failures leading to downtime)
- Repair costs vs. downtime costs

## Integration with BMS

To integrate with your Building Management System:
1. Set up a data pipeline from your BMS to extract sensor readings in a compatible format
2. Configure the preprocessing pipeline to match your sensor data structure
3. Deploy the LSTM and RL models to make real-time predictions and recommendations
4. Connect the output recommendations to your maintenance scheduling system

## License

[MIT License](LICENSE)

## Contact

For any questions or inquiries, please contact: your.email@example.com
