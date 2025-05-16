# ML-Project-1

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/StableBaselines3-Latest-green)](https://stable-baselines3.readthedocs.io/)
![License](https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0%20International-blue.svg)

## Overview
ML-Project-1 is a comprehensive Streamlit application that integrates multiple machine learning models to enhance property management through predictive maintenance and operational optimization. This platform leverages time series analysis and reinforcement learning to anticipate equipment failures, optimize maintenance schedules, and reduce costly downtime, empowering our partners to make data-driven decisions with greater confidence.

## 🎯 Key Objectives

- **Prevent Costly Failures**: Identify equipment issues before they lead to complete failure and service disruption
- **Optimize Maintenance Schedules**: Plan repairs at the most cost-effective times
- **Reduce Downtime**: Minimize service interruptions through proactive maintenance
- **Data-Driven Decision Making**: Provide portfolio managers with actionable insights
- **Competitive Edge**: Gain advantage through reduced operational costs and improved service reliability

## 🧠 Models

ML-Project-1 currently includes the following specialized models:

### LSTM Model

A sophisticated deep learning model designed for predictive maintenance based on sensor time series data.

- **Primary Use**: General equipment failure prediction
- **Key Features**: 
  - Sequence-based analysis of sensor readings
  - Failure probability estimation
  - Integration with reinforcement learning for optimal maintenance scheduling

[View LSTM Documentation](https://github.com/analyzethis1/ML-Project-1/blob/main/LSTM/readme.md)

### LightGBM Model

A gradient boosting framework optimized for HVAC system analysis and prediction.

- **Primary Use**: HVAC energy consumption and performance monitoring
- **Key Features**:
  - Fast training and inference time
  - High accuracy with temporal feature engineering
  - Hyperparameter optimization via Optuna
  - Robust handling of building-specific features

[View LightGBM Documentation](https://github.com/analyzethis1/ML-Project-1/blob/main/LightGBM/readme.md)

## 🔧 Technology Stack

### Core Dependencies

```
# Common
numpy
pandas
matplotlib
scikit-learn
joblib

# LSTM Module
tensorflow
gymnasium
stable-baselines3

# LightGBM Module
lightgbm
seaborn
optuna
psutil
```

## 📁 Project Structure

```
ML-Project-1/
├── LSTM/                       # LSTM-based predictive maintenance module
│   ├── Dockerfile              # Containerization for LSTM model
│   ├── requirements.txt        # LSTM-specific dependencies
│   ├── train_model.py          # LSTM training script
│   └── README.md               # LSTM module documentation
├── LightGBM/                   # LightGBM-based HVAC analysis module
│   ├── Dockerfile              # Containerization for LightGBM model
│   ├── requirements.txt        # LightGBM-specific dependencies
│   ├── train_model.py          # LightGBM training script
│   └── README.md               # LightGBM module documentation
├── LICENSE                     # Project license
└── README.md                   # Main project documentation (this file)
```

## ▶️ Getting Started

### Installation

Each model module can be installed and run independently:

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-project-1.git
cd ML-Project-1

# For LSTM model
cd LSTM
docker build -t ml-project-1-lstm .
docker run -p 8000:8000 ml-project-1-lstm

# For LightGBM model
cd ../LightGBM
docker build -t ml-project-1-lightgbm .
docker run -p 8001:8000 ml-project-1-lightgbm
```

### Using the Models

Each model module contains its own README with detailed usage instructions.

## 📊 Data Integration

This project is designed to connect with Building Management Systems (BMS) to ingest and analyze real-time sensor data. While the models are trained on benchmark datasets, they can be adapted to your specific property's data structure.

### Sample Datasets Used

- **LSTM Model**: NASA Turbofan Engine Degradation Simulation Dataset
- **LightGBM Model**: Building energy consumption dataset

## 👥 Target Users

- **Portfolio Management Office Teams**: For strategic maintenance planning and budget allocation
- **Building Managers**: For operational oversight and maintenance implementation
- **Service Managers**: For optimizing service team scheduling and resource allocation

## 🔄 Development Status

ML-Project-1 is currently in **Alpha** stage. Core functionality is implemented and operational, but the system is still undergoing active development and testing.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💻 Contributing

Interested in contributing to ML-Project-1? Please contact the repository owner for collaboration opportunities.

## 🛟 Help & Questions

For any questions or inquiries about this project, please feel free to contact me!
