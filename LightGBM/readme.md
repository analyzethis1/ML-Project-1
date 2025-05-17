# ML-Project-1: LightGBM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/StableBaselines3-Latest-green)](https://stable-baselines3.readthedocs.io/)
![License](https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0%20International-blue.svg)



## 📝 Overview

This LightGBM module is part of a larger ML-Project-1 framework that aims to provide real-time monitoring and analysis capabilities for building HVAC systems. While initially trained on historical ASHRAE data, the primary goal is to connect this model to BMS systems for real-time energy consumption monitoring and analysis.

**✨ Key Features:**
- 🚦 Real-time HVAC energy consumption monitoring
- 📉 Anomaly detection for identifying irregular consumption patterns
- 📊 Feature engineering optimized for temporal building data
- 🔧 Hyperparameter optimization through Optuna
- 🏢 BMS integration capabilities
- 🐳 Containerized deployment options

## 🛠️ Requirements

The following dependencies are required:
```
lightgbm>=3.3.2
pandas>=1.3.5
numpy>=1.21.6
scikit-learn>=1.0.2
matplotlib>=3.5.2
seaborn>=0.11.2
optuna>=2.10.1
joblib>=1.1.0
psutil>=5.9.0
```

## ⚙️ Setup

1. Navigate to the LightGBM directory:
   ```bash
   cd ML-Project-1/LightGBM
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🗂️ Data

The initial model training utilizes the ASHRAE energy prediction dataset. To train the model:

1. 📥 Download the ASHRAE dataset from [Kaggle](https://www.kaggle.com/c/ashrae-energy-prediction/data)
2. 📂 Place the following files in a `datasets` folder within the LightGBM directory:
   - `building_metadata.csv`
   - `train.csv`
   - `weather_train.csv`

**💡 Note:** While the model is initially trained on this dataset, it's designed to be connected to your BMS system for real-time monitoring and analysis of actual building data.

## 🧠 Model Architecture

The LightGBM model is specifically engineered for HVAC energy consumption analysis with the following characteristics:

- **🕒 Time-Series Focus**: Uses time-based features and lag variables to capture temporal patterns in energy usage
- **🌡️ Weather Integration**: Incorporates weather data to account for external environmental factors
- **🏢 Building Metadata**: Utilizes building-specific information to personalize predictions
- **🛠️ Feature Engineering**: Includes:
  - Temporal cyclical encoding (sin/cos hour transformations)
  - Historical lag features (1-hour, 24-hour, 7-day)
  - Temperature differentials and interaction features
  - Building-specific energy patterns

## 📝 Training Process

The training script (`train_model.py`) performs:

1. **🔄 Data Preparation**:
   - Loads and merges building metadata, energy consumption data, and weather information
   - Creates temporal features and engineered variables
   - Applies log-transformation to normalize skewed energy consumption values
   - Normalizes numerical features using RobustScaler

2. **🎯 Hyperparameter Optimization**:
   - Uses Optuna to find optimal LightGBM parameters
   - Implements time-series cross-validation to maintain temporal integrity
   - Optimizes for RMSE to ensure accurate predictions

3. **🧠 Model Training**:
   - Trains final model using optimal parameters
   - Handles categorical features appropriately
   - Evaluates model performance on historical data

## 🔗 Integration with BMS

This model is specifically designed to connect with Building Management Systems:

1. **🔌 Data Connector**: The model includes interfaces that can be adapted to most standard BMS data formats
2. **⚡ Real-Time Processing**: Optimized for efficient processing of streaming sensor data
3. **🔍 Monitoring Capabilities**: Can be used to:
   - Track real-time energy consumption against predictions
   - Identify potential inefficiencies in HVAC operation
   - Alert on anomalous energy usage patterns

## 📊 Evaluation

The model's performance is assessed using:

- **📐 RMSE (Root Mean Squared Error)**: Primary metric for overall accuracy
- **📊 MAE (Mean Absolute Error)**: For understanding average prediction deviation
- **📉 R²**: To assess the model's explanatory power

Evaluation outputs are saved to the `outputs/` directory, including:
- Actual vs. predicted scatter plots
- Performance metrics logs
- Feature importance visualization

## 📁 Outputs

The training process generates:

- **🗂️ Trained Model**: `outputs/best_lightgbm_model.pkl`
- **🔄 Feature Scaler**: `outputs/scaler_lgb.pkl`
- **📝 Optuna Study Results**: `outputs/optuna_study.pkl`
- **📊 Performance Visualizations**: Various plots in the `outputs/` directory

## 🚀 Usage

To train the model:
```bash
python train_model.py
```

For BMS integration, adapt the provided connectors in the integration directory to your specific BMS API or data format.

## 🐳 Docker Support

The module can be containerized for deployment:

```bash
# Build the Docker image
docker build -t lightgbm-hvac-monitor .

# Run the container
docker run -v $(pwd)/datasets:/app/datasets -v $(pwd)/outputs:/app/outputs lightgbm-hvac-monitor
```

This containerized approach allows for easier deployment in production environments and integration with BMS systems.

## 📝 License

[MIT License](LICENSE)

## 💻 Contributing

Interested in contributing to ML-Project-1? Please contact the repository owner for collaboration opportunities.

## 🛟 Help & Questions

For any questions or inquiries about this project, please feel free to contact me!
