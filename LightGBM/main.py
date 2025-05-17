import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import time
import psutil


# Logging Helper

def log(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    memory = psutil.virtual_memory().percent
    print(f"[{timestamp} | Memory: {memory}%] {message}")


try:
    # Reproducibility
    log("Setting random seeds and environment variables...")
    random.seed(42)
    np.random.seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    log("Starting script...")

    # Load datasets efficiently
    dataset_folder = os.path.join(os.getcwd(), "datasets")
    log(f"Using dataset folder: {dataset_folder}")

    # Load full data without sampling for better model accuracy
    metadata = pd.read_csv(os.path.join(dataset_folder, "building_metadata.csv"), low_memory=True)
    train = pd.read_csv(os.path.join(dataset_folder, "train.csv"), low_memory=True)
    weather_train = pd.read_csv(os.path.join(dataset_folder, "weather_train.csv"), low_memory=True)
    log(f"Loaded CSVs: metadata ({len(metadata)} rows), train ({len(train)} rows), weather_train ({len(weather_train)} rows)")
    metadata = pd.read_csv(os.path.join(dataset_folder, "building_metadata.csv"), low_memory=True)
    train = pd.read_csv(os.path.join(dataset_folder, "train.csv"), low_memory=True).sample(frac=0.1, random_state=42)
    weather_train = pd.read_csv(os.path.join(dataset_folder, "weather_train.csv"), low_memory=True).sample(frac=0.1,
                                                                                                           random_state=42)
    log(f"Loaded CSVs: metadata ({len(metadata)} rows), train ({len(train)} rows), weather_train ({len(weather_train)} rows)")

    # Merge & Feature Engineering
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['weekend'] = df['timestamp'].dt.weekday >= 5
    df['day_of_month'] = df['timestamp'].dt.day
    df['quarter'] = df['timestamp'].dt.quarter
    log("Added additional time-based features")
    log("Merging datasets and creating features...")
    df = pd.merge(train, metadata, on='building_id', how='left')
    df = pd.merge(df, weather_train, on=['site_id', 'timestamp'], how='left')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['temp_diff'] = df['air_temperature'] - df['dew_temperature']
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['temp_diff_x_square_feet'] = df['temp_diff'] * df['square_feet']
    df['building_temp_diff'] = df['air_temperature'] * df['square_feet']
    df['site_hour'] = df['site_id'].astype(int) * df['hour']
    df['meter_reading'] = df['meter_reading'].clip(lower=1e-5)
    df['meter_reading_lag_1h'] = df.groupby('building_id')['meter_reading'].shift(1).fillna(0)
    df['meter_reading_lag_24h'] = df.groupby('building_id')['meter_reading'].shift(24).fillna(0)
    df['meter_reading_lag_7d'] = df.groupby('building_id')['meter_reading'].shift(24 * 7).fillna(0)
    df['hourly_meter_change'] = df['meter_reading'].diff().fillna(0)

    # Log-transform target
    df['meter_reading'] = np.log1p(df['meter_reading'])

    # Fill missing numeric values
    log("Filling missing numeric values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Add categorical features
    if 'primary_use' in df.columns:
        df['primary_use'] = df['primary_use'].astype('category')
        log("Converted 'primary_use' to categorical")
    log("Converting categorical features...")
    df['building_id'] = df['building_id'].astype('category')
    df['site_id'] = df['site_id'].astype('category')

    # Normalize features (except categorical)
    log("Normalizing numeric features...")
    numeric_features = [col for col in df.columns if
                        not pd.api.types.is_categorical_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(
                            df[col]) and col != 'timestamp']
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(scaler, 'outputs/scaler_lgb.pkl')

    log("Data preparation complete. Starting Optuna tuning...")

    # Define features and target
    feature_cols = [
        'air_temperature', 'dew_temperature', 'square_feet', 'floor_count',
        'sin_hour', 'cos_hour', 'month', 'week_of_year',
        'meter_reading_lag_1h', 'meter_reading_lag_24h', 'meter_reading_lag_7d',
        'temp_diff', 'hourly_meter_change', 'temp_diff_x_square_feet',
        'building_temp_diff', 'site_hour', 'building_id', 'site_id', 'primary_use'
    ]
    X = df[feature_cols]
    y = df['meter_reading']


    # Optuna Objective Function
    def objective(trial):
        param_grid = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'seed': 42
        }

        cv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            train_data = lgb.Dataset(X_train, label=y_train,
                                     categorical_feature=['building_id', 'site_id', 'primary_use'])
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=['building_id', 'site_id', 'primary_use'])
            model = lgb.train(param_grid, train_data, valid_sets=[val_data], num_boost_round=1000,
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
            preds = model.predict(X_val)
            mse = mean_squared_error(np.expm1(y_val), np.expm1(preds))
            score = np.sqrt(mse)
            scores.append(score)
        return np.mean(scores)


    # Run Optuna Study
    log("Starting Optuna study...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    log(f"Best params: {study.best_params}")
    joblib.dump(study, 'outputs/optuna_study.pkl')

    # Train Best Model
    best_params = study.best_params.copy()
    best_params.update(
        {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'boosting_type': 'gbdt', 'seed': 42})
    log("Training final model with best parameters...")
    final_model = lgb.train(best_params,
                            lgb.Dataset(X, label=y, categorical_feature=['building_id', 'site_id', 'primary_use']),
                            num_boost_round=1000)
    joblib.dump(final_model, 'outputs/best_lightgbm_model.pkl')

    # Model Evaluation
    log("Calculating feature importance...")
    importance_df = pd.DataFrame(
        {'feature': final_model.feature_name(), 'importance': final_model.feature_importance()}).sort_values(
        by='importance', ascending=False)
    importance_df.to_csv('outputs/feature_importance.csv', index=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.show()

    log("Evaluating final model...")
    y_pred = final_model.predict(X)
    y_exp = np.expm1(y)
    y_pred_exp = np.expm1(y_pred)
    rmse = np.sqrt(mean_squared_error(y_exp, y_pred_exp))
    mae = mean_absolute_error(y_exp, y_pred_exp)
    r2 = r2_score(y_exp, y_pred_exp)
    log(f"Final Model - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

    # Plot Predictions
    plt.figure(figsize=(6, 6))
    plt.scatter(y_exp[:5000], y_pred_exp[:5000], alpha=0.4)
    plt.xlabel('Actual Meter Reading')
    plt.ylabel('Predicted Meter Reading')
    plt.title('Actual vs Predicted (Sample)')
    plt.grid(True)
    plt.plot([y_exp.min(), y_exp.max()], [y_exp.min(), y_exp.max()], '--', color='gray')
    plt.tight_layout()
    plt.savefig('outputs/actual_vs_predicted.png')
    plt.show()

    log("Model training and evaluation complete.")

except Exception as e:
    log(f"Unexpected error: {e}")
    exit(1)
