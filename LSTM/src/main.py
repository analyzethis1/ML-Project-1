import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Load and preprocess data
path = os.path.expanduser("~/Desktop/train_FD001.txt")
train = pd.read_csv(path, sep="\s+", header=None, engine="python").drop([24, 25], axis=1)
columns = ["id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 20)]
train.columns = columns
train["rul"] = train.groupby("id")["cycle"].transform(max) - train["cycle"]
sensor_cols = [col for col in train.columns if "sensor" in col]
scaler = MinMaxScaler()
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])

# Sequence generation
def create_sequences(data, window_size=30, step_size=1):
    sequences, labels = [], []
    for engine_id in data["id"].unique():
        engine_data = data[data["id"] == engine_id].reset_index(drop=True)
        for i in range(0, len(engine_data) - window_size, step_size):
            seq = engine_data.iloc[i:i+window_size][sensor_cols].values
            label = engine_data.iloc[i+window_size]["rul"]
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(train)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 30, -1)
X_val = X_val.reshape(X_val.shape[0], 30, -1)

# Model definition and training
model = Sequential([
    Input(shape=(30, 19)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Evaluation
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)
print(classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Save model and scaler
model.save("lstm_model.h5")
joblib.dump(scaler, "scaler.pkl")

# RL environment
def get_failure_probs(model, X):
    return model.predict(X).flatten()

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
        self.state = np.array([self.failure_probs[self.current_step], self.state[1] + 1 if action == 0 else 0.0], dtype=np.float32)
        self.current_step += 1
        return self.state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        return self.state, {}

failure_probs = get_failure_probs(model, X_train)
env = MaintenanceEnv(failure_probs)
model_rl = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, gamma=0.99)
model_rl.learn(total_timesteps=10_000)
model_rl.save("rl_maintenance_agent")

# Simulation
total_cost = 0
obs, _ = env.reset()
actions = []
for _ in range(len(failure_probs)):
    action, _ = model_rl.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_cost += abs(reward)
    actions.append(action)
    if done:
        break
print(f"Total cost: ${total_cost}")
print(f"Repairs scheduled: {sum(actions)}")

# Recommendation
def recommend_actions(new_sensor_data):
    sequences, _ = create_sequences(new_sensor_data)
    failure_probs_new = model.predict(sequences).flatten()
    env_new = MaintenanceEnv(failure_probs_new)
    obs, _ = env_new.reset()
    actions = []
    for _ in range(len(failure_probs_new)):
        action, _ = model_rl.predict(obs)
        actions.append(action)
        obs, _, _, _, _ = env_new.step(action)
    return actions
