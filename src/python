import os
import pandas as pd

cmapss_folder = os.path.join(os.path.expanduser("~"), "Desktop", "CMaps")
train_file = os.path.join(cmapss_folder, "train_FD001.txt")

train_df = pd.read_csv(train_file, sep="\s+", header=None, engine="python")

columns = ["engine_id", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
train_df.columns = columns

train_df.sort_values(["engine_id", "cycle"], inplace=True)

print(train_df.head())
print(train_df.info())

from sklearn.preprocessing import MinMaxScaler

feature_cols = [col for col in train_df.columns if col.startswith("op_setting_") or col.startswith("sensor_")]

scaler = MinMaxScaler(feature_range=(0, 1))
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

print(train_df.head())

import joblib

joblib.dump(scaler, 'scaler.pkl')

import numpy as np

def create_sequences(data, window_size=30, step_size=1):
    sequences = []
    labels = []
    for engine_id in data["engine_id"].unique():
        engine_data = data[data["engine_id"] == engine_id].reset_index(drop=True)
        for i in range(0, len(engine_data) - window_size + 1, step_size):
            seq = engine_data.iloc[i:i+window_size][feature_cols].values  
            label = engine_data.iloc[i+window_size-1]["RUL"]               
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(train_df, window_size=30)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

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
        e = K.tanh(K.dot(inputs, self.W) + self.b)  
        e = K.squeeze(e, axis=-1)  
        alpha = K.softmax(e)  
        alpha_expanded = K.expand_dims(alpha, axis=-1)  
        context_vector = inputs * alpha_expanded  
        context_vector = K.sum(context_vector, axis=1)  
        return context_vector

time_steps = X.shape[1]  
num_features = X.shape[2]  

inputs = layers.Input(shape=(time_steps, num_features))
x = layers.LSTM(64, return_sequences=True)(inputs)
x = layers.LSTM(32, return_sequences=True)(x)
context = AttentionLayer()(x)
outputs = layers.Dense(1, activation="linear")(context)

lstm_attention_model = models.Model(inputs=inputs, outputs=outputs)
lstm_attention_model.summary()

lstm_attention_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = lstm_attention_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop]
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss (MSE)")
plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
plt.title("Model Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("Model MAE During Training")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.show()

val_loss, val_mae = lstm_attention_model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss (MSE): {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

from sklearn.metrics import r2_score

y_pred = lstm_attention_model.predict(X_val).flatten()

r2 = r2_score(y_val, y_pred)
print("R² score:", r2)

lstm_attention_model.save("lstm_rul_model.h5")

!pip install gym

import gym
from gym import spaces
import numpy as np
from gym.utils import seeding  

class MaintenanceEnv(gym.Env):
    def __init__(self, initial_rul=1.0, repair_cost=5.0, failure_penalty=100.0, degrade_rate=0.01, max_cycles=300):
        super(MaintenanceEnv, self).__init__()
        self.initial_rul = initial_rul
        self.repair_cost = repair_cost
        self.failure_penalty = failure_penalty
        self.degrade_rate = degrade_rate
        self.max_cycles = max_cycles
        
        self.action_space = spaces.Discrete(2)  
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.seed()  
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        reward = 0.0
        
        if action == 1:
            reward -= self.repair_cost
            self.current_rul = self.initial_rul
        else:
            self.current_rul -= self.degrade_rate

        self.current_cycle += 1
        
        if self.current_rul <= 0:
            reward -= self.failure_penalty
            self.current_rul = 0.0
            done = True

        if self.current_cycle >= self.max_cycles:
            done = True
        
        if not done:
            reward += 1.0
        
        next_state = np.array([self.current_rul], dtype=np.float32)
        return next_state, reward, done, {}

    def reset(self):
        self.current_cycle = 0
        self.current_rul = self.initial_rul
        return np.array([self.current_rul], dtype=np.float32)
    
    def render(self, mode="human"):
        print(f"Cycle: {self.current_cycle}, RUL: {self.current_rul:.2f}")

!pip install "shimmy>=2.0"

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = MaintenanceEnv()

vec_env = make_vec_env(lambda: env, n_envs=4)

ppo_agent = PPO("MlpPolicy", vec_env, learning_rate=0.0003, gamma=0.99, verbose=1)

ppo_agent.learn(total_timesteps=100000)


episodes = 5
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = ppo_agent.predict(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Episode {ep+1}: Total Reward = {total_reward}")

ppo_agent.save("ppo_maintenance_agent.zip")


import pandas as pd

txt_file_path = "~/Desktop/CMaps/test_FD002.txt"  
csv_file_path = "~/Desktop/CMaps/test_FD002.csv"  

df = pd.read_csv(txt_file_path, sep="\s+", header=None)  

column_names = ["id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + \
               [f"sensor_{i}" for i in range(1, 22)]  
df.columns = column_names  

df.to_csv(csv_file_path, index=False)

print(f"✅ Successfully converted to CSV: {csv_file_path}")
