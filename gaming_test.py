import streamlit as st
import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import sys
import time
import matplotlib.pyplot as plt
import pickle
import os

st.title("Reinforcement Learning AI Comparison")

try:
    class DoubleDQNAgent:
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=10000)
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.999
            self.learning_rate = 0.001
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

        def _build_model(self):
            model = keras.Sequential([
                keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
                keras.layers.Dense(24, activation='relu'),
                keras.layers.Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
            return model

        def update_target_model(self):
            self.target_model.set_weights(self.model.get_weights())

        def act(self, state):
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])

        def load(self, name):
            self.model.load_weights(name)

    # Define the relative path for the models
    model_dir = "trained_model"

    # Create both environments
    cartpole_env = gym.make("CartPole-v1", render_mode="rgb_array")
    taxi_env = gym.make("Taxi-v3", render_mode="rgb_array")

    # Initialize agents for both environments
    cartpole_state_size = cartpole_env.observation_space.shape[0]
    cartpole_action_size = cartpole_env.action_space.n
    cartpole_agent = DoubleDQNAgent(cartpole_state_size, cartpole_action_size)
    cartpole_model_path = os.path.join(model_dir, "cartpole_dqn.weights.h5")
    cartpole_agent.load(cartpole_model_path)

    taxi_state_size = taxi_env.observation_space.n
    taxi_action_size = taxi_env.action_space.n
    taxi_agent = DoubleDQNAgent(taxi_state_size, taxi_action_size)
    taxi_model_path = os.path.join(model_dir, "taxi_agent_model.pkl")

    # Load the Taxi model
    with open(taxi_model_path, 'rb') as f:
        taxi_agent.q_table = pickle.load(f)

    environment_descriptions = {
        "Taxi": (
            "In the Taxi environment, navigate a taxi to pick up passengers "
            "and drop them off at designated locations. Earn positive rewards for successful "
            "pickups and drop-offs while avoiding negative rewards for incorrect actions."
        ),
        "CartPole": (
            "In the CartPole environment, balance a pole on a moving cart by applying forces "
            "to the left or right. The goal is to keep the pole upright for as long as possible "
            "without it falling over."
        )
    }

    def get_action(obs, mode, env_name):
        if mode == "Random Key Press":
            if env_name == "Taxi":
                return random.randint(0, 5)  # 0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff
            else:
                return random.choice([0, 1])  # 0: Left, 1: Right for CartPole
        else:  # Improved Model
            if env_name == "Taxi":
                return np.argmax(taxi_agent.q_table[obs])
            else:
                return cartpole_agent.act(np.array([obs]))

    # Layout with spacing adjustments
    env_name = st.selectbox("Select environment", ["Taxi", "CartPole"])
    
    # Add spacing with markdown or empty lines before description
    st.markdown("<br>", unsafe_allow_html=True)  
    description_placeholder = st.empty()
    description_placeholder.write(environment_descriptions[env_name])

    st.markdown("<br>", unsafe_allow_html=True)  
    mode = st.selectbox("Select gameplay mode", ["Random Key Press", "Improved Model"])

    if st.button("Start Game"):
        st.write("Starting game...")
        env = taxi_env if env_name == "Taxi" else cartpole_env
        obs = env.reset()[0]
        done = False
        truncated = False
        frame_placeholder = st.empty()
        score = 0

        st.sidebar.title("Action History")
        action_history_placeholder = st.sidebar.empty()
        action_history = []
        actions_taken = []

        pole_angles = []
        cart_positions = []

        while not (done or truncated) and (env_name != "Taxi" or score > -25):
            action = get_action(obs, mode, env_name)
            obs, reward, done, truncated, info = env.step(action)
            score += reward

            frame = env.render()
            image = Image.fromarray(frame)

            frame_placeholder.image(image, caption=f"{env_name} - {mode} Gameplay - Score: {score}", use_column_width=True)

            if env_name == "Taxi":
                action_name = ["South", "North", "East", "West", "Pickup", "Dropoff"][action]
                time.sleep(1)  # 1 second delay for Taxi environment
            else:
                action_name = "Left" if action == 0 else "Right"
                pole_angles.append(obs[2])
                cart_positions.append(obs[0])
                time.sleep(0.05)  # Keep the original delay for CartPole
            
            action_history.append(action_name)
            actions_taken.append(action)
            if len(action_history) > 15:
                action_history.pop(0)
            action_history_placeholder.text("\n".join(action_history))

        if env_name == "Taxi" and score <= -25:
            st.write("Game ended due to score reaching -25.")
        else:
            st.write(f"Game finished. Final score: {score}")
        env.close()

        # Action Distribution Bar Chart
        st.subheader("Action Distribution")
        action_counts = np.bincount(actions_taken, minlength=taxi_action_size if env_name == "Taxi" else 2)
        fig, ax = plt.subplots()
        
        if env_name == "Taxi":
            ax.bar(range(taxi_action_size), action_counts)
            ax.set_xticks(range(taxi_action_size))
            ax.set_xticklabels(['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'])
        else:
            ax.bar(['Left', 'Right'], action_counts)

        ax.set_ylabel('Count')
        ax.set_title('Action Distribution')
        st.pyplot(fig)

        # Performance Metrics
        st.subheader("Performance Metrics")
        st.write(f"Score: {score}")
        st.write(f"Episode Length: {len(actions_taken)}")

        if env_name == "CartPole":
            st.write(f"Average Action Value: {np.mean(actions_taken):.2f}")

            # Pole Angle Visualization
            st.subheader("Pole Angle Over Time")
            fig, ax = plt.subplots()
            ax.plot(range(len(pole_angles)), pole_angles)
            ax.set_xlabel('Step')
            ax.set_ylabel('Pole Angle (radians)')
            ax.set_title('Pole Angle Over Time')
            st.pyplot(fig)

            # Cart Position Visualization
            st.subheader("Cart Position Over Time")
            fig, ax = plt.subplots()
            ax.plot(range(len(cart_positions)), cart_positions)
            ax.set_xlabel('Step')
            ax.set_ylabel('Cart Position')
            ax.set_title('Cart Position Over Time')
            st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Error details:")
    st.error(sys.exc_info())
