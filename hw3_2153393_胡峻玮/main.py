import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from tools import make_env, Plotter
from collections import deque
import time
import pickle
import os
import psutil
import shutil
from pathlib import Path


# Deep Q Network defined by the Deepmind paper
def create_q_model(n_actions, input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    return model


# Replay buffer to get agent remember what it did
class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.max_mem = 0
        self.input_shape = input_shape
        self.state_history = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_next_history = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.rewards_history = np.zeros(self.mem_size, dtype=np.float32)
        self.done_history = np.zeros(self.mem_size, dtype=np.float32)
        self.action_history = np.zeros(self.mem_size, dtype=np.int8)

    # Store information regarding the game into memory
    def store_transition(self, state, action, reward, state_next, done):
        index = self.mem_cntr % self.mem_size
        self.state_history[index] = state
        self.state_next_history[index] = state_next
        self.done_history[index] = done
        self.rewards_history[index] = reward
        self.action_history[index] = action
        self.mem_cntr += 1

    # Return sample buffer from memory
    def sample_buffer(self, batch_size):
        self.max_mem = min(self.mem_cntr, self.mem_size)
        batch_index = np.random.choice(self.max_mem, size=batch_size)

        state_sample = self.state_history[batch_index]
        state_next_sample = self.state_next_history[batch_index]
        rewards_sample = self.rewards_history[batch_index]
        action_sample = self.action_history[batch_index]
        done_sample = self.done_history[batch_index]

        return state_sample, action_sample, rewards_sample, state_next_sample, done_sample


class BreakoutAgent:
    def __init__(self, input_shape, n_actions, gamma, epsilon, epsilon_min, eps_dec,
                 max_memory_length, batch_size, update_network_action_interval, lr, save_delete_frame_interval, update_target_network_frame_interval):

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_dec = eps_dec
        self.max_memory_length = max_memory_length
        self.batch_size = batch_size
        self.update_network_action_interval = update_network_action_interval
        self.learning_rate = lr
        self.update_target_network_frame_interval = update_target_network_frame_interval
        self.save_delete_frame_interval = save_delete_frame_interval
        self.replay_memory = ReplayBuffer(max_size=self.max_memory_length, input_shape=self.input_shape)
        self.model = create_q_model(self.n_actions, self.input_shape)
        self.model_target = create_q_model(self.n_actions, self.input_shape)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()
        self.start_time = 0
        self.total_time = 0
        self.frame_count = 0
        self.clipped_reward = 0
        self.episode_reward_history = deque(maxlen=100)
        self.plotting_frame_count_history = []
        self.plotting_running_reward_history = []
        self.plotting_epsilon_value_history = []
        self.best_score = 0

    def choose_action(self, observation):
        # Take random action if epsilon is high enough
        if self.epsilon > np.random.random():
            action = np.random.choice(self.n_actions)
        # Predict best action
        else:
            state_array = np.expand_dims(observation, 0)
            action_probs = self.model(state_array, training=False)
            action = np.argmax(action_probs[0]).item()
        return action

    def update_network(self):
        if self.frame_count % self.update_target_network_frame_interval == 0:
            # Update target network with model weights
            self.model_target.set_weights(self.model.get_weights())
            # Print information about training
            print(self.information())

    def information(self):
        # Save necessary information to use in plotting
        self.plotting_frame_count_history.append(self.frame_count)
        self.plotting_running_reward_history.append(self.clipped_reward)
        self.plotting_epsilon_value_history.append(self.epsilon)

        # Calculate epoch time
        elapsed_time = time.time() - self.start_time
        self.total_time += elapsed_time / 60
        self.start_time = time.time()  # Record the start time for real-time measurement

        # Calculate memory usage
        process = psutil.Process()
        memory_info = process.memory_info().rss / 1024 / 1024  # in megabytes

        # New Best Score
        if self.clipped_reward > self.best_score:
            self.best_score = self.clipped_reward
            self.save_bestScore()
            # Log details
            template = "New Best Score!\nBest Score: {:.2f}, Score: {:.2f}, Frame Number {}, Epsilon {:.3f}, Time {:.2f}Sec, Total Time {:.2f}Min, Memory Usage {:.2f}MB"
            information = template.format(self.best_score, self.clipped_reward, self.frame_count, self.epsilon,
                                          elapsed_time, self.total_time, memory_info)
        # Not New Best Score
        else:
            # Log details
            template = "Best Score: {:.2f}, Score: {:.2f}, Frame Number {}, Epsilon {:.3f}, Time {:.2f}Sec, Total Time {:.2f}Min, Memory Usage {:.2f}MB"
            information = template.format(self.best_score, self.clipped_reward, self.frame_count, self.epsilon,
                                          elapsed_time, self.total_time, memory_info)

        return information

    # Training part
    def learn(self):
        if self.frame_count % self.update_network_action_interval == 0 and self.replay_memory.mem_cntr > self.batch_size:
            state_sample, action_sample, rewards_sample, state_next_sample, done_sample = self.replay_memory.sample_buffer(
                batch_size=self.batch_size)

            future_rewards = self.model_target.predict(state_next_sample, verbose=0)

            updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            masks = tf.one_hot(action_sample, self.n_actions)

            with tf.GradientTape() as tape:
                q_values = self.model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = self.loss_function(updated_q_values, q_action)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # Saves game data, if agent get best score after reaching 2_000_000 number of frames
    def save_bestScore(self):
        if self.frame_count > 2_000_000:
            model_name = 'breakout_model_final'
            base_directory = "DQN_bestScore"
            self.model.save(f"{base_directory}/{self.frame_count}/models/{model_name}")
            self.model_target.save(f"{base_directory}/{self.frame_count}/target_models/{model_name}")
            self.save_state(f"{base_directory}/{self.frame_count}/states/{model_name}.pkl")
            undesired_directory = f"{self.frame_count}"
            try:
                all_directories = next(os.walk(base_directory))[1]

                for directory in all_directories:
                    if directory != undesired_directory:
                        directory_to_delete = os.path.join(base_directory, directory)
                        shutil.rmtree(directory_to_delete)
            except Exception as e:
                print(f"Error: {e}")

    # Saves game data according to interval
    def save(self):
        if self.frame_count % self.save_delete_frame_interval == 0:
            model_name = 'breakout_model_final'
            self.model.save(f"DQN/{self.frame_count}/models/{model_name}")
            self.model_target.save(f"DQN/{self.frame_count}/target_models/{model_name}")
            self.save_state(f"DQN/{self.frame_count}/states/{model_name}.pkl")

    # Loads game data
    def load(self, path_m, path_tm, path_s):
        try:
            self.model = load_model(path_m)
            self.model_target = load_model(path_tm)
            self.load_state(path_s)
            self.start_time = time.time()
            print("Model, Target Model and States are loaded successfully.")
        except Exception as e:
            print("Error: Files couldn't loaded.")
            print("Model Path:", path_m)
            print("Target Model Path:", path_tm)
            print("States Path:", path_s)

    # Deletes old game data after reaching 100_000 number of frames according to interval
    def delete(self):
        if self.frame_count > 100_000 and self.frame_count % self.save_delete_frame_interval == 0:
            try:
                subdirectory_path = Path("DQN")
                all_folders = [folder for folder in subdirectory_path.iterdir() if folder.is_dir()]

                sorted_folders = sorted(all_folders, key=lambda folder: folder.stat().st_ctime, reverse=True)
                folders_to_delete = sorted_folders[2:]
                for folder in folders_to_delete:
                    shutil.rmtree(folder)
                    print(f"{folder} deleted successfully.")
            except Exception as e:
                print(f'Error: {e}')

    # Part of game data save
    def save_state(self, filename):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump({
                    'frame_count': self.frame_count,
                    'epsilon': self.epsilon,
                    'episode_reward_history': self.episode_reward_history,
                    'replay_memory': self.replay_memory,
                    'total_time': self.total_time,
                    'best_score': self.best_score,
                    'plotting_frame_count_history': self.plotting_frame_count_history,
                    'plotting_running_reward_history': self.plotting_running_reward_history,
                    'plotting_epsilon_value_history': self.plotting_epsilon_value_history}, f)
        except Exception as e:
            print(f"Error: {e}")

    # Part of game data load
    def load_state(self, filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.frame_count = data['frame_count']
                self.epsilon = data['epsilon']
                self.episode_reward_history = data['episode_reward_history']
                self.replay_memory = data['replay_memory']
                self.total_time = data['total_time']
                self.best_score = data['best_score']
                self.plotting_frame_count_history = data['plotting_frame_count_history']
                self.plotting_running_reward_history = data['plotting_running_reward_history']
                self.plotting_epsilon_value_history = data['plotting_epsilon_value_history']
        except Exception as e:
            print(f"Error: {e}")

    # Epsilon value selection
    def epsilon_func(self):
        self.epsilon = self.epsilon - self.eps_dec
        self.epsilon = max(self.epsilon, self.epsilon_min)


# In case of training, use this function
def train_model(env, agent):
    max_steps_per_episode = 10_000
    max_frame = 10_000_000
    isMaxFrame = True

    while isMaxFrame:
        if agent.frame_count == 0:
            agent.start_time = time.time()
        state = env.reset()
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            agent.frame_count += 1
            agent.epsilon_func()

            action = agent.choose_action(state)
            state_next, reward, done, info = env.step(action)
            episode_reward += reward
            agent.replay_memory.store_transition(state=state, action=action, reward=reward, state_next=state_next,
                                                 done=done)
            state = state_next

            agent.learn()
            agent.update_network()
            agent.save()
            agent.delete()

            if agent.frame_count % max_frame == 0:
                print(f"Reached {max_frame} Frames!")
                isMaxFrame = False

            if done:
                break

        agent.episode_reward_history.append(episode_reward)
        agent.clipped_reward = np.mean(agent.episode_reward_history)


# In case of model loading, use this function. Usage in training is not recommended.
def render_model(env, agent):
    TIME_FOR_NO_REWARD = 10
    WAITING_TIME_FOR_REWARD = 10
    while True:
        state = env.reset()
        last_reward_time = 0
        future = 0
        start = False
        remain_time_start = False

        while True:
            action = agent.choose_action(state)
            state_next, reward, done, info = env.step(action)
            state = state_next
            if reward != 0:
                if last_reward_time != 0 and start == False:
                    print("Got reward. Operation is cancelled.")
                last_reward_time = time.time()
                start = True
                remain_time_start = False

            if start and time.time() - last_reward_time > TIME_FOR_NO_REWARD:
                print("AI has not got any rewards for 10 seconds.")
                print("After 10 seconds of getting no reward, game will finish.")
                start = False
                now = time.time()
                future = now + WAITING_TIME_FOR_REWARD
                remain_time_start = True

            if remain_time_start and time.time() > future:
                print("Time is up. DONE")
                env.was_real_done = True
                done = True

            if done:
                break

        if env.was_real_done:
            break

    env.close()


# Plot the graph
def plotModel(agent):
    filename = f'breakout-v4-{agent.frame_count}.png'
    plot_graph = Plotter(agent.plotting_frame_count_history, agent.plotting_running_reward_history, agent.plotting_epsilon_value_history, filename, agent.best_score, agent.total_time)
    plot_graph.plot()
    print("Image Saved.")


# Load Game Data
# Make sure the model path exists and structure is correct
def loadModel(agent):
    frame_number = input("Enter frame number")
    model_name = 'breakout_model_final'
    path_m = f"DQN/{frame_number}/models/{model_name}"
    path_tm = f"DQN/{frame_number}/target_models/{model_name}"
    path_s = f"DQN/{frame_number}/states/{model_name}.pkl"
    agent.load(path_m, path_tm, path_s)


def atariBreakout(load, train, plot, render):
    if render:
        env = make_env("BreakoutNoFrameskip-v4", render_mode='human')
    else:
        env = make_env("BreakoutNoFrameskip-v4", render_mode=None)

    agent = BreakoutAgent(input_shape=(84, 84, 4), n_actions=4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, eps_dec=0.0000004,
                          max_memory_length=20_000, batch_size=32, update_network_action_interval=20,
                          lr=0.00025, save_delete_frame_interval=50_000, update_target_network_frame_interval=10_000)

    if load:
        loadModel(agent=agent)
        if render and not train:
            render_model(env=env, agent=agent)
    if train:
        train_model(env=env, agent=agent)

    if plot:
        plotModel(agent=agent)


if __name__ == "__main__":
    atariBreakout(train=False, load=True, plot=True, render=True)
