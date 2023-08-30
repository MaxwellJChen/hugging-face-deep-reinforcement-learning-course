import numpy as np
import gymnasium as gym
import random
import imageio
import tqdm

import pickle5 as pickle
from tqdm import tqdm

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json

from urllib.error import HTTPError
from huggingface_hub import hf_hub_download

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state][:])
    return action

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """Return average reward and std of reward"""

    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed = seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state

        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def load_from_hub(repo_id, filename):
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """
    # Get the model from the Hub, download and cache the model on your local disk
    pickle_model = hf_hub_download(
        repo_id=repo_id,
        filename=filename
    )

    with open(pickle_model, 'rb') as f:
        downloaded_model_file = pickle.load(f)

    return downloaded_model_file

# Taxi
model = load_from_hub(repo_id="ThomasSimonini/q-Taxi-v3", filename="q-learning.pkl") # Try to use another model

print(model)
env = gym.make(model["env_id"])

print(evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"]))

# Frozen Lake
model = load_from_hub(repo_id="ThomasSimonini/q-FrozenLake-v1-no-slippery", filename="q-learning.pkl") # Try to use another model

env = gym.make(model["env_id"], is_slippery=False)

print(evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"]))