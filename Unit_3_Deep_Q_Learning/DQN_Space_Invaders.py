import gymnasium as gym
import stable_baselines3.common.atari_wrappers
from stable_baselines3 import DQN
import numpy as np

env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode = "rgb_array")
env = stable_baselines3.common.atari_wrappers.AtariWrapper(env, frame_skip = 1)
print(np.__version__)

state, info = env.reset()
print(state)