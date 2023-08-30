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

env = gym.make('Taxi-v3', render_mode = "rgb_array")

Qtable_taxi = np.zeros((env.observation_space.n, env.action_space.n))
n_training_episodes = 25000
learning_rate = 0.7

n_eval_episodes = 100

eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
 161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
 112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148]

env_id = "Taxi-v3"
max_steps = 99
gamma = 0.95

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.005

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(Qtable, state)
    else:
        action = env.action_space.sample()
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            new_state, reward, terminated, truncated, info = env.step(action)

            Qtable[state][action] = Qtable[state][action] + learning_rate*(reward + gamma*np.max(Qtable[new_state]) - Qtable[state][action])

            if terminated or truncated:
                break

            state = new_state
    return Qtable

# Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi)

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

# print(evaluate_agent(env, 99, n_eval_episodes, Qtable_taxi, eval_seed))

def record_video(env, Qtable, out_directory, fps=1):
    """
    Generate a replay video of the agent
    """

    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    steps = 0

    while not terminated or truncated:
        action = np.argmax(Qtable[state][:])
        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)
        steps += 1
        if steps > 60:
            break
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps = fps)

def push_to_hub(repo_id, model, env, video_fps=1, local_repo_path="hub"):
    _, repo_name = repo_id.split("/")

    eval_env = env
    api = HfApi()

    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    if env.spec.kwargs.get("map_name"):
        model["map_name"] = env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery", "") == False:
            model["slippery"] = False

    with open((repo_local_path) / "q-learning.pkl", "wb") as f:
        pickle.dump(model, f)

    mean_reward, std_reward = evaluate_agent(eval_env, model["max_steps"], model["n_eval_episodes"], model["qtable"],
                                             model["eval_seed"])

    evaluate_data = {
        "env_id": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }

    with open(repo_local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    env_name = model["env_id"]
    if env.spec.kwargs.get("map_name"):
        env_name += "-" + env.spec.kwargs.get("map_name")

    if env.spec.kwargs.get("is_slippery", "") == False:
        env_name += "-" + "no_slippery"

    metadata = {}
    metadata["tags"] = [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]

    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
    )

    model_card = f"""
    # **Q-Learning** Agent playing1 **{env_id}**
    This is a trained model of a **Q-Learning** agent playing **{env_id}** .

    ## Usage

    ```python

    model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

    # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
    env = gym.make(model["env_id"])"""

    evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])

    readme_path = repo_local_path / "README.md"
    readme = ""
    print(readme_path.exists())
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    metadata_save(readme_path, metadata)

    video_path = repo_local_path / "replay.mp4"
    record_video(env, model["qtable"], video_path, video_fps)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo="..",
    )

    print("Your model is pushed to the Hub. You can view your model here: ", repo_url)

# model = {
#     "env_id": env_id,
#     "max_steps": max_steps,
#     "n_training_episodes": n_training_episodes,
#     "n_eval_episodes": n_eval_episodes,
#     "eval_seed": eval_seed,
#
#     "learning_rate": learning_rate,
#     "gamma": gamma,
#
#     "max_epsilon": max_epsilon,
#     "min_epsilon": min_epsilon,
#     "decay_rate": decay_rate,
#
#     "qtable": Qtable_taxi
# }

# username = "MJC-1" # FILL THIS
# repo_name = "Q-learning-Taxi-v3" # FILL THIS
# push_to_hub(
#     repo_id=f"{username}/{repo_name}",
#     model=model,
#     env=env)