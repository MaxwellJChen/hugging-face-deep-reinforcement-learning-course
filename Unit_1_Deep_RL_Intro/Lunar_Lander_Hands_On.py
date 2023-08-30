import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

def test_single_env():
    env = gym.make("LunarLander-v2")

    observation, info = env.reset()
    print(info)

    for _ in range(20):
        action = env.action_space.sample()
        print(action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Environment is reset.")
            observation, info = env.reset()

    env.close()

env = make_vec_env("LunarLander-v2", n_envs = 16)

model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

model_name = "Lunar_Lander_Agent"

model.learn(total_timesteps=1_000_000)
model_name = "ppo-LunarLander-v2"
model.save(model_name)

eval_env = gym.make("LunarLander-v2", render_mode = "rgb_array")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes = 10)
print(mean_reward)
print(std_reward)