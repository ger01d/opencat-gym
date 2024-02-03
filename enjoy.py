import time
import pybullet as p
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
parallel_env = 1
env = make_vec_env(OpenCatGymEnv, n_envs=parallel_env)
model = PPO.load("trained/opencat_gym_release_local_2e6")

obs = env.reset()
sum_reward = 0

for i in range(500):    
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    sum_reward += reward
    env.render(mode="human")
    if done:
        print("Reward", sum_reward[0])
        sum_reward = 0
        obs = env.reset()
