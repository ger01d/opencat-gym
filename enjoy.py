import time
import pybullet as p

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
env = OpenCatGymEnv()

model = PPO.load("trained/ppo_reference_no_penalty")
obs = env.reset()

while True:    
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    env.render(mode="human")
    
    if done:
        obs = env.reset()
