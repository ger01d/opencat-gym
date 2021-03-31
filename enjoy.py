import time
import pybullet as p

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
env = OpenCatGymEnv()
check_env(env)

model = SAC.load("trained/sac_opencat_crawl")
obs = env.reset()

for _ in range(500):
    
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
	    obs = env.reset()
    time.sleep(1./240.)
