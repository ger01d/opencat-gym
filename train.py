import time
import pybullet as p

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
env = OpenCatGymEnv()
check_env(env)

# Set up number of parallel environments
parallel_env = 8
env = make_vec_env(lambda: env, n_envs=parallel_env) # only for PPO

model = PPO('MlpPolicy', env, verbose=1, n_steps=2048,  batch_size=64, learning_rate=2.5e-4, ent_coef=0,  tensorboard_log="./trained/tensorboard_logs/").learn(10.0e6) # usally 10e6
model.save("trained/ppo_reference_no_penalty")


# Load model to continue previous training
 #model = PPO.load("trained/ppo_reference_no_penalty", env, n_steps=int(2048),  batch_size=64, learning_rate=2.5e-4, ent_coef=0, tensorboard_log="./trained/tensorboard_logs/").learn(10.0e6)
#model.save("trained/ppo_reference_no_penalty")





