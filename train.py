import time
import pybullet as p

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
env = OpenCatGymEnv()
check_env(env)

# Training
#env = make_vec_env(lambda: env, n_envs=25) # only for PPO
model = SAC('MlpPolicy', env, learning_starts = 0, use_sde=False, use_sde_at_warmup=False, verbose=1).learn(5e5)

model.save("sac_opencat")

# Load model to continue previous training
#model = SAC.load("sac_opencat", env, verbose=1)
#model.load_replay_buffer("sac_replay_buffer")
#model.learn(500000) #tensorboard_log="./sac_opencat_tensorboard/").learn(10000000)


