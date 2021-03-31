import time
import pybullet as p

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
env = OpenCatGymEnv()
check_env(env)

# Training
#env = make_vec_env(lambda: env, n_envs=25) # only for PPO
model = SAC('MlpPolicy', env, learning_starts = 20000, use_sde=False, use_sde_at_warmup=False, verbose=1).learn(1000000)

model.save("sac_opencat_desperate_long_learn")
model.save_replay_buffer("sac_replay_buffer_desperate_long_learn")

# Load model to continue previous training
#model = SAC.load("sac_opencat_desperate", env, verbose=1)
#model.load_replay_buffer("sac_replay_buffer_desperate")
#model.learn(500000) #tensorboard_log="./a2c_opencat_tensorboard/").learn(10000000)


