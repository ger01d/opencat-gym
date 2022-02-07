from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from opencat_gym_env_nybble import OpenCatGymEnv

import torch

if __name__ == '__main__':
    # Training
    try:
        env = OpenCatGymEnv(render=False)
        env = make_vec_env(lambda: env, n_envs=1) # 25 for PPO
        env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True) # This might be necessary or break things

        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
        model.learn(total_timesteps=500000)
        model.save("sac_opencat")
    except KeyboardInterrupt:
        model.save("sac_opencat")

    # Load model to continue previous training
    #model = SAC.load("sac_opencat", env, verbose=1)
    #model.load_replay_buffer("sac_replay_buffer")
    #model.learn(500000) #tensorboard_log="./sac_opencat_tensorboard/").learn(10000000)