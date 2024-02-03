from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class and check if structure is correct
#env = OpenCatGymEnv()
#check_env(env)

if __name__ == "__main__":
    # Set up number of parallel environments
    parallel_env = 8
    env = make_vec_env(OpenCatGymEnv, 
                       n_envs=parallel_env, 
                       vec_env_cls=SubprocVecEnv)

    # Change architecture of neural network to two hidden layers of size 256
    custom_arch = dict(net_arch=[256, 256])

    # Define PPO agent and train
    model = PPO('MlpPolicy', env, seed=42, 
                policy_kwargs=custom_arch, 
                n_steps=int(2048*8/parallel_env), 
                verbose=1).learn(2e6)

    model.save("trained/opencat_gym_esp32_trained_controller")

    # Load model to continue previous training
    #model = PPO.load("trained/opencat_gym_esp32_trained_controller", 
    #                   env, policy_kwargs=custom_policy_kwargs, 
    #                   n_steps=int(2048*8/parallel_env), verbose=1, 
    #                   tensorboard_log="trained/tensorboard_logs/").learn(2e6)
    #model.save("trained/opencat_gym_esp32_trained_controller_2")


