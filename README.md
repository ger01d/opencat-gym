# OpenCat Gym
A gym reinforcement learning environment for OpenCat robots based on Stable-Baselines3 and PyBullet.

## Simulation and Application
<img src=animations/trained_agent_playground.gif width="400" /> <img src=animations/application.gif width="400" />

## Installation and Usage
Install python packages:
``` python
!pip install "stable-baselines3[extra]"
!pip install pybullet
```

Start training with 
``` 
python train.py 
```
To take a look at the pre-trained example, execute 
``` 
python enjoy.py
```
Alternatively you can open `opencat-gym.ipyn` in Jupyter notebook and perform training.

### Playing with training parameters
The training parameters are listed as constants in the `opencat-gym-env.py`. They change the weight of the reward-function.
``` python
PENALTY_STEPS = 2e6       # Increase of penalty by step_counter/PENALTY_STEPS
FAC_MOVEMENT = 1000.0     # Reward movement in x-direction
FAC_STABILITY = 0.1       # Punish body roll and pitch velocities
FAC_Z_VELOCITY = 0.0      # Punish z movement of body
FAC_SLIP = 0.0            # Punish slipping of paws
FAC_ARM_CONTACT = 0.01    # Punish crawling on arms and elbows
FAC_SMOOTH_1 = 1.0        # Punish jitter and vibrational movement, 1st order
FAC_SMOOTH_2 = 1.0        # Punish jitter and vibrational movement, 2nd order
FAC_CLEARANCE = 0.0       # Factor to enfore foot clearance to PAW_Z_TARGET
PAW_Z_TARGET = 0.005      # Target height (m) of paw during swing phase
```

## Links
For more information on the reinforcement training implementation: https://stable-baselines3.readthedocs.io/en/master/index.html
And for the simulation environment please refer to: https://pybullet.org/wordpress/ \
The API for creating the training environment: https://gymnasium.farama.org/

## Related Work
The reward and penalty functions are based on: https://www.nature.com/articles/s41598-023-38259-7 \
Including a joint angle history was inspired by: https://www.science.org/doi/10.1126/scirobotics.aau5872
