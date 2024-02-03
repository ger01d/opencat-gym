# OpenCat Gym
A gym reinforcement learning environment for OpenCat robots based on Stable-Baselines3 and PyBullet.

## Simulation and Application
<img src=animations/trained_agent.gif width="500" /> <img src=animations/application.gif width="500" />

## Usage
Start training with 
``` 
python train.py 
```
To take a look at the pre-trained example, execute 
``` 
python enjoy.py
```

## Links
For more information on the reinforcement training implementation: https://stable-baselines3.readthedocs.io/en/master/index.html
And for the simulation environment please refer to: https://pybullet.org/wordpress/

## Related Work
The reward and penalty functions are based on: https://www.nature.com/articles/s41598-023-38259-7 \
Including a joint angle history was inspired by: https://www.science.org/doi/10.1126/scirobotics.aau5872
