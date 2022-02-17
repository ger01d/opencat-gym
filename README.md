# Nybble Gym
A gym reinforcement learning environment for the Nybble OpenCat robot based on Stable-Baselines3 and PyBullet.

## Usage
You should use a virtual environment. (python3 -m venv .venv)  
pip install -r requirements.txt  
You may need to install torch with CUDA for GPU usage. https://pytorch.org/get-started/locally/  
Start training with train_nybble.py or use the Google Colab example train-colab.ipynb  
To take a look at the pre-trained example, execute enjoy_nybble.py.

## TODO
Implement https://arxiv.org/pdf/1812.11103.pdf.  
~~Improve 3D model.~~
Compare algorithms.  
## Links
For more information on the reinforcement training implementation: https://stable-baselines3.readthedocs.io/en/master/index.html  
And for the simulation environment please refer to: https://pybullet.org/wordpress/
