import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data
import time

from sklearn.preprocessing import normalize



MAX_EPISODE_LEN = 500  # Number of steps for one training episode
REWARD_FACTOR = 1000
REWARD_WEIGHT_1 = 1.0
REWARD_WEIGHT_2 = 1.0


class OpenCatGymEnv(gym.Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        self.state_robot_history = np.array([])
        self.jointAngles_history = np.array([])
        self.boundAngles = np.deg2rad(70)

        p.connect(p.GUI) #, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60") # uncommend to create a video
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])
        
        # The action space are the 8 joint angles        
        self.action_space = spaces.Box(np.array([-1]*8), np.array([1]*8))

        # The observation space are the torso roll, pitch and the joint angles incl. the last 5 steps
        self.observation_space = spaces.Box(np.array([-1]*70), np.array([1]*70))
 

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        lastPosition = p.getBasePositionAndOrientation(self.robotUid)[0][0]
        jointAngles = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]
        ds = np.deg2rad(10) # Maximum joint angle derivative (maximum change per step)
        jointAngles += action * ds  # Change per step including agent action
        
        # Apply joint boundaries       
        jointAngles = np.clip(jointAngles, -self.boundAngles, self.boundAngles)
        self.jointAngles_history = np.append(self.jointAngles_history, jointAngles)
        self.jointAngles_history = np.delete(self.jointAngles_history, np.s_[0:8])
          
        # Set new joint angles
        p.setJointMotorControlArray(self.robotUid, self.jointIds, p.POSITION_CONTROL, jointAngles)
        p.stepSimulation()
        
       # Read robot state (pitch, roll and their derivatives of the torso-link)
        state_robot_pos, state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)
        state_robot_ang_euler = np.asarray(p.getEulerFromQuaternion(state_robot_ang)[0:2])
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel_norm = normalize(state_robot_vel.reshape(-1,1))

        self.state_robot_history = np.append(self.state_robot_history, np.concatenate((state_robot_ang, state_robot_vel_norm.reshape(1,-1)[0])))
        self.state_robot_history = np.delete(self.state_robot_history, np.s_[0:6])
        
        # Reward is the advance in x-direction
        currentPosition = p.getBasePositionAndOrientation(self.robotUid)[0][0] # Position in x-direction of torso-link
        reward = REWARD_WEIGHT_1 * (currentPosition - lastPosition) * REWARD_FACTOR
        done = False
        
        # Stop criteria of current learning episode: Number of steps or robot fell
        self.step_counter += 1
        if (self.step_counter > MAX_EPISODE_LEN) or self.is_fallen():
            reward = 0
            done = True

        info = {}
        self.observation = np.hstack((self.state_robot_history, self.jointAngles_history/self.boundAngles))
        
        return np.array(self.observation).astype(np.float32), reward, done, info
        
 

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # Disable rendering during loading
        p.setGravity(0,0,-9.81)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        planeUid = p.loadURDF("plane.urdf")
        p.changeDynamics(planeUid, -1, lateralFriction = 1.4)
        
        startPos = [0,0,0.08]
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        self.robotUid = p.loadURDF("nybble.urdf",startPos, startOrientation)
        
        self.jointIds = []
        paramIds = []

        for j in range(p.getNumJoints(self.robotUid)):
            info = p.getJointInfo(self.robotUid, j)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)
                paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8")))
                
        #resetPos = np.deg2rad(np.array([30, 30, 30, 30, -30, -30, -30, -30]))
        resetPos = np.zeros(8)

        # Reset joint position to rest pose
        for i in range(len(paramIds)):
            p.resetJointState(self.robotUid,i, resetPos[i])




        # Read robot state (pitch, roll and their derivatives of the torso-link)
        state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)[1]
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel = normalize(state_robot_vel.reshape(-1,1))
        state_robot = np.concatenate((state_robot_ang, state_robot_vel.reshape(1,-1)[0]))
        
        # Initialize robot state history with reset position
        self.state_robot_history = np.tile(state_robot,5)   
        state_joints = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]   
        self.jointAngles_history = np.tile(state_joints, 5)
        
        self.observation = np.concatenate((self.state_robot_history ,self.jointAngles_history/self.boundAngles))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # Re-activate rendering
        
        return np.array(self.observation).astype(np.float32)


    def render(self, mode='human'):
        pass


    def close(self):
        p.disconnect()


    def is_fallen(self):
        """ Check if robot is fallen. It becomes "True", when pitch or roll is more than 0.3 rad or the z-position of the torso is below 0.05 m.
        """

        position, orientation = p.getBasePositionAndOrientation(self.robotUid)
        orientation = p.getEulerFromQuaternion(orientation)
        is_fallen = np.fabs(orientation[0]) > 0.9 or np.fabs(orientation[1]) > 0.9 #or np.fabs(position[2]) < 0.00
        
        return is_fallen
