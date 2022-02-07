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
BOUND_ANGLE = 40
STEP_ANGLE = 15 # Maximum angle delta per step

class OpenCatGymEnv(gym.Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, render=True):
        self.step_counter = 0
        self.state_robot_history = np.array([])
        self.jointAngles_history = np.array([])
        self.boundAngles = np.deg2rad(BOUND_ANGLE)

        if render:
            p.connect(p.GUI) #, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60") # uncommend to create a video
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])
        
        # The action space are the 8 joint angles        
        self.action_space = spaces.Box(np.array([-1]*8), np.array([1]*8))

        # The observation space are the torso roll, pitch and the joint angles and a history of the last 20 joint angles
        self.observation_space = spaces.Box(np.array([-1]*166), np.array([1]*166))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        lastPosition = p.getBasePositionAndOrientation(self.robotUid)[0]
        jointAngles = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]
        ds = np.deg2rad(STEP_ANGLE) # Maximum joint angle derivative (maximum change per step)
        jointAngles += action * ds  # Change per step including agent action
        
        # Apply joint boundaries       
        jointAngles[0] = np.clip(jointAngles[0], -self.boundAngles, self.boundAngles)       # shoulder_left
        jointAngles[1] = np.clip(jointAngles[1], 0, self.boundAngles)                       # elbow_left
        jointAngles[2] = np.clip(jointAngles[2], -self.boundAngles, self.boundAngles)       # shoulder_right
        jointAngles[3] = np.clip(jointAngles[3], 0, self.boundAngles)                       # elbow_right
        jointAngles[4] = np.clip(jointAngles[4], -self.boundAngles, 0)                      # hip_right
        jointAngles[5] = np.clip(jointAngles[5], -self.boundAngles, 0)                      # knee_right
        jointAngles[6] = np.clip(jointAngles[6], -self.boundAngles, 0)                      # hip_left
        jointAngles[7] = np.clip(jointAngles[7], -self.boundAngles, 0)                      # knee_left
        
        
        if(self.step_counter % 2 == 0): # Every 2nd iteration will be added to the joint history
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

        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel_norm.reshape(1,-1)[0]))
        
        # Reward is the advance in x-direction - deviation in the y-direction
        currentPosition = p.getBasePositionAndOrientation(self.robotUid)[0] # Position of torso-link
        reward = REWARD_WEIGHT_1 * ((currentPosition[0] - lastPosition[0]) - (abs(currentPosition[1]) - abs(lastPosition[1]))) * REWARD_FACTOR
        done = False
        
        # Stop criteria of current learning episode: Number of steps or robot fell
        self.step_counter += 1
        if (self.step_counter > MAX_EPISODE_LEN) or self.is_fallen():
            reward = 0
            done = True

        info = {}
        self.observation = np.hstack((self.state_robot, self.jointAngles_history/self.boundAngles))

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

        self.robotUid = p.loadURDF("models/nybble.urdf",startPos, startOrientation)
        
        self.jointIds = []
        paramIds = []

        for j in range(p.getNumJoints(self.robotUid)):
            info = p.getJointInfo(self.robotUid, j)
            #print(info)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)
                paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8")))
                
        resetPos = np.deg2rad(np.array([30, 30, 30, 30, -30, -30, -30, -30]))
        #resetPos = np.zeros(8)
    
        # Reset joint position to rest pose
        for i in range(len(paramIds)):
            p.resetJointState(self.robotUid,i, resetPos[i])


        # Read robot state (pitch, roll and their derivatives of the torso-link)
        state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)[1]
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel = normalize(state_robot_vel.reshape(-1,1))
        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel.reshape(1,-1)[0]))
        
        # Initialize robot state history with reset position
        state_joints = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]   
        self.jointAngles_history = np.tile(state_joints, 20)
        
        self.observation = np.concatenate((self.state_robot ,self.jointAngles_history/self.boundAngles))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # Re-activate rendering
        
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def is_fallen(self):
        """ Check if robot is fallen. It becomes "True", when pitch or roll is more than 0.9 rad.
        """

        position, orientation = p.getBasePositionAndOrientation(self.robotUid)
        orientation = p.getEulerFromQuaternion(orientation)
        is_fallen = np.fabs(orientation[0]) > 0.9 or np.fabs(orientation[1]) > 0.9
        
        return is_fallen