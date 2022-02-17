import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data

from sklearn.preprocessing import normalize


## Hyper Params
MAX_EPISODE_LEN = 500  # Number of steps for one training episode
REWARD_FACTOR = 1000
REWARD_WEIGHT_1 = 1.0
REWARD_WEIGHT_2 = 1.0
BOUND_ANGLE = 90
STEP_ANGLE = 45 # Maximum angle delta per step


class OpenCatGymEnv(gym.Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        # Store robot state and joint history
        self.state_robot_history = np.array([])
        self.jointAngles_history = np.array([])

        # Max joint angles
        self.boundAngles = np.deg2rad(BOUND_ANGLE)

        # Create the simulation, p.GUI for GUI, p.DIRECT for only training
        # Use options="--opengl2" if it decides to not work?
        p.connect(p.GUI)#, options="--opengl2") #, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60") # uncomment to create a video

        # Stop rendering
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        # Move camera
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=-10,
                                     cameraPitch=-40, 
                                     cameraTargetPosition=[0.4,0,0])
        
        # The action space contains the 11 joint angles
        self.action_space = spaces.Box(np.array([-1]*11), np.array([1]*11))

        # The observation space are the torso roll, pitch and the joint angles and a history of the last 20 joint angles
        # 11 * 20 + 6 = 226       
        self.observation_space = spaces.Box(np.array([-1]*226), np.array([1]*226))
 
    def add_to_revolute_joints(self, jointAngles, action):
        """"Adds the action vector to the revolute joints. Joint angles are
        clipped. `jointAngles` is changed to have the updated angles of the
        entire robot. The vector returned contains the only the revolute joints
        of the robot."""

        # Below is the mapping from the old 3D model to the new 3D model.
        # -----------------------------------------------------------
        # | PREVIOUS        |    NEW                   | NEW INDEX  |
        # |=========================================================|
        # | hip_right       |  R_Rear_Hip_Servo_Thigh  |  i = 1     |
        # | knee_right      |  R_Rear_Knee_Servo       |  i = 3     |
        # |---------------------------------------------------------|
        # | shoulder_right  |  R_Front_Hip_Servo_Thigh |  i = 7     |
        # | elbow_right     |  R_Front_Knee_Servo      |  i = 9     | 
        # |---------------------------------------------------------|
        # | #############   |  Body_Neck_Servo         |  i = 13    |
        # | #############   |  Neck_Head_Servo         |  i = 14    |
        # | --------------------------------------------------------|
        # | #############   |  Tail_Servo_Tail         |  i = 19    |
        # |---------------------------------------------------------|
        # | hip_left        |  L_Rear_Hip_Servo_Thigh  |  i = 21    |
        # | knee_left       |  L_Rear_Knee_Servo       |  i = 23    |
        # |---------------------------------------------------------|
        # | shoulder_left   |  L_Front_Hip_Servo_Thigh |  i = 27    |
        # | elbow_left      |  L_Front_Knee_Servo      |  i = 29    |
        # -----------------------------------------------------------

        # Define the maximum change in angle for each joint.
        diff_angle_max = np.deg2rad(STEP_ANGLE)
        jointAngles += action * diff_angle_max

        # Apply joint boundaries TODO: theses angles need clipping.
        jointAngles[0] = np.clip(jointAngles[0], -self.boundAngles, self.boundAngles) # R_Rear_Hip_Servo_Thigh
        jointAngles[1] = np.clip(jointAngles[1], -self.boundAngles, self.boundAngles) # R_Rear_Knee_Servo
        jointAngles[2] = np.clip(jointAngles[2], -self.boundAngles, self.boundAngles) # R_Front_Hip_Servo_Thigh
        jointAngles[3] = np.clip(jointAngles[3], -self.boundAngles, self.boundAngles) # R_Front_Knee_Servo
        jointAngles[4] = np.clip(jointAngles[4], -self.boundAngles, self.boundAngles) # Body_Neck_Servo
        jointAngles[5] = np.clip(jointAngles[5], -self.boundAngles, self.boundAngles) # Neck_Head_Servo
        jointAngles[6] = np.clip(jointAngles[6], -self.boundAngles, self.boundAngles) # Tail_Servo_Tail
        jointAngles[7] = np.clip(jointAngles[7], -self.boundAngles, self.boundAngles) # L_Rear_Hip_Servo_Thigh
        jointAngles[8] = np.clip(jointAngles[8], -self.boundAngles, self.boundAngles) # L_Rear_Knee_Servo
        jointAngles[9] = np.clip(jointAngles[0], -self.boundAngles, self.boundAngles) # L_Front_Hip_Servo_Thigh
        jointAngles[10] = np.clip(jointAngles[10], -self.boundAngles, self.boundAngles) # L_Front_Knee_Servo


    def step(self, action):
        # `action` is a vector of values -1 <= x <= 1 .

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        # Get position and vector of all joint angles of robot from simulation.
        lastPosition = p.getBasePositionAndOrientation(self.robotUid)[0][0]
        jointAngles = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]
        
        # Change per step including agent action
        self.add_to_revolute_joints(jointAngles, action)
        
        # Every 2nd iteration will be added to the joint history
        if(self.step_counter % 2 == 0): 
            self.jointAngles_history = np.append(self.jointAngles_history, jointAngles)
            self.jointAngles_history = np.delete(self.jointAngles_history, np.s_[0:11])

        
        # Set new joint angles
        p.setJointMotorControlArray(self.robotUid, self.jointIds, p.POSITION_CONTROL, jointAngles)
        p.stepSimulation()
        
        # Read robot state (pitch, roll and their derivatives of the torso-link)
        # Get pitch and roll of torso
        state_robot_pos, state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)
        state_robot_ang_euler = np.asarray(p.getEulerFromQuaternion(state_robot_ang)[0:2])
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel_norm = normalize(state_robot_vel.reshape(-1,1))

        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel_norm.reshape(1,-1)[0]))
        
        # Reward is the advance in x-direction
        currentPosition = p.getBasePositionAndOrientation(self.robotUid)[0][0] # Position in x-direction of torso-link
        reward = REWARD_WEIGHT_1 * (currentPosition - lastPosition) * REWARD_FACTOR
        done = False
        
        # Stop criteria of current learning episode: Number of steps or robot fell
        self.step_counter += 1
        if (self.step_counter > MAX_EPISODE_LEN) or self.is_fallen():
            reward = 0
            done = True

        # No debug info
        info = {}
        
        self.observation = np.hstack((self.state_robot, 
                                      self.jointAngles_history / self.boundAngles))

        return np.array(self.observation).astype(np.float32), reward, done, info
        
    def reset(self):     
        """Reset the simulation to its original state."""
        
        self.step_counter = 0
        p.resetSimulation()

        # Disable rendering during loading
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) 
        p.setGravity(0,0,-9.81)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 

        planeUid = p.loadURDF("plane.urdf")
        friction = np.random.uniform(0.5,0.9,1)[0]
        p.changeDynamics(planeUid, -1, lateralFriction = friction)
        
        startPos = [0,0,0.03]
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        self.robotUid = p.loadURDF("models/CatModel.urdf",
                                   startPos,
                                   startOrientation, 
                                   flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        # Get joint IDs of robot.
        self.jointIds = []
        for j in range(p.getNumJoints(self.robotUid)):

            jointType = p.getJointInfo(self.robotUid, j)[2]
            if (jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)
                
        # Reset joint position to rest pose
        resetPos = np.array([np.pi / 4, 0, -np.pi / 4, 0, 0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0])
        for i, j in enumerate(self.jointIds):
            p.resetJointState(self.robotUid, j, resetPos[i])


        # Get pitch and roll of torso
        state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)[1]

        # Get angular velocity of robot torso and normalise
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel = normalize(state_robot_vel.reshape(-1,1))

        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel.reshape(1,-1)[0]))
        
        # Initialize robot state history with reset position
        state_joints = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]   
        self.jointAngles_history = np.tile(state_joints, 20)
        
        self.observation = np.concatenate((self.state_robot, self.jointAngles_history / self.boundAngles))

        # Re-activate rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) 
        
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
