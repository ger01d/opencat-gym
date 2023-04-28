import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data
import time
from threading import Timer


MAX_EPISODE_LEN = 500 # Number of steps for one training episode
REWARD_FACTOR = 1000
PENALTY_FACTOR = 1.0 
PENALTY_STEPS = 10e6 # increasing penalty with reference to PENALTY_STEPS
ANG_FACTOR = 0.1 # improve angular velocity resolution before saturation [-1,1] 
REWARD_WEIGHT_1 = 1.0
REWARD_WEIGHT_2 = 0.0
REWARD_WEIGHT_3 = 0.0
REWARD_WEIGHT_4 = 0.0
REWARD_WEIGHT_5 = 0.0
BOUND_ANGLE = 110 # degree
STEP_ANGLE = 11 # Maximum angle (deg) delta per step
PAW_Z_TARGET = 0.005 # m

# Values for randomization, to improve sim to real transfer
RANDOM_GYRO = 10 # Percent, not implemented yet
RANDOM_JOINT_ANG = 0 # Percent
RANDOM_MASS = 0 # Percent
RANDOM_FRICTION = 0 # Percent

LENGTH_JOINT_HISTORY = 30
SIZE_OBSERVATION = LENGTH_JOINT_HISTORY * 8 + 6


class OpenCatGymEnv(gym.Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        self.step_counter_session = 0
        self.state_robot_history = np.array([])
        self.jointAngles_history = np.array([])
        self.boundAngles = np.deg2rad(BOUND_ANGLE)
        self.randJointAngle = np.deg2rad(RANDOM_JOINT_ANG)

        #p.connect(p.DIRECT) # use for training without visualisation (significant faster training)
        p.connect(p.GUI)#, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60") # uncommend to create a video
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])    

        # The action space are the 8 joint angles        
        self.action_space = spaces.Box(np.array([-1]*8), np.array([1]*8))

        # The observation space are the torso roll, pitch and the angular velocities and a history of the last 20 joint angles
        self.observation_space = spaces.Box(np.array([-1]*SIZE_OBSERVATION), np.array([1]*SIZE_OBSERVATION))


    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        lastPosition = p.getBasePositionAndOrientation(self.robotUid)[0][0]
        jointAngles = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]
        ds = np.deg2rad(STEP_ANGLE) # Maximum joint angle derivative (maximum change per step)
        jointAngles += action * ds  # Change per step including agent action

        # Apply joint boundaries       
        jointAngles[0] = np.clip(jointAngles[0], -self.boundAngles, self.boundAngles)       # shoulder_left
        jointAngles[1] = np.clip(jointAngles[1], -self.boundAngles, self.boundAngles)       # elbow_left
        jointAngles[2] = np.clip(jointAngles[2], -self.boundAngles, self.boundAngles)       # shoulder_right
        jointAngles[3] = np.clip(jointAngles[3], -self.boundAngles, self.boundAngles)       # elbow_right
        jointAngles[4] = np.clip(jointAngles[4], -self.boundAngles, self.boundAngles)       # hip_right
        jointAngles[5] = np.clip(jointAngles[5], -self.boundAngles, self.boundAngles)       # knee_right
        jointAngles[6] = np.clip(jointAngles[6], -self.boundAngles, self.boundAngles)       # hip_left
        jointAngles[7] = np.clip(jointAngles[7], -self.boundAngles, self.boundAngles)       # knee_left
        
        
        # Transform angle to degree and perform rounding, because OpenCat robot have only integer values
        jointAnglesDeg = np.rad2deg(jointAngles.astype(np.float64))
        jointAnglesDegRounded = jointAnglesDeg.round()
        jointAngles = np.deg2rad(jointAnglesDegRounded)

        # Simulate delay for data transfer (delay has to be modeled to close "reality gap")
        p.stepSimulation()
        p.stepSimulation()

        
        # Check for friction of paws, to prevent slipping while training
        paw_contact = []
        paw_idx = [2, 5, 8, 11]
        for idx in paw_idx:
            paw_contact.append(True if p.getContactPoints(bodyA=self.robotUid, linkIndexA=idx) else False)

        paw_slipping = 0        
        for in_contact in np.nonzero(paw_contact)[0]:
            paw_slipping += np.linalg.norm((p.getLinkState(self.robotUid, linkIndex=paw_idx[in_contact], computeLinkVelocity=1)[0][0:1]))
          
        # Read clearance of paw from ground 
        paw_clearance = 0
        for idx in paw_idx:
            paw_z_pos = p.getLinkState(self.robotUid, linkIndex=idx)[0][2]
            paw_clearance += (paw_z_pos-PAW_Z_TARGET)**2 * np.linalg.norm((p.getLinkState(self.robotUid, linkIndex=idx, computeLinkVelocity=1)[0][0:1]))**0.5
        
        # Read clearance of torso from ground
        base_clearance = p.getBasePositionAndOrientation(self.robotUid)[0][2]    

        # Set new joint angles
        p.setJointMotorControlArray(self.robotUid, self.jointIds, p.POSITION_CONTROL, jointAngles, forces=np.ones(8)*0.2)
        p.stepSimulation() # delay of data transfer

        # Normalize jointAngles
        jointAngles[0] /= self.boundAngles
        jointAngles[1] /= self.boundAngles
        jointAngles[2] /= self.boundAngles
        jointAngles[3] /= self.boundAngles                   
        jointAngles[4] /= self.boundAngles
        jointAngles[5] /= self.boundAngles                   
        jointAngles[6] /= self.boundAngles          
        jointAngles[7] /= self.boundAngles                     
    
        # Adding angles of current step to joint angle history
        if(self.step_counter % 2 == 0): # Every 2nd iteration will be added to the joint history
            self.jointAngles_history = np.append(self.jointAngles_history, self.randomize(jointAngles, RANDOM_JOINT_ANG))
            self.jointAngles_history = np.delete(self.jointAngles_history, np.s_[0:8])
          
        # Data transfer delay
        p.stepSimulation() 
        p.stepSimulation()

        # Read robot state (pitch, roll and their derivatives of the torso-link)
        state_robot_pos, state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)
        p.stepSimulation()
        state_robot_ang_euler = np.asarray(p.getEulerFromQuaternion(state_robot_ang)[0:2])
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]*ANG_FACTOR
        state_robot_vel_clip = np.clip(state_robot_vel, -1, 1)
        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel_clip))
        
        # Reward and penalty functions
        zVelocity = p.getBaseVelocity(self.robotUid)[0][2]
        bodyStability = REWARD_WEIGHT_2 * (state_robot_vel_clip[0]**2 + state_robot_vel_clip[1]**2) + REWARD_WEIGHT_4 * zVelocity**2 
        currentPosition = p.getBasePositionAndOrientation(self.robotUid)[0][0] # Position in x-direction of torso-link
        info = {}
        reward = REWARD_WEIGHT_1 * (currentPosition - lastPosition) * REWARD_FACTOR - PENALTY_FACTOR * self.step_counter_session/PENALTY_STEPS * (bodyStability + REWARD_WEIGHT_5 * paw_clearance + REWARD_WEIGHT_3 * paw_slipping**2 )     
                
        done = False

        # Stop criteria of current learning episode: Number of steps or robot fell
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN:
            self.step_counter_session += self.step_counter
            info["TimeLimit.truncated"] = True # handle termination due to maximum number of steps correctly                  
            done = True
        elif self.is_fallen():
            self.step_counter_session += self.step_counter
            info["TimeLimit.truncated"] = False
            reward = 0      
            done = True
   
        self.observation = np.hstack((self.state_robot, self.jointAngles_history))

        return np.array(self.observation).astype(np.float32), reward, done, info
        
 

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # Disable rendering during loading
        p.setGravity(0,0,-9.81)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        planeUid = p.loadURDF("plane.urdf")

        startPos = [0,0,0.05]
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        self.robotUid = p.loadURDF("models/bittle.urdf",startPos, startOrientation, flags=p.URDF_USE_SELF_COLLISION) #
        p.changeDynamics(self.robotUid,-1, mass=self.randomize(0.179, RANDOM_MASS))
        self.jointIds = []
        paramIds = []

        for j in range(p.getNumJoints(self.robotUid)):
            info = p.getJointInfo(self.robotUid, j)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)
                paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8")))
                p.changeDynamics(self.robotUid, j, maxJointVelocity = np.pi*3) # limiting speed of motors

        jointAngles = np.deg2rad(np.array([0, 0, 0, 0, 0, 0, 0, 0]))

        i = 0
        for j in self.jointIds:
            p.resetJointState(self.robotUid,j, jointAngles[i])
            i = i+1

        # Normalize jointAngles
        jointAngles[0] /= self.boundAngles
        jointAngles[1] /= self.boundAngles
        jointAngles[2] /= self.boundAngles
        jointAngles[3] /= self.boundAngles                   
        jointAngles[4] /= self.boundAngles
        jointAngles[5] /= self.boundAngles                   
        jointAngles[6] /= self.boundAngles          
        jointAngles[7] /= self.boundAngles                       
 
        # Read robot state (pitch, roll and their derivatives of the torso-link)
        state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)[1]
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]*ANG_FACTOR
        self.state_robot = np.concatenate((state_robot_ang, np.clip(state_robot_vel, -1, 1)))   

        # Initialize robot state history with reset position
        state_joints = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]   
        self.jointAngles_history = np.tile(state_joints, LENGTH_JOINT_HISTORY)        
        self.observation = np.concatenate((self.state_robot ,self.jointAngles_history))
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
        is_fallen = np.fabs(orientation[0]) > 1.3 or np.fabs(orientation[1]) > 1.3 # Ref 0.9
        
        return is_fallen


    def randomize(self, value, percentage):
        """ Randomize value within percentage boundaries.
        """
        percentage /= 100
        value_randomized = value * (1 + percentage*(2*np.random.rand()-1))

        return value_randomized

    def set_false(self):
        self.delaying = False


            
    
