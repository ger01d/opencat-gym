import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
p.changeDynamics(planeId, -1, lateralFriction = 1.4)
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("CatModel.urdf",cubeStartPos, cubeStartOrientation) #, 
                   # useMaximalCoordinates=1, ## New feature in Pybullet 
                   #flags=p.URDF_USE_INERTIA_FROM_FILE)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
for i in range(p.getNumJoints(robotId)):
    info = p.getJointInfo(robotId, i)
    
    print(info)
    print("\n\n")

maxForce = 500
p.setJointMotorControl2(bodyUniqueId=robotId, 
                        jointIndex=1, 
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity = 1,
                        force = maxForce)


for i in range (10000):
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()

