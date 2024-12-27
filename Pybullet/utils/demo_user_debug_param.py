# 可以单独运行这个 python 文件
# demo for 滑块控件的使用方法，网上找的例子
import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# 启动仿真引擎的GUI
p.connect(p.GUI)

# 设置重力加速度
p.setGravity(0, 0, -9.81)

# 设置视角
p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=120, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

# load models
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
tableId = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])

# load other models
p.setAdditionalSearchPath("../models")
robotId = p.loadURDF("panda/panda.urdf", useFixedBase=True)  # basePosition=[0.0,0,0.62]

initial_position = [0, 0, 0.62]  # 设置初始位置为 [0, 0, 0.62] 米
initial_orientation = p.getQuaternionFromEuler([0, 0, 0])  # 设置初始方向为欧拉角 [0, 0, 0] 对应的四元数
p.resetBasePositionAndOrientation(robotId, initial_position, initial_orientation)

# 获取机械臂末端执行器的索引
endEffectorIndex = 6

# 获取机械臂的关节数量
numJoints = p.getNumJoints(robotId)
print("关节数量:" + str(numJoints))

# 打印每个关节的信息
for joint_index in range(numJoints):
    joint_info = p.getJointInfo(robotId, joint_index)
    print(f"Joint {joint_index}: {joint_info}")

# 机械臂的初始位置
restingPosition = [0, 3.14, -1.57, 1.57, 1.57, 1.57, -1.57, 0]
for jointNumber in range(numJoints):
    p.resetJointState(robotId, jointNumber, restingPosition[jointNumber])

# 设置圆形的参数
circle_radius = 3
circle_center = [0, 0]
numPoints = 50
angles = [2 * math.pi * float(i) / numPoints for i in range(numPoints)]

# 调参控件
target_x = p.addUserDebugParameter("Target X", -10, 10, 0)
target_y = p.addUserDebugParameter("Target Y", -10, 10, 0)
target_z = p.addUserDebugParameter("Target Z", -10, 10, 0)

# 开始按钮
button_start = p.addUserDebugParameter("Start", 1, 0, 1)
# 初始状态变量
button_state_prev = 0

# 画个球
sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
sphere_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,
                              basePosition=[0, 0, 1])


# 多组解
def calculate_ik_multiple_solutions(robot_id, end_effector_index, target_position, target_orientation, num_solutions=5):
    solutions = []
    for i in range(num_solutions):
        # 生成一个随机的初始关节状态
        random_joint_positions = [np.random.uniform(-np.pi, np.pi) for _ in range(numJoints)]
        # 计算逆运动学
        ik_solution = p.calculateInverseKinematics(robot_id, end_effector_index, target_position, target_orientation,
                                                   jointDamping=[0.01] * numJoints, lowerLimits=[-np.pi] * numJoints,
                                                   upperLimits=[np.pi] * numJoints, jointRanges=[2 * np.pi] * numJoints,
                                                   restPoses=random_joint_positions)
        solutions.append(ik_solution)
    return solutions


try:
    while True:
        x = p.readUserDebugParameter(target_x)
        y = p.readUserDebugParameter(target_y)
        z = p.readUserDebugParameter(target_z)

        # 更新球的位置
        p.resetBasePositionAndOrientation(sphere_id, [x, y, z], [0, 0, 0, 1])

        # 判断按钮状态
        button_state = p.readUserDebugParameter(button_start)
        if button_state != button_state_prev:
            print("Button pressed")
            # 使用逆向运动学计算关节角度
            jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, [x, y, z], [0, 0, 0, 1])

            # 移动机械臂
            p.setJointMotorControl2(bodyIndex=robotId, jointIndex=1, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[0], targetVelocity=0, force=500, positionGain=0.03,
                                    velocityGain=1)
            p.setJointMotorControl2(bodyIndex=robotId, jointIndex=2, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[1], targetVelocity=0, force=500, positionGain=0.03,
                                    velocityGain=1)
            p.setJointMotorControl2(bodyIndex=robotId, jointIndex=3, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[2], targetVelocity=0, force=500, positionGain=0.03,
                                    velocityGain=1)
            p.setJointMotorControl2(bodyIndex=robotId, jointIndex=4, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[3], targetVelocity=0, force=500, positionGain=0.03,
                                    velocityGain=1)
            p.setJointMotorControl2(bodyIndex=robotId, jointIndex=5, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[4], targetVelocity=0, force=500, positionGain=0.03,
                                    velocityGain=1)
            p.setJointMotorControl2(bodyIndex=robotId, jointIndex=6, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[5], targetVelocity=0, force=500, positionGain=0.03,
                                    velocityGain=1)
        button_state_prev = button_state

        p.stepSimulation()
        time.sleep(0.01)

except KeyboardInterrupt:
    # 用户中断程序时，退出循环
    print("Circle drawing interrupted by user.")
# 断开与仿真引擎的连接
p.disconnect()
