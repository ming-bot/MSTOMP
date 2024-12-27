# 可以单独运行这个 python 文件
# demo for 记录机器人的轨迹数据
import sys
sys.path.append(r'C:/Users/hp/PycharmProjects/trajOptimize/franka-pybullet/src')

import pybullet as p
import pybullet_data
import time
import json  # 用于保存记录的数据

t = 0.0
stepsize = 1e-3
realtime = 0

# 连接到PyBullet物理仿真服务器
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# 调整初始摄像头视角
# camera parameters：摄像头到目标位置的距离，摄像头的偏航角（水平旋转角度），摄像头的仰角（垂直旋转角度），摄像头的目标位置，即摄像头对准的点的坐标
p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=30, cameraPitch=-30,
                             cameraTargetPosition=[0, 0, 0])

# load models
p.setAdditionalSearchPath("../models")      # 设置 PyBullet 的资源路径
plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
p.changeDynamics(plane, -1, restitution=.95)
robot = p.loadURDF("panda/panda.urdf", basePosition=[0.00000, -0.200000, 1.200000], baseOrientation=[0.000000, 0.000000,
                     0.000000, 1.000000], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
ball_1 = p.loadURDF("ball/ball.urdf", basePosition=[0.7, -0.2, 1.3], globalScaling=1)

# load other models
p.setAdditionalSearchPath(pybullet_data.getDataPath())
table_1 = p.loadURDF("table/table.urdf", basePosition=[0.5, -0.2, 0.0],
                          baseOrientation=[0, 0, 1, 0], globalScaling=2)
cube_1 = p.loadURDF("cube.urdf", basePosition=[0.7, -0.2, 1.3], globalScaling=0.10)

# robot parameters
dof = p.getNumJoints(robot) - 1
joint_indices = list(range(dof))    # 关节索引
initial_pos = [0, -0.7, 0, -1.6, 0, 3.5, 0.7]
for j in range(dof - 1):
    p.resetJointState(robot, j, targetValue=initial_pos[j])

# 初始化记录结构
trajectory_data = []

# 启用实时仿真以允许拖动
p.setRealTimeSimulation(1)

# 仿真循环
try:
    while True:
        # 获取所有关节的状态
        joint_states = p.getJointStates(robot, joint_indices)

        # 提取所有关节的位置和速度
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # 记录当前时间的关节状态
        trajectory_data.append({
            'time': time.time(),
            'positions': joint_positions,
            'velocities': joint_velocities
        })

        # 数据采样间隔 0.01 秒
        time.sleep(0.01)

except KeyboardInterrupt:
    # 保存记录的数据到文件
    with open(r'C:/Users/hp/PycharmProjects/trajOptimize/franka-pybullet/src/results/0530/aaa.json', 'w') as f:
        json.dump(trajectory_data, f, indent=4)
    print("Trajectory data has been saved to 'aaa.json'")

finally:
    # 断开连接
    p.disconnect()







