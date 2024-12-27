import pybullet as p
import numpy as np
import time
import pybullet_data
import copy

import argparse

'''
RobotArm类, 用于Pybullet Simulator的仿真，放置于urdf定义的同级文件夹下。
init初始化机器人相关函数功能；
step推进仿真；
reset重置仿真器至初始状态；
'''
class RobotArm():
    def __init__(self, args, stepsize=1e-3, realtime=0) -> object:
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.p = p

        self.robot_name = args.Robot
        self.control_freq = args.control_frequency
        self.sample_freq = args.sample_frequency
        self.control_mode = "torque"

        # connect pybullet
        p.connect(p.GUI) # p.DIRECT, p.GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime) # 0: disable real-time simulation, 1: enable real-time simulation 启用实时仿真以允许拖动
        p.setGravity(0, 0, -9.81) # 设置重力

        # 位置控制参数
        if self.robot_name == 'Panda':
            self.position_control_gain_p = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            self.position_control_gain_d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.max_torque = [100, 100, 100, 100, 100, 100, 100]
        elif self.robot_name == 'Z1':
            self.position_control_gain_p = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            self.position_control_gain_d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.max_torque = [30, 60, 30, 30, 30, 30]
        
        # load models
        p.setAdditionalSearchPath("./Pybullet/models")
        self.plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=.95)
        if self.robot_name == 'Panda':
            self.robot = p.loadURDF("panda/panda.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        elif self.robot_name == 'Z1':
            self.robot = p.loadURDF("z1_description/z1.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        else:
            raise ValueError('Invalid robot name.')

        
        if args.ObstacleCost and self.robot_name == 'Panda':
            self.obstacle = []
            bp = [0.25, 0.0, 0.9]
            obs = p.loadURDF("obstacle/sphere.urdf", basePosition=bp, globalScaling=1.0)
            self.obstacle.append(copy.copy(bp + [0.08]))

            bp = [0.5, 0.3, 0.4]
            obs = p.loadURDF("obstacle/sphere.urdf", basePosition=bp, globalScaling=3.0)
            self.obstacle.append(copy.copy(bp + [0.24]))

            # bp = [0.2, -0.3, 0.2]
            # obs = p.loadURDF("obstacle/sphere.urdf", basePosition=bp, globalScaling=3.0)
            # self.obstacle.append(copy.copy(bp + [0.24]))
            
            # related obstacle
            bp = [0.18, 0.38, 0.76]
            obs = p.loadURDF("obstacle/sphere.urdf", basePosition=bp, globalScaling=1.0)
            self.obstacle.append(copy.copy(bp + [0.08]))

            # bp = [0.0, -0.3, 0.9]
            # obs = p.loadURDF("obstacle/sphere.urdf", basePosition=bp, globalScaling=2.0)
            # self.obstacle.append(copy.copy(bp + [0.16]))
        elif args.ObstacleCost and self.robot_name == 'Z1':
            self.obstacle = []

            # bp = [0.2, 0.15, 0.3]
            # obs = p.loadURDF("obstacle/sphere.urdf", basePosition=bp, globalScaling=1)
            # self.obstacle.append(copy.copy(bp + [0.08]))

        # example 2：机械臂和小方块、小圆球在桌上

        # self.robot = p.loadURDF("panda/panda.urdf", basePosition=[0.00000, -0.200000, 1.200000], baseOrientation=[0.000000, 0.000000,
        #                      0.000000, 1.000000], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        # self.ball_1 = p.loadURDF("ball/ball.urdf", basePosition=[0.9, -0.2, 1.3], globalScaling=1)

        # # load other models
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.table_1 = p.loadURDF("table/table.urdf", basePosition=[0.5, -0.2, 0.0],
        #                           baseOrientation=[0, 0, 1, 0], globalScaling=2)
        # self.cube_1 = p.loadURDF("cube.urdf", basePosition=[0.7, -0.2, 1.3], globalScaling=0.10)

        '''
        robot自由度，panda是7个自由度，Z1是6个
        IMPORTANT! panda关节是joint1~7旋转，第八个joint固定；Z1是Joint0固定，Joint1~6旋转。
        '''
        self.dof = p.getNumJoints(self.robot) - 1

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_vel = []
        self.target_torque = []

        for j in range(self.dof):
            joint_info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_pos.append((self.q_min[j] + self.q_max[j]) / 2.0)
            self.target_torque.append(0.)
        # 调整初始摄像头视角
        # camera parameters：摄像头到目标位置的距离，摄像头的偏航角（水平旋转角度），摄像头的仰角（垂直旋转角度），摄像头的目标位置，即摄像头对准的点的坐标
        if self.robot_name == 'Panda':
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=120, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.4])
        elif self.robot_name == 'Z1':
            p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.2])
        
        # logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "./video.mp4")
        # p.stopStateLogging(logId)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.control_mode = "torque"
        if self.robot_name == 'Panda':
            # self.target_pos = [0., -0.7, 0.0, -1.6, 0., 3.5, 0.7]  # 末端处于高位
            self.target_pos = [0., 0., 0., -1.6, 0., 1.87, 0.]    # 末端处于低位
        elif self.robot_name == 'Z1':
            self.target_pos = [0.005228521470025748, 0.4504296325181785, -0.8985288816139108, 0.48647146872395064, 0.16775067443267064, 0.24798319934352273] # 末端处于中间

        for j in range(self.dof):
            self.target_torque[j] = 0.0
            p.resetJointState(self.robot, j, targetValue=self.target_pos[j])
        self.resetController()

    # robot functions
    def resetController(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(self.dof)])

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "velocity":
            self.control_mode = "velocity"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception('wrong control mode')

    def setTargetPositions(self, target_pos):
        self.target_pos = target_pos
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def setTargetVelocity(self, target_velocity):
        self.target_vel = target_velocity
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=self.target_vel)

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        # 同时控制多个关节的运动
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.target_torque)

    def getJointStates(self):   # 获取关节的位置和速度 返回值：关节位置，关节速度
        joint_states = p.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel

    def solveInverseDynamics(self, pos, vel, acc) -> object:
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori):
        return list(p.calculateInverseKinematics(self.robot, self.dof, pos, ori))

    def solveListKinematics(self, joints_array):
        joints_num = joints_array.shape[1] # joints_array should be N * dof
        if joints_num != self.dof:
            raise Exception('wrong joint trajectory: number of joints is not correct.')

        original_joint_positions, _ = self.getJointStates() # 保存原始关节角度

        end_effector_array = np.zeros(shape=(joints_array.shape[0], 7)) # 存储末段的位置和方向
        for k in range(joints_array.shape[0]):
            for j in range(joints_array.shape[1]):
                p.resetJointState(self.robot, j, joints_array[k, j])
            link_state = p.getLinkState(self.robot, self.dof)
            end_effector_array[k, 0: 3] = np.array(link_state[0])
            end_effector_array[k, 3: ] = np.array(link_state[1])

        # 恢复原始关节状态
        for i in range(len(original_joint_positions)):
            p.resetJointState(self.robot, i, original_joint_positions[i])

        return end_effector_array
    
    def GetAllLink(self, joints_array):
        original_joint_positions, _ = self.getJointStates() # 保存原始关节角度
        Link_pos_list = []
        for k in range(joints_array.shape[0]):
            Link_pos_list.append([])
            for j in range(joints_array.shape[1]):
                p.resetJointState(self.robot, j, joints_array[k, j])
                link_state = p.getLinkState(self.robot, j)
                Link_pos_list[k].append(np.array(link_state[0]))

        # 恢复原始关节状态
        for i in range(len(original_joint_positions)):
            p.resetJointState(self.robot, i, original_joint_positions[i])

        return Link_pos_list

    def traj_pos_control(self, joints_array):
        # 位置控制（STODI没用，参考力矩控制修改）
        iter_steps = 0
        num = 0
        while(num < joints_array.shape[0]):
            if iter_steps % 500 == 0:
                print("Simulation time: {:.3f}".format(self.t))
            
            if iter_steps % int(1.0 / (self.control_freq * self.stepsize)) == 0:
                self.setTargetPositions(joints_array[num, :])
                num += 1
            self.step()
            # time.sleep(self.stepsize)
            iter_steps += 1

    def traj_vel_control(self, joints_vel_array):
        # 速度控制（STODI没用，参考力矩控制修改）
        iter_steps = 0
        num = 0
        while(num < joints_vel_array.shape[0]):
            if iter_steps % 500 == 0:
                print("Simulation time: {:.3f}".format(self.t))

            if iter_steps % int(1.0 / (self.control_freq * self.stepsize)) == 0:
                self.setTargetVelocity(joints_vel_array[num, :])
                num += 1
            self.step()
            # time.sleep(self.stepsize)
            iter_steps += 1

    def traj_torque_control(self, pos_planned, vel_planned, acc_planned):
        # 计算轨迹起点，若此时不在起点，则控制机械臂先运动到起始位置
        self.setControlMode("position")
        # logId = self.p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "./video.mp4")
        Threshold = 0.01
        start_pos = pos_planned[0]
        cur_pos, _ = self.getJointStates()
        diff = np.array(cur_pos) - start_pos
        while(np.sum(diff) > Threshold):
            self.setTargetPositions(list(start_pos))
            self.step()
            cur_pos, _ = self.getJointStates()
            diff = np.array(cur_pos) - start_pos

        self.resetController()
        input("输入任意键开始轨迹模仿......")
        # 接下来进行torque控制
        num = 1
        self.setControlMode("torque")
        duration = (1.0 / self.control_freq) * len(pos_planned) + 2

        for i in range(int(duration / self.stepsize)):
            if i % 500 == 0:
                print("Simulation time: {:.3f}".format(self.t))

            if i % int(1.0 / (self.control_freq * self.stepsize)) == 0:
                num = min(num, len(pos_planned) - 1)
                pos_desired_update = pos_planned[num]
                vel_desired_update = vel_planned[num]
                acc_desired_update = acc_planned[num]
                num += 1

            if i % int(1.0 / (self.sample_freq * self.stepsize)) == 0:
                pos, vel = self.getJointStates()
            
            kv, kp = 40, 100
            
            acc_feedback = list((acc_desired_update - kv * (np.array(vel) - vel_desired_update) - kp * (np.array(pos) - pos_desired_update)))

            # solveInverseDynamics 的输入必须是 list 格式的
            joints_torque = self.solveInverseDynamics(list(pos), list(vel), acc_feedback)
            self.setTargetTorques(joints_torque)
            self.step()
            time.sleep(self.stepsize)
        # self.p.stopStateLogging(logId)
    
    def get_current_end_effector(self):
        end_effector_array = np.zeros(shape=(7,))
        link_state = p.getLinkState(self.robot, self.dof)
        end_effector_array[0: 3] = np.array(link_state[0])
        end_effector_array[3: ] = np.array(link_state[1])
        return end_effector_array

    def disconnet(self):
        p.disconnect()

    def demo_mode(self):  # 记录 demo 时，重力设置为0；启用实时仿真；设置关节阻尼
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(1)
        for j in range(self.dof):
            p.changeDynamics(self.robot, j, linearDamping=0.1, angularDamping=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Robot", type=str, choices=["Panda", "Z1"], default="Z1") # 选择机器人模型
    parser.add_argument("--control-frequency", type=float, default=20) # 假定的控制频率
    parser.add_argument("--sample-frequency", type=float, default=100) # 假定的采样频率

    args = parser.parse_args()

    robot = RobotArm(args)
    robot.setTargetPositions([0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    for i in range(100):
        robot.step()
    input()
    robot.setTargetPositions([0, 0, 0, -1.6, 0, 1.87])
    for i in range(5000):
        robot.step()
    input()

