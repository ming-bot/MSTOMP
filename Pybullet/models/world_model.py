import numpy as np

'''
World负责记录世界信息，如：障碍物信息，机器人当前状态距离最近障碍面的距离
'''
class World(object):
    def __init__(self, robot, obstacle:list):
        self.robot = robot # 机器人模型
        self.obstacles = obstacle # 包含障碍物的三维坐标，以及半径
        self.secure_space = 0.01 # Panda:3cm的缓冲区
        self.robot_radiu = 0.00 # 建模为机械臂关节之间为线段，Panda宽度为4cm

        print("Successfully initialize {} obstales.".format(len(self.obstacles)))

    def Update_obs(self, obs):
        self.obstacles = obs
    
    def calculate_EDT(self, joints_array):
        # joints_array是一系列的关节位置，希望返回的是一个shape = [joints_array[0]，1]的距离
        edt = np.zeros(shape=(joints_array.shape[0], 1))
        # 获取各个link的三维位置坐标
        links_pos = self.robot.GetAllLink(joints_array)
        for j, li in enumerate(links_pos):
            min_dis = np.inf
            for i in range(len(li) - 1):
                for obs in self.obstacles:
                    min_dis = min(min_dis, point_to_segment_distance(obs[:3], li[i], li[i+1]) - obs[3]) # 计算障碍物点到机械臂的最短距离
            # edt[j] = max(self.secure_space + self.robot_radiu - min_dis, 0)
            edt[j] = max(self.robot_radiu - min_dis, 0)
            if edt[j] > self.secure_space:
                print("Warning: The robot is too close to the obstacle.")
                edt[j] = edt[j] + 50 # 惩罚项
            else:
                edt[j] = 0
        return edt


def point_to_segment_distance(point, segment_start, segment_end):
    # 将点和线段端点转换为numpy数组
    P = np.array(point)
    A = np.array(segment_start)
    B = np.array(segment_end)
    
    # 计算向量
    AB = B - A
    AP = P - A
    
    # 计算投影系数
    AB_AB = np.dot(AB, AB)
    AP_AB = np.dot(AP, AB)
    if AB_AB == 0.0:
        AB_AB = 1e-10 # 避免除以零错误
    t = AP_AB / AB_AB
    
    # 确定最近点的位置
    if t < 0.0:
        closest_point = A
    elif t > 1.0:
        closest_point = B
    else:
        closest_point = A + t * AB
    
    # 计算点到最近点的距离
    distance = np.linalg.norm(P - closest_point)
    return distance