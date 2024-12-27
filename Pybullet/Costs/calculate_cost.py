import numpy as np
from .contour_cost import *
from models import World
import copy

def generate_cartesian_state(trajectory, args):
    # trajectory is point list, and we assume the begin point's velocity is zero
    velocity = np.zeros(trajectory.shape)
    dt = 1.0 / args.sample_frequency
    acceleration = np.zeros(trajectory.shape)

    for i in range(trajectory.shape[0] - 1):
        acceleration[i, :] = (2.0 / dt ** 2) * (trajectory[i + 1, :] - trajectory[i, :] - velocity[i, :] * dt)
        velocity[i + 1, :] = velocity[i, :] + acceleration[i, :] * dt
    cartesian_state = {"position": copy.copy(trajectory),
            "velocity": copy.copy(velocity),
            "acceleration": copy.copy(acceleration)}
    return cartesian_state

class Multi_Cost():
    def __init__(self, robot, init_trajectory, args):
        self.robot = robot
        self.args = args
        self.state = None
        self.end_effector_traj_list = None
        self.cartesian_effector_state = None
        self.init_cartesian_traj = self.robot.solveListKinematics(init_trajectory)[:, :3] # N * 3
        if args.ObstacleCost:
            self.world = World(robot, robot.obstacle)
    
    def Update_state(self, state):
        self.state = state # {"position": N * 7, "velocity": N * 7, "acceleration": N * 7}
        self.end_effector_traj_list = self.robot.solveListKinematics(self.state["position"]) # N * 7
        self.cartesian_effector_state = generate_cartesian_state(self.end_effector_traj_list, self.args) # {"position": N * 7, "velocity": N * 7, "acceleration": N * 7}
        # print(self.cartesian_effector_state)

    def calculate_control_cost(self):
        accelerate_state = self.state["acceleration"] # N * 7
        control_cost = 0.5 * (accelerate_state * accelerate_state) * (1.0 / self.args.sample_frequency)**2
        # control_cost = 0.5 * (accelerate_state * accelerate_state)
        # print(control_cost.shape)
        return control_cost

    def calculate_contour_cost(self, str):
        if str == 'KL':
            cost = calculate_kl_contour_cost(self.init_cartesian_traj, self.end_effector_traj_list[:, :3])
        elif str == 'NMSEPS':
            cost = calculate_nmse_contour_cost(self.init_cartesian_traj, self.end_effector_traj_list[:, :3])
        elif str == 'MSEPS':
            cost = calculate_mse_contour_cost(self.init_cartesian_traj, self.end_effector_traj_list[:, :3])
        elif str == 'DTW':
            cost = cal_dtw_similarity(self.init_cartesian_traj, self.end_effector_traj_list[:, :3])
        elif str == 'MSE':
            cost = cal_mse_euclidean_similarity(self.init_cartesian_traj, self.end_effector_traj_list[:, :3])
        # elif str == 'ex':
        #     cost = cal_ex_item(self.init_cartesian_traj, self.end_effector_traj_list[:, :3])
        elif str == 'MSES':
            cost = calculate_omse_contour_cost(self.init_cartesian_traj, self.end_effector_traj_list[:, :3])
        elif str == 'None':
            cost = 0
        else:
            raise("Wrong Cost function!")
        # print(cost)
        return cost

    def calculate_special_cost(self):
        # TODO
        cost = 0
        return cost

    # 其实这个函数取名有点问题，其实是用作在评分中的函数
    def end_effector_state_cost(self):
        # 暂定的q = q_o + q_s + q_v + q_e
        q = np.zeros(shape=(self.cartesian_effector_state["velocity"].shape[0], 1))
        if self.args.ObstacleCost:
            q_o = self.world.calculate_EDT(self.state['position']) # N * 7
            q = q + q_o
        # VELOCITY_BAR = 0.1
        # dt = 1.0 / self.args.sample_frequency
        # q_v = dt * (VELOCITY_BAR * np.ones(velocity.shape[0]) - velocity_vector_size)**2 \
        # - acceleration_vector_size * dt**2 * (VELOCITY_BAR * np.ones((velocity.shape[0])) - velocity_vector_size) \
        # + (1.0 / 3) * dt **3 * acceleration_vector_size * acceleration_vector_size

        if self.args.ContourCost:
            q_s = np.ones(q.shape) * self.calculate_contour_cost(self.args.ContourCost)/max(np.max(q),1.0) # N * 1
            # q_s = np.ones(q.shape) * self.calculate_contour_cost(self.args.ContourCost)
            q = q + q_s
        return q # N * 1


    def calculate_total_cost(self, str):
        control_cost = np.sum(self.calculate_control_cost())
        contour_cost = self.calculate_contour_cost(str)
        # if self.args.ContourCost:
        #     effector_state_cost = np.sum(self.end_effector_state_cost() - np.ones(shape=control_cost.shape) * contour_cost)
        # else:
        #     effector_state_cost = np.sum(self.end_effector_state_cost())
        effector_state_cost = np.sum(self.end_effector_state_cost())
        special_cost = self.calculate_special_cost()

        return np.array([control_cost, contour_cost, effector_state_cost, special_cost])
