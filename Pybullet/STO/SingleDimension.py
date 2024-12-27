import numpy as np
import copy

def generate_state(trajectory, args):
    # Target: Generate state including "position", "velocity" and "acceleration" according to the position trajectory
    # Parameters: trajectory is a point list, the velocity list and the acceleration list are the same shape 
    # We assume the begin point's velocity is zero
    velocity = np.zeros(trajectory.shape)
    acceleration = np.zeros(trajectory.shape)
    dt = 1.0 / args.sample_frequency
    # one dimension
    if trajectory.ndim == 1:
        for i in range(trajectory.shape[0] - 1):
            acceleration[i] = (2.0 / dt ** 2) * (trajectory[i + 1] - trajectory[i] - velocity[i] * dt)
            velocity[i + 1] = velocity[i] + acceleration[i] * dt
        state = {"position": copy.copy(trajectory),
                "velocity": copy.copy(velocity),
                "acceleration": copy.copy(acceleration)}
    # multiple dimensions
    elif trajectory.ndim == 2:
        for i in range(trajectory.shape[0] - 1):
            acceleration[:, i] = (2.0 / dt ** 2) * (trajectory[:, i + 1] - trajectory[:, i] - velocity[:, i] * dt)
            velocity[:, i + 1] = velocity[:, i] + acceleration[:, i] * dt
        state = {"position": copy.copy(trajectory),
                "velocity": copy.copy(velocity),
                "acceleration": copy.copy(acceleration)}
    return state

class single_dimension_stomp():
    def __init__(self, num_points, args, rangelimit_low, rangelimit_high,
                vellimit_low=-1, vellimit_high=100, acclimit_low=-1, acclimit_high=100):
        # 初始化长度
        self.num_points = num_points
        self.duration = (num_points - 1) * 1.0 / args.sample_frequency
        self.args = args
        # 记录限位
        self.range_limit = []
        if rangelimit_low < rangelimit_high:
            self.range_limit.append(rangelimit_low)
            self.range_limit.append(rangelimit_high)
        else:
            self.range_limit.append(rangelimit_high)
            self.range_limit.append(rangelimit_low)
        
        self.vel_limit = []
        if vellimit_low < vellimit_high:
            self.vel_limit.append(vellimit_low)
            self.vel_limit.append(vellimit_high)
        else:
            self.vel_limit.append(vellimit_high)
            self.vel_limit.append(vellimit_low)
        
        self.acc_limit = []
        if acclimit_low < acclimit_high:
            self.acc_limit.append(acclimit_low)
            self.acc_limit.append(acclimit_high)
        else:
            self.acc_limit.append(acclimit_high)
            self.acc_limit.append(acclimit_low)
        
        # 记录 state, according to the num_points
        self.state_record = {"position": np.zeros(self.num_points), 
                            "velocity": np.zeros(self.num_points), 
                            "acceleration": np.zeros(self.num_points)}
        # according to the num_points
        self.diffusion_trajectory = np.zeros((1, 7))
        self.diffusion_noisy = np.zeros((1, 7))
        # 记录 key matrix, according to the num_points
        self.A, self.inv_R, self.M = self.calculate_inv_R_and_M()

    def calculate_inv_R_and_M(self):
        # calculate key matrix
        A = np.zeros((self.num_points, self.num_points))
        A[0, 0] = 1
        A[self.num_points - 1, self.num_points - 1] = 1
        for i in range(1, self.num_points - 1):
            A[i, i-1] = 1
            A[i, i] = -2
            A[i, i+1] = 1
        R = np.dot(A.T, A)

        inv_R = np.linalg.inv(R)
        # scale num can be changed
        scale_number = np.array((1.0 / (self.num_points)) / np.max(inv_R, axis=0))
        # scale_number = np.array((np.abs(self.state_record["position"][-1] - self.state_record["position"][0])\
        # / (self.num_points)) / np.max(inv_R, axis=0))
        M = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_points - 1):
            M[ : , i ] = scale_number[i] * inv_R[ : , i ]
        return A, inv_R, M
    
    def limit_check(self, state):
        # Check limit in positon, velocity, acceleration
        # position check
        max_position = np.max(state["position"])
        min_position = np.min(state["position"])
        if max_position > self.range_limit[1] or min_position < self.range_limit[0]:
            # print(max_position ,min_position)
            # print("Position exceed limit!\n")
            return False

        # velocity check
        max_velocity = np.max(state["velocity"])
        min_velocity = np.min(state["velocity"])
        if max_velocity > self.vel_limit[1] or min_velocity < self.vel_limit[0]:
            # print(max_velocity ,min_velocity)
            # print("Velocity exceed limit!\n")
            return False

        # acceleration check
        max_acceleration = np.max(state["acceleration"])
        min_acceleration = np.min(state["acceleration"])
        if max_acceleration > self.acc_limit[1] or min_acceleration < self.acc_limit[0]:
            # print(max_acceleration ,min_acceleration)
            # print("Acceleration exceed limit!\n")
            return False
        return True

    def Update_state(self, point_array):
        if np.array(point_array).shape[0] != self.num_points:
            self.num_points = np.array(point_array).shape[0]
            self.A, self.inv_R, self.M = self.calculate_inv_R_and_M()
        self.state_record = generate_state(np.array(point_array), self.args)
    
    def diffusion(self):
        # 生成K条噪声轨迹, 暂时最后的点还未加噪声
        eph = np.random.multivariate_normal(np.zeros((self.num_points)), 
            # 0.1 * np.abs(self.state_record["position"][-1] - self.state_record["position"][0]) * self.inv_R)
            self.inv_R * (1.0 / self.args.sample_frequency)**4, self.args.K)
        # print(eph)
        self.diffusion_trajectory = generate_state(np.tile(self.state_record["position"], (self.args.K, 1)) + eph, self.args)
        self.diffusion_noisy = eph

