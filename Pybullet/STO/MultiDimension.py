from .SingleDimension import single_dimension_stomp
from .SingleDimension import generate_state
import numpy as np

class Multi_dimensions_stomp():
    def __init__(self, num_points, args, cost_func, 
        rangelimit_low_list, rangelimit_high_list, 
        vellimit_low_list, vellimit_high_list, 
        acclimit_low_list, acclimit_high_list):
        # 初始化参数
        self.dimensions_num = args.dimension
        self.args = args
        self.num_points =  num_points
        
        # 初始化限位
        self.rangelimit_low_list = rangelimit_low_list
        self.rangelimit_high_list = rangelimit_high_list
        self.vellimit_low_list = vellimit_low_list
        self.vellimit_high_list = vellimit_high_list
        self.acclimit_low_list = acclimit_low_list
        self.acclimit_high_list = acclimit_high_list

        # 初始化各个关节
        self.single_dimension_list = []
        if len(self.rangelimit_high_list) != self.dimensions_num or len(self.rangelimit_low_list) != self.dimensions_num or\
            len(self.vellimit_low_list) != self.dimensions_num or len(self.vellimit_high_list) != self.dimensions_num or\
                len(self.acclimit_low_list) != self.dimensions_num or len(self.acclimit_high_list) != self.dimensions_num:
            raise Exception('wrong limit initialize: number of joints limits is not correct.')
        else:
            for k in range(self.dimensions_num):
                self.single_dimension_list.append(single_dimension_stomp(
                    num_points=self.num_points,
                    args=self.args,
                    rangelimit_low=self.rangelimit_low_list[k],
                    rangelimit_high=self.rangelimit_high_list[k],
                    vellimit_low=self.vellimit_low_list[k],
                    vellimit_high=self.vellimit_high_list[k],
                    acclimit_low=self.acclimit_low_list[k],
                    acclimit_high=self.acclimit_high_list[k]
                ))
        # 初始化计算cost的函数
        self.cost_function = cost_func

        # 记录 reuse state
        if self.args.reuse_state:
            self.reuse_trajectory = 0
            self.reuse_state = {"position": np.zeros((self.args.reuse_num, self.num_points, self.dimensions_num)),
                                "velocity": np.zeros((self.args.reuse_num, self.num_points, self.dimensions_num)), 
                                "acceleration": np.zeros((self.args.reuse_num, self.num_points, self.dimensions_num))}
            self.reuse_weights = np.zeros((self.args.reuse_num, self.num_points))
    
    def input_demonstration(self, joints_traj):
        # joints_traj should be N * dimension
        for i in range(self.dimensions_num):
            self.single_dimension_list[i].Update_state(joints_traj[:, i])
    
    def multi_limit_check(self, joints_traj):
        states_info = []
        flag = True
        for i in range(self.dimensions_num):
            states_info.append(generate_state(joints_traj[:, i], self.args))
            flag = self.single_dimension_list[i].limit_check(states_info[-1])
            if not flag:
                break
        return flag
    
    def multi_update(self, joints_traj):
        for i in range(self.dimensions_num):
            self.single_dimension_list[i].Update_state(joints_traj[:, i])

    def multiDimension_diffusion(self):
        for i in range(self.dimensions_num):
            self.single_dimension_list[i].diffusion()
    
    def calculate_weights(self):
        # 初始化
        weights = np.zeros((self.args.K, self.num_points)) # K * N
        state = {"position": np.zeros((self.num_points, self.dimensions_num)), # N * dimension
        "velocity": np.zeros((self.num_points, self.dimensions_num)),
        "acceleration": np.zeros((self.num_points, self.dimensions_num))}
        for i in range(self.args.K):
            for j in range(self.dimensions_num):
                state["position"][:, j] = self.single_dimension_list[j].diffusion_trajectory["position"][i,:]
                state["velocity"][:, j] = self.single_dimension_list[j].diffusion_trajectory["velocity"][i,:]
                state["acceleration"][:, j] = self.single_dimension_list[j].diffusion_trajectory["acceleration"][i,:]
            self.cost_function.Update_state(state)
            weights[i, :] = self.cost_function.end_effector_state_cost().squeeze() # N * 1
        
        # reuse trajectory
        if self.args.reuse_state:
            q_sum = np.sum(weights, axis=1) # K * 1
            q_max_ilist = np.argsort(q_sum)[::-1]
            # 把最大的几个换成reuse的
            for i in range(self.reuse_trajectory):
                k = q_max_ilist[i]
                weights[k,:] = self.reuse_weights[i, :]
                # 更新noise
                for j in range(self.dimensions_num):
                    self.single_dimension_list[j].diffusion_noisy[k,:] = self.reuse_state["position"][i,:,j] - self.single_dimension_list[j].state_record["position"]

        min_col = np.min(weights, axis=0) # N * 1
        max_col = np.max(weights, axis=0) # N * 1
        difference = max_col - min_col    # N * 1
        min_col = np.dot(np.ones(shape=(self.args.K, 1)), min_col.reshape(1,-1))
        # print(min_col.shape)
        difference = np.dot(np.ones(shape=(self.args.K, 1)), difference.reshape(1,-1))
        # print(difference.shape)
        weights = np.exp(-1 * (weights - min_col) / difference)
        # print(weights.shape)
        # print(weights)
        return weights # K * N
    
    def calculate_delta_noise(self, weights_p):
        proved_noise = np.zeros((self.num_points, self.dimensions_num)) # N * dimension
        for j in range(self.dimensions_num):
            proved_noise[:, j] = np.dot(self.single_dimension_list[j].M, 
            np.sum(weights_p * np.array(self.single_dimension_list[j].diffusion_noisy), axis=0))
            # proved_noise[:, j] = np.sum(weights_p * np.array(self.single_dimension_list[j].diffusion_noisy), axis=0)
        return proved_noise

    def update_reuse_traj(self, n_state):
        self.cost_function.Update_state(n_state)
        temp_cost = self.cost_function.end_effector_state_cost().squeeze() # N * 1
        cost = np.sum(temp_cost)
        if self.reuse_trajectory < self.args.reuse_num:
            self.reuse_weights[self.reuse_trajectory,:] = np.copy(temp_cost)
            self.reuse_state["position"][self.reuse_trajectory, :, :] = n_state["position"]
            self.reuse_state["velocity"][self.reuse_trajectory, :, :] = n_state["velocity"]
            self.reuse_state["acceleration"][self.reuse_trajectory, :, :] = n_state["acceleration"]
            self.reuse_trajectory += 1
        else:
            max_list = np.sum(self.reuse_weights, axis=1)
            index = np.argmax(max_list)
            if max_list[index] > cost:
                self.reuse_weights[index, :] = np.copy(temp_cost)
                self.reuse_state["position"][index, :, :] = n_state["position"]
                self.reuse_state["velocity"][index, :, :] = n_state["velocity"]
                self.reuse_state["acceleration"][index, :, :] = n_state["acceleration"]
                # print("Exchange Successfully!\n")
            else:
                pass
    
    def Reset(self):
        # 记录 reuse state
        if self.args.reuse_state:
            self.reuse_trajectory = 0
            self.reuse_state = {"position": np.zeros((self.args.reuse_num, self.num_points, self.dimensions_num)),
                                "velocity": np.zeros((self.args.reuse_num, self.num_points, self.dimensions_num)), 
                                "acceleration": np.zeros((self.args.reuse_num, self.num_points, self.dimensions_num))}
            self.reuse_weights = np.zeros((self.args.reuse_num, self.num_points))