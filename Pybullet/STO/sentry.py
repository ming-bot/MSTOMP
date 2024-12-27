# from main import generate_multi_state
import numpy as np
from copy import deepcopy
import copy

def generate_multi_state(n7_trajectory, args):
    velocity = np.zeros(shape=n7_trajectory.shape)
    acceleration = np.zeros(shape=n7_trajectory.shape)
    dt = 1.0 / args.sample_frequency
    
    for i in range(n7_trajectory.shape[0] - 1):
        acceleration[i, :] = (2.0 / dt ** 2) * (n7_trajectory[i + 1, :] - n7_trajectory[i, :] - velocity[i, :] * dt)
        velocity[i + 1, :] = velocity[i, :] + acceleration[i, :] * dt

    return {"position": copy.copy(n7_trajectory), "velocity": copy.copy(velocity), "acceleration": copy.copy(acceleration)}

class TrajNode():
    def __init__(self, traj, args) -> None:
        self.traj = traj
        self.state = generate_multi_state(self.traj, args)
        self.cost = np.inf
        self.args = args
    
    def __lt__(self, other):
        if self.cost < other.cost:
            return True
        else:
            return False
    
    def Update_Cost(self, cost):
        self.cost = cost

    def Update_traj(self, traj):
        self.traj = traj
        self.state = generate_multi_state(traj, self.args)


class Sentry():
    def __init__(self, init_traj, cost_fun, args):
        init_trajNode = TrajNode(init_traj, args)
        self.cost_func = cost_fun
        self.args = args

        self.cost_func.Update_state(init_trajNode.state)
        init_trajNode.Update_Cost(np.sum(self.cost_func.calculate_total_cost(args.ContourCost)[1:]))
        self.pioneer = {"best": deepcopy(init_trajNode), 
                        "voyager": deepcopy(init_trajNode),
                        "neighbour": deepcopy(init_trajNode)}
    
    def Update(self, str, new_traj = None):
        # 更新虽小cost的轨迹
        if str == 'best':
            if self.pioneer['best'] > self.pioneer['voyager'] or self.pioneer['best'] > self.pioneer['neighbour']:
                if self.pioneer['voyager'] < self.pioneer['neighbour']:
                    self.pioneer['best'] = deepcopy(self.pioneer['voyager'])
                else:
                    self.pioneer['best'] = deepcopy(self.pioneer['neighbour'])
        # 更新远航者和近邻者
        elif str == 'voyager' or str == 'neighbour':
            self.pioneer[str].Update_traj(new_traj)
            self.cost_func.Update_state(self.pioneer[str].state)
            self.pioneer[str].Update_Cost(np.sum(self.cost_func.calculate_total_cost(self.args.ContourCost)[1:]))
        elif str == 'clock':
            self.pioneer['neighbour'] = deepcopy(self.pioneer['best'])
    
    def New_turn(self, new_init):
        init_trajNode = TrajNode(new_init, self.args)
        self.cost_func.Update_state(init_trajNode.state)
        init_trajNode.Update_Cost(np.sum(self.cost_func.calculate_total_cost(self.args.ContourCost)[1:]))
        self.pioneer = {"best": deepcopy(init_trajNode), 
                        "voyager": deepcopy(init_trajNode),
                        "neighbour": deepcopy(init_trajNode)}