# 根据记录的 .json 文件，重新生成轨迹
import sys
sys.path.append(r'C:/Users/hp/PycharmProjects/trajOptimize/franka-pybullet/src')

import argparse
from sentry import Sentry, generate_multi_state
from robot_model import Panda
from generate_initial_traj import *
import json  # 用于数据记录


if __name__ == "__main__":
    robot = Panda()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str,
                        default=r'C:/Users/hp/PycharmProjects/trajOptimize/franka-pybullet/src')
    parser.add_argument("--expt-name", type=str, default=7) # required=True)

    parser.add_argument("--dimension", type=int, default=7)
    parser.add_argument("--add-contourCost", type=bool, default=True)
    parser.add_argument("--reuse-state", type=bool, default=True)
    parser.add_argument("--reuse-num", type=int, default=10)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.99)

    parser.add_argument("--control-frequency", type=float, default=10)
    parser.add_argument("--sample-frequency", type=float, default=20)

    args = parser.parse_args()
    print(robot)
    # # 生成初始轨迹
    # initial_trajectory = Joint_linear_initial(begin=[0, -0.7, 0, -1.6, 0, 3.5, 0.7], end=[0.17, 1.58, -0.52, -0.46, -0.25, 2.17, 0])
    #
    # initial_traj_state = generate_multi_state(initial_trajectory, args)
    # robot.traj_torque_control(initial_traj_state["position"], initial_traj_state["velocity"],
    #                           initial_traj_state["acceleration"])

    # 读取记录的轨迹
    with open(r'C:/Users/hp/PycharmProjects/trajOptimize/franka-pybullet/src/results/0530/aaa.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
    demo_trajectory = []

    for i in range(len(fcc_data)):
        demo_trajectory.append(fcc_data[i]['positions'])

    demo_trajectory = np.array(demo_trajectory)
    demo_traj_state = generate_multi_state(demo_trajectory, args)
    robot.traj_torque_control(demo_traj_state["position"], demo_traj_state["velocity"], demo_traj_state["acceleration"])




