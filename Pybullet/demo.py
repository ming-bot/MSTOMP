import argparse
import os
import os.path as osp
import numpy as np
import copy
import time
import json
from tqdm import *

from models import RobotArm
from Costs import Multi_Cost
from STO import *
from Visualization import *

POSTURE = {'Panda': {'front': [-0.4873441065691155, 0.5814711977864513, 0.9234217980183635, -1.2212066335195748, -0.3227819795704253, 2.247907005921566, 0.18368041341779082],
           'back':[-0.43700637536473935, -1.4892142186045723, 0.4836279074937232, -2.19738087759112, 0.8110974564673381, 0.6360887472170061, 0.3197152547171901],
           'left': [0.3884776329219501, -0.9546968008083578, -0.9810819639199435, -1.9271960854579653, -0.8515225212999344, 1.6225280247076888, -0.4879094307978122],
           'middle':[0, 0, 0, -1.6, 0, 1.87, 0],
           'up': [0.0, -0.7, 0.0, -1.6, 0.0, 3.5, 0.7],
           'right':[-0.3948859155257995, -0.7270775750246515, 0.9705156650028227, -1.7756625474152135, -0.2109975244307001, 2.7461024636268965, 1.071975144388428],
           },
           'Z1':{'front': [-0.08810875890124847, 2.192829586859926, -1.7333875430498975, -0.36000533920198435, 0.17326781174833564, -1.9370062281298799],
           'left': [-0.9266763769103827, 1.7147408172359093, -1.1721008206747938, -0.2879519938608258, 0.23762854665550445, -0.44929498962951486],
           'middle':[0.005228521470025748, 0.4504296325181785, -0.8985288816139108, 0.48647146872395064, 0.16775067443267064, 0.24798319934352273],
           'up': [0.0018171402272963246, 1.172799383820571, -1.453885548356379, -0.28382613124306705, 0.10265041464355613, -0.22976814022886144],
           'right': [0.9116573261152304, 1.4085440983973196, -1.09840056058309, -0.024398221825178104, 0.4507839924308636, -1.1895309425464662],
           "down": [0.0010549407545234265, 0.26017824292386665, -0.16489775436491044, 0.14232966504401762, 0.041497875686174725, -1.581590191712276]
           }
           }

def demo(args):
    # 0. initial the Simulator(Franka, Z1)
    robot = RobotArm(args)

    # 1.需要的输入为: Configuration Space的起点和终点; 使用线性插值的手段生成初始轨迹
    begin_inCspace = np.array([0, 0, -1.04, 0.18, 0.245, -0.17])
    end_inCspace = np.array([0, 2.00, -1.50, -0.46, -0.27, -1.80])
    initial_trajectory = Joint_linear_initial(begin=begin_inCspace, end=end_inCspace, N=128)
    '''
    想要模仿的Demonstration
    目前的版本是在Cspace下直接生成轨迹，中间有个中间点进行弯折，比较简单，后期可以留给TODO
    '''
    external = np.array([-1.02, 1.58, -0.92, -0.40, -0.38, 0.26])
    demonstration = Generate_demo_demonstration(begin=begin_inCspace, end=end_inCspace, external=external, dimension=args.dimension)
    # initial_trajectory = Joint_linear_initial(begin=begin_inCspace, end=begin_inCspace, N=384)
    # demonstration = Generate_multi_demo_demonstration(begin=begin_inCspace, end=begin_inCspace, external_list=[POSTURE[args.Robot]["left"], POSTURE[args.Robot]["right"]], dimension=args.dimension)

    init_end_effector = robot.solveListKinematics(initial_trajectory)
    # Cost Function
    cost_function = Multi_Cost(robot=robot, init_trajectory=demonstration, args=args)
    # Sentry
    sentry = Sentry(initial_trajectory, cost_function, args)
    # 多维Stomp框架，目前这个是Panda的配置
    if args.Robot == 'Panda':
        stomp_py = Multi_dimensions_stomp(
            num_points=initial_trajectory.shape[0], args=args, cost_func=cost_function, 
            rangelimit_low_list=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]), 
            rangelimit_high_list=np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.752,2.8973]), 
            vellimit_low_list=np.array([-2.1750,-2.1750,-2.1750,-2.1750,-2.6100,-2.6100,-2.6100]), 
            vellimit_high_list=np.array([2.1750,2.1750,2.1750,2.1750,2.6100,2.6100,2.6100]),
            acclimit_low_list=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]), 
            acclimit_high_list=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
            )
    # 导入Z1相关参数
    elif args.Robot == 'Z1':
        stomp_py = Multi_dimensions_stomp(
            num_points=initial_trajectory.shape[0], args=args, cost_func=cost_function, 
            rangelimit_low_list=np.array([-2.6179938779914944,0.0,-2.8797932657906435,-1.5184364492350666,-1.3439035240356338,-2.792526803190927]), 
            rangelimit_high_list=np.array([2.6179938779914944,2.9670597283903604,0.0,1.5184364492350666,1.3439035240356338,2.792526803190927]), 
            vellimit_low_list=np.array([-3.1415,-3.1415,-3.1415,-3.1415,-3.1415,-3.1415]), 
            vellimit_high_list=np.array([3.1415,3.1415,3.1415,3.1415,3.1415,3.1415]),
            acclimit_low_list=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]), 
            acclimit_high_list=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
    # 远航者的相关配置
    voyager_Qcost_list = []
    voyager_Qcost_total_list = []
    voyager_iter_joints = []

    if args.STO == 'STODI':
        neighbour_Qcost_list = []
        neighbour_Qcost_total_list = []
        neighbour_iter_joints = []

    iter_num = 0
    total_iter_num = args.iter_num

    # 计算Cost标准步骤(for example)
    cost_function.Update_state(generate_multi_state(initial_trajectory, args=args))
    voyager_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
    voyager_Qcost_total_list.append(np.sum(voyager_Qcost_list[-1]))
    voyager_iter_joints.append(initial_trajectory)
    
    if args.STO == 'STODI':
        neighbour_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
        neighbour_Qcost_total_list.append(np.sum(neighbour_Qcost_list[-1]))
        neighbour_iter_joints.append(initial_trajectory)
    
    begin_time = time.time()
    # begin optimal loops
    
    while(iter_num < total_iter_num):
        iter_num += 1
        # 0. generate noisy trajectory
        stomp_py.multiDimension_diffusion()
        # 1.1 calculate voyager weights
        temp_iter_traj = copy.copy(sentry.pioneer['voyager'].traj)
        stomp_py.multi_update(joints_traj=sentry.pioneer['voyager'].traj)
        kn_weights = stomp_py.calculate_weights()
        n7_proved_noise = stomp_py.calculate_delta_noise(weights_p=kn_weights)

        # 1.2 get new trajectory
        temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] # N * 7
        # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
        # 1.3 limit check
        if stomp_py.multi_limit_check(temp_iter_traj):
            # 1.4 Update Voyager
            sentry.Update('voyager', temp_iter_traj)
            # 1.5 Using for reuse
            if args.reuse_state:
                stomp_py.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
            # Cost voyager cost( Just for visualize )
            cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
            voyager_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
            voyager_Qcost_total_list.append(np.sum(voyager_Qcost_list[-1]))
            # Record voyager
            voyager_iter_joints.append(copy.copy(temp_iter_traj))
        
        if args.STO == 'STODI':
            # 2.1 calculate neighbour weights
            temp_iter_traj = copy.copy(sentry.pioneer['neighbour'].traj)
            stomp_py.multi_update(joints_traj=sentry.pioneer['neighbour'].traj)
            kn_weights = stomp_py.calculate_weights()
            n7_proved_noise = stomp_py.calculate_delta_noise(weights_p=kn_weights)

            # 2.2 get new trajectory
            temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * 0.5 * n7_proved_noise[1:-1, :] # N * 7
            # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
            # 2.3 limit check
            if stomp_py.multi_limit_check(temp_iter_traj):
                # 2.4 Update Neighbour
                sentry.Update('neighbour', temp_iter_traj)
                # 2.5 Using for reuse
                if args.reuse_state:
                    stomp_py.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
                # Cost voyager cost( Just for visualize )
                cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
                neighbour_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
                neighbour_Qcost_total_list.append(np.sum(neighbour_Qcost_list[-1]))
                # Record voyager
                neighbour_iter_joints.append(copy.copy(temp_iter_traj))
            
            # 3. Update the best
            sentry.Update('best')
            # 4. clock align the best and the neighbour
            if iter_num % (total_iter_num / 10):
                sentry.Update('clock')

    print("Loop is over!")
    Optimization_time = time.time() - begin_time
    print('The total optimization time : {} s.'.format(Optimization_time))
    print(len(voyager_Qcost_list))

    if args.STO == 'STODI':
        final_traj_state = generate_multi_state(sentry.pioneer['best'].traj, args)
    elif args.STO == 'STOMP':
        final_traj_state = generate_multi_state(sentry.pioneer['voyager'].traj, args)
    
    robot.traj_torque_control(final_traj_state["position"], final_traj_state["velocity"], final_traj_state["acceleration"])
    print("演示结束!")

    if args.STO == 'STODI':
        end_effector = robot.solveListKinematics(sentry.pioneer['best'].traj)
    elif args.STO == 'STOMP':
        end_effector = robot.solveListKinematics(sentry.pioneer['voyager'].traj)
    
    demonstration_eff = robot.solveListKinematics(demonstration)
    Draw_3trajectory(init_end_effector, end_effector, demonstration_eff)

    writing_results = {
        'init_traj': init_end_effector.tolist(),
        'result_traj': end_effector.tolist(),
        'demonstration': demonstration_eff.tolist(),
    }
    
    if args.STO == 'STODI':
        writing_results['cost'] = np.array(neighbour_Qcost_list).tolist()
    elif args.STO == 'STOMP':
        writing_results['cost'] = np.array(voyager_Qcost_list).tolist()
    
    with open(f"{args.file_path}/demos/{args.expt_name}.json", 'w') as f:
        json.dump(writing_results, f)
    
    joints_info = {
        "position": final_traj_state["position"].tolist(),
        "velocity": final_traj_state["velocity"].tolist(),
        "acceleration": final_traj_state["acceleration"].tolist()
    }
    with open(f"{args.file_path}/demos/{args.expt_name}_joints.json", 'w') as f:
        json.dump(joints_info, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--file-path", type=str, default=r'./Pybullet')
    parser.add_argument("--expt-name", type=str, required=True)
    # STO参数
    parser.add_argument("--Robot", type=str, choices=["Panda", "Z1"], default="Z1") # 选择机器人模型
    parser.add_argument("--dimension", type=int, default=6) # 输入的维数:tips Panda是7，Z1是6
    parser.add_argument("--iter-num", type=int, default=20) # STO迭代的次数
    parser.add_argument("--reuse-state", type=bool, default=True) # 是否启用reuse
    parser.add_argument("--K", type=int, default=20) # STO选取的K条轨迹的数量 20
    parser.add_argument("--reuse-num", type=int, default=10) # reuse的数量 10
    parser.add_argument("--time", type=float, default=0.0) # STO的迭代时间
    parser.add_argument("--decay", type=float, default=0.9) # decay for better收敛
    parser.add_argument("--STO", type=str, choices=["STODI", "STOMP"], default="STODI") # 是什么随机优化框架
    # loss参数
    parser.add_argument("--ContourCost", type=str, choices=[None, "DTW", "MSES", "MSEPS", "NMSEPS", "MSE"], default="DTW") # 模仿学习的loss函数指标选择
    parser.add_argument("--ObstacleCost", type=str, choices=[None, "STOMP"], default="STOMP") # 避障的loss
    parser.add_argument("--ConstraintCost", type=str, choices=[None, "STOMP"], default="STOMP") # 约束的loss
    parser.add_argument("--TorqueCost", type=str, choices=[None, "STOMP"], default="STOMP") # 力矩的loss
    # 结果可视化参数
    parser.add_argument("--visual-loss", action="store_true")
    parser.add_argument("--visual-traj", action="store_true")
    # 其他参数
    parser.add_argument("--control-frequency", type=float, default=50) # 假定的控制频率
    parser.add_argument("--sample-frequency", type=float, default=50) # 假定的采样频率

    args = parser.parse_args()

    if not osp.isdir(f"{args.file_path}/demos/"):
        os.makedirs(f"{args.file_path}/demos/")

    # 单次选择运行（STOMP,STODI）的主函数
    demo(args=args)