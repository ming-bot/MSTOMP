# MSTOMP
Multi-policy Stochastic Trajectory Optimization for Motion Planning

## 1. Introduction

This Stochastic Trajectory Optimization Method is explained in our paper: **Stochastic Trajectory Optimization for Robotic Skill Acquisition From a Suboptimal Demonstration**. In this code base, we present an example demo to show how to use MSTOMP in a Learning from Demonstration Task. We use both the Franka Panda and Unitree Z1 robotic arm in the Pybullet to show our method's performance.



## 2. Quick use

### 2.1 Environment Setup

```
pip install pybullet, numpy
```

You may occur error message : error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output], then go to the website mentioned above and download VStudio



### 2.2 Run the demo

```shell
python ./Pybullet/demo.py --expt-name='your expert name'
```

Please cite our paper if you use this code:

```
@ARTICLE{10976394,
  author={Ming, Chenlin and Wang, Zitong and Zhang, Boxuan and Cao, Zhanxiang and Duan, Xiaoming and He, Jianping},
  journal={IEEE Robotics and Automation Letters}, 
  title={Stochastic Trajectory Optimization for Robotic Skill Acquisition From a Suboptimal Demonstration}, 
  year={2025},
  volume={10},
  number={6},
  pages={6127-6134},
  keywords={Trajectory;Costs;Noise measurement;Cost function;Trajectory optimization;Measurement;Frequency-domain analysis;Planning;Motion and path planning;optimization and optimal control;learning from demonstration},
  doi={10.1109/LRA.2025.3564208}}
```

