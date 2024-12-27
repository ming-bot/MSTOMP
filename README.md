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

