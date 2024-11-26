# Combining physics-informed graph neural network and finite difference for solving forward and inverse spatiotemporal PDEs

This repo is the official implementation of ["Combining physics-informed graph neural network and finite difference for solving forward and inverse spatiotemporal PDEs"](https://doi.org/10.1145/3590003.3590029) by Hao Zhang, Longxiang Jiang, Xinkun Chu, Yong Wen, Luxiong Li, Jianbo Liu, Yonghao Xiao, and Liyuan Wang$^{*}$.

## Abstract
The great success of Physics-Informed Neural Network (PINN) in addressing partial differential equations (PDEs) has enhanced our ability to simulate and understand complex physical systems across various science
and engineering disciplines. Despite their achievements, existing PINN-like methods often face limitations in scalability and are primarily effective within in-sample scenarios. To overcome these challenges, 
this work proposes a novel discrete approach termed Physics-Informed Graph Neural Network (PIGNN) to solve both forward and inverse problems associated with nonlinear PDEs. Our approach seamlessly integrates 
the strength of graph neural network (GNN), physical laws and finite difference method to approximate the solutions of physical systems. Through a series of comprehensive numerical experiments, we compare the
performance of our PIGNN against the established PINN baseline using three well-known nonlinear PDEs: the heat equation, the Burgers equation, and the FitzHugh-Nagumo equation. Experimental outcomes high-
light the superior performance of our PIGNN in handling irregular meshes, long time steps, flexible spatial resolutions, and diverse initial and boundary conditions. These results also demonstrate the superiority
of our approach in terms of accuracy, time extrapolability, generalizability and scalability. A key advantage of our approach lies in its exceptional adaptability: models initiallytrained on small, simplified domains
exhibit robust fitting capabilities that can be seamlessly transferred to more complex, larger-scale scenarios.


## Example

We provide example for solving heat, Burgers and FitzHugh-Nagumo equations, just create a conda environment with python==3.8
```
conda create -n meshpde python==3.8 && conda activate meshpde
```
then, install the required package with

```
pip install -r requirements.txt
```
and start the training process with

```
python train.py
```
When train finished, to evaluate the trained model and visualize solution results, just run 
```
python test.py
```
and,the results images will be saved in the `testImg_save_dir` folder.



