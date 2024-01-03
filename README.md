# Combining physics-informed graph neural network and finite difference for solving forward and inverse spatiotemporal PDEs

This repo is the official implementation of ["Combining physics-informed graph neural network and finite difference for solving forward and inverse spatiotemporal PDEs"](https://doi.org/10.1145/3590003.3590029) by Hao Zhang, Longxiang Jiang, Xinkun Chu, Yong Wen, Luxiong Li, Yonghao Xiao, and Liyuan Wang$^{*}$.

## Abstract
The great success of Physics-Informed Neural Networks (PINN) in solving partial differential equations (PDEs) has significantly advanced our simulation and understanding of complex physical systems in science and engineering. However, many PINN-like methods are poorly scalable and are limited to in-sample scenarios. To address these challenges, this work proposes a novel discrete approach termed Physics-Informed Graph Neural Network (PIGNN) to solve forward and inverse nonlinear PDEs. In particular, our approach seamlessly integrates the strength of graph neural networks (GNN), physical equations and finite difference to approximate solutions of physical systems. Our approach is compared with the PINN baseline on three well-known nonlinear PDEs (heat, Burgers and FitzHugh-Nagumo). We demonstrate the excellent performance of the proposed method to work with irregular meshes, longer time steps, arbitrary spatial resolutions, varying initial conditions (ICs) and boundary conditions (BCs) by conducting extensive numerical experiments. Numerical results also illustrate the superiority of our approach in terms of accuracy, time extrapolability, generalizability and scalability. The main advantage of our approach is that models trained in small domains with simple settings have excellent fitting capabilities and can be directly applied to more complex situations in large domains.


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
and,the results images will be saved in the `images` folder.



