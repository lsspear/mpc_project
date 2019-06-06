# mpc_project
Project exploring scenario generation methods for MPC applications in energy

This project is inspired by the paper [Dynamic Energy Management](http://stanford.edu/~boyd/papers/dyn_ener_man.html), and uses the library, wind farm data, and example code out of the repo associated with their paper, [cvxpower](https://github.com/cvxgrp/cvxpower).

Essentially, the goal of the project is to explore variations to the Robust Model Predictive Control framework used to solve the problem of operating a battery and generator on a network that has a fixed load (which requires a certain amount of power in each interval) and a windfarm (which produces and unknown amount of power in each interval). 

The main effort so far has been on exploring the use of Gaussian Processes as both forecasters and scenario generators. The main notebook used for experiments is the Gaussian_Process.ipynb notebook. A list of the objective value of every experiment run is in the MPC_experiment_results.pdf. 
