# Scheduling Directional Joint Radar-Communication

Supplementary material for the following paper:
  
[J. Lee, Y. Cheng, D. Niyato, Y. L. Guan and G. David González, "Deep Reinforcement Learning for Time Allocation and Directional Transmission in Joint Radar-Communication," 2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2559-2564, doi: 10.1109/WCNC51071.2022.9771580.](https://ieeexplore.ieee.org/abstract/document/9771580)

Also available on [Digital Repository of NTU](https://hdl.handle.net/10356/155437).

Current strategies for joint radar-communication (JRC) rely on prior knowledge of the communication and radar systems within the vehicle network. In this paper, we propose a framework for intelligent vehicles to conduct JRC, with minimal prior knowledge, in an environment where surrounding vehicles execute radar detection periodically, which is typical in contemporary protocols. We introduce a metric on the usefulness of data to help the vehicle decide what, and to whom, data should be transmitted. The problem framework is cast as a Markov Decision Process (MDP). We show that deep reinforcement learning results in superior performance compared to nonlearning algorithms. In addition, experimental results show that the trained deep reinforcement learning agents are robust to changes in the number of vehicles in the environment.

## Getting started
Install the dependencies listed in 'requirements.txt'.

## Running Experiments
Experiemnts may be run from the command line. Agents using the "Heuristic" or "Round Robin" policies can be run using the examples below:

Heuristic
```
python test_beamformJRC_multi.py beamform_JRC_4lane --w_comm 1 --w_rad 8 --num_lanes 4 --num_users_NN 8 --num_agents 1 --mode heuristic --age_max 8 --ang_range 35 --x_dim 150 160 10 -ep 400 -n 500 -b 2000 -e 3 --CW 2 4 --exp_name test_heuristic
```

Round Robin
```
python test_beamformJRC_multi.py beamform_JRC_4lane --w_comm 1 --w_rad 8 --num_lanes 4 --num_users_NN 8 --num_agents 1 --mode rotate --age_max 8 --ang_range 35 --x_dim 150 160 10 -ep 400 -n 500 -b 2000 -e 3 --CW 2 4 --exp_name test_round_robin
```

Train an agent using the deep reinforcement learning algorithms Advantage Actor-Critic (A2C) or Proximal Policy Optimisation (PPO) using the examples below:

A2C:
```
python trainA2C_JRC.py beamformJRC_4lane --w_comm 1 --w_rad 8 --num_lanes 4 --num_users_NN 8 --num_users 8 --num_intelligent 1 --num_agents 1 --mode rotate --age_max 8 --ang_range 35 --x_dim 150 160 10 -s 64 64 -sc 64 64 -dc -ep 400 -n 500 -b 2000 -e 3 -lr 0.0001 --entrop_loss_coef 0.01 --ob_time --exp_name A2C_8vehicles
```

PPO:
```
python trainPPO_JRC.py beamformJRC_4lane --w_comm 1 --w_rad 8 --num_lanes 4 --num_users_NN 8 --num_users 8 --num_intelligent 1 --num_agents 1 --mode rotate --age_max 8 --ang_range 35 --x_dim 150 160 10 -s 64 64 -sc 64 64 -dc -ep 400 -n 500 -b 2000 -e 3 -lr 0.0001 --entrop_loss_coef 0.01 --ob_time --exp_name PPO_8vehicles
```