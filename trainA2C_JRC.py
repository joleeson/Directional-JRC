#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Train an agent using Advantage Actor Critic (A2C) with this program. Running instructions in the README file.

J. Lee, Y. Cheng, D. Niyato, Y. L. Guan and G. David González,
"Deep Reinforcement Learning for Time Allocation and Directional Transmission in Joint Radar-Communication,"
2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2559-2564, doi: 10.1109/WCNC51071.2022.9771580.


"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process
from torch.distributions import Categorical, Normal
from torch.distributions.multinomial import Multinomial
from typing import Callable, Union

from trainPPO_JRC import Agent

from beamformJRCenv import Beamform_JRC
import json


# In[]

device = torch.device(0 if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

def save_itr_info(fname, itr, av_reward):
    with open(fname, "w") as f:
        f.write(f"Iteration: {itr}\n Average reward: {av_reward}\n")

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)
    
def save_variables(model, model_file):
    """Save parameters of the NN 'model' to the file destination 'model_file'. """
    torch.save(model.state_dict(), model_file)

def load_variables(model, load_path):
#    model.cpu()
    model.load_state_dict(torch.load(load_path)) #, map_location=lambda storage, loc: storage))
    # model.eval() # TODO1- comment:  to set dropout and batch normalization layers to evaluation mode before running inference
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func



# In[]

class A2CAgent(Agent):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        Agent.__init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args)
        
        
    def update(self,ob_no, ac_na, adv_n, log_prob_na_old, next_ob_no, re_n, terminal_n, update=0, h_ns1=None):      
        policy_loss, policy_loss_after = {}, {}
        num_mb = int(np.ceil(self.timesteps_this_batch / self.mb_size))     # num minibatches
        
        loss, v_loss, v_loss_after = {}, {}, {}
        loss_agent = {}
        losses, pg_losses, kl_divs, approx_kl_divs, value_losses, entropy_losses = {}, {}, {}, {}, {}, {}
        losses_agent = {}
        v_loss_av = 0
        
        
        for i in range(1, self.num_users+1):
            losses[(i)], pg_losses[(i)], kl_divs[(i)], value_losses[(i)], entropy_losses[(i)] = [], [], [], [], []
        for j in range(1, self.num_agents+1):
            loss_agent[(j)] = 0
            losses_agent[(j)] = 0
        
        for epoch in range(self.ppo_epochs):
            shuffle_idx = np.random.permutation(self.timesteps_this_batch)
            mb_idx = np.arange(num_mb)
            np.random.shuffle(mb_idx)
            
            target_n = self.eval_target(next_ob_no, re_n, terminal_n)
            
            for i in range(1, self.num_users+1):
                approx_kl_divs[(i)] = []
            
            for k in mb_idx:
                idx = shuffle_idx[k*self.mb_size : (k+1)*self.mb_size]
                values, log_prob_na, entropy = self.eval_ac(ob_no, ac_na, idx, h_ns1)
                

                for i in range(self.num_intelligent):
                    policy_loss[(i+1)] = 0
                    policy_loss_after[(i+1)] = 0
                    
                    # Calc policy loss
                    policy_loss[(i+1)] = - torch.mul(log_prob_na[(i+1)], adv_n[(i+1)][idx]).mean()
                    
                    # Calc value loss
                    v_loss[(i+1)] = (values[(i+1)] - target_n[(i+1)][idx]).pow(2).mean()
                    v_loss_av = v_loss_av + v_loss[(i+1)].detach()/self.num_users
                
                    loss[(i+1)] = policy_loss[(i+1)] + self.v_coef * v_loss[(i+1)] - self.entrop_loss_coef * entropy[(i+1)]
                    j = int(np.ceil((i+1)/self.users_per_agent))
                    loss_agent[(j)] = loss_agent[(j)] + loss[(i+1)]
                    
                    # Logging
                    losses[(i+1)].append(loss[(i+1)].item())
                    pg_losses[(i+1)].append(policy_loss[(i+1)].item())
                    approx_kl_divs[(i+1)].append(torch.mean(log_prob_na_old[(i+1)][idx]- log_prob_na[(i+1)]).item())
                    value_losses[(i+1)].append(v_loss[(i+1)].item())
                    entropy_losses[(i+1)].append(entropy[(i+1)].item())
                
                for j in range(self.num_agents):
                    self.optimizer[(j+1)].zero_grad()
                    loss_agent[(j+1)].backward() #retain_graph = True)
                    nn.utils.clip_grad_norm_(self.net[(j+1)].parameters(), self.max_grad_norm)
                    self.optimizer[(j+1)].step()
                    loss_agent[(j+1)] = 0
                
                    
        # Log
        for i in range(1, self.num_intelligent+1):
            logz.log_tabular("Loss "+str(i), np.mean(losses[(i)]))
            logz.log_tabular("Policy Gradient Loss "+str(i), np.mean(pg_losses[(i)]))
            logz.log_tabular("KL Divergence "+str(i), np.mean(approx_kl_divs[(i)]))
            logz.log_tabular("Value Loss "+str(i), np.mean(value_losses[(i)]))
            logz.log_tabular("Entropy Loss "+str(i), np.mean(entropy_losses[(i)]))
            

# In[]
        
def train_AC(
        exp_name,
        env_name,
        env_config,
        num_intelligent,
        mode,
        radar_interval,
        n_iter, 
        gamma,
        lamb,
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate, 
        reward_to_go, 
        gae,
        n_step,
        animate, 
        logdir, 
        normalize_advantages,
        critic,
        decentralised_critic,
        seed,
        n_layers,
        conv,
        size,
        size_critic,
        recurrent,
        input_prev_ac,
        ppo_epochs,
        minibatch_size,
        ppo_clip,
        v_coef,
        entrop_loss_coef,
        max_grad_norm,
        policy_net_dir=None,
        value_net_dir=None,
        test = None):
    
    start = time.time()
    setup_logger(logdir, locals())  # setup logger for results
    
    env = Beamform_JRC(env_config)
    
    env.seed(seed)
    torch.manual_seed(seed)
#    np.random.seed(seed)
    
    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps
    
    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]                                 # OK
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]  # OK
    
    num_users = int(env.N)
    
    entrop_loss_coef = linear_schedule(entrop_loss_coef)
       
    computation_graph_args = {
        'num_intelligent': num_intelligent,
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'num_users': env_config['num_users'],
        'num_users_NN': env_config['num_users_NN'],
        'num_agents': env_config['num_agents'],
        'mode': mode,
        'radar_interval': radar_interval,
        'discrete': discrete,
        'size': size,
        'CNN': conv,
        'size_critic': size_critic,
        'learning_rate': learning_rate,
        'recurrent': recurrent,
        'input_prev_ac': input_prev_ac,
        'ppo_epochs': ppo_epochs,
        'minibatch_size': minibatch_size,
        'ppo_clip': ppo_clip,
        'v_coef': v_coef,
        'entrop_loss_coef': entrop_loss_coef,
        'max_grad_norm': max_grad_norm,
        'test': test,
        }
    
    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'lambda': lamb,
        'reward_to_go': reward_to_go,
        'gae': gae,
        'n_step': n_step,
        'critic': critic,
        'decentralised_critic': decentralised_critic,
        'normalize_advantages': normalize_advantages,
    }
    
    agent = A2CAgent(computation_graph_args, sample_trajectory_args, estimate_return_args)
    
    solved = False
    if policy_net_dir != None:
        load_variables(agent.net, policy_net_dir)
        solved = True
    
        
    #========================================================================================#
    # Training Loop
    #========================================================================================#
    
    total_timesteps = 0
    best_av_reward = None
    policy_model_file = {}
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch
        
        agent.update_current_progress_remaining(itr, n_iter)
        agent.update_entropy_loss()

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no, ac_na, re_n, log_prob_na, next_ob_no, next_ac_na, h_ns1, entropy = {}, {}, {}, {}, {}, {}, {}, {}
        returns = np.zeros((num_users,len(paths)))
        for i in range(num_intelligent):
            ob_no[(i+1)] = np.concatenate([path[(i+1)]["observation"] for path in paths])         # shape (batch self.size /n/, ob_dim)
            ac_na[(i+1)] = np.concatenate([path[(i+1)]["action"] for path in paths]) #, axis = -2)   # shape (batch self.size /n/, ac_dim) recurrent: (1, n, ac_dim)
            re_n[(i+1)] = np.concatenate([path[(i+1)]["reward"] for path in paths])               # (batch_size, num_users)
            log_prob_na[(i+1)] = torch.cat([path[(i+1)]["log_prob"] for path in paths])          # torch.size([5200, ac_dim])
            next_ob_no[(i+1)] = np.concatenate([path[(i+1)]["next_observation"] for path in paths]) # shape (batch self.size /n/, ob_dim)
            next_ac_na[(i+1)] = np.concatenate([path[(i+1)]["next_action"] for path in paths]) # shape (batch self.size /n/, ac_dim)
            if agent.recurrent:
                h_ns1[(i+1)] = torch.cat([path[(i+1)]["hidden"] for path in paths],dim=1)    #torch.size([1, batchsize, 32])
            assert ob_no[(i+1)].shape == (timesteps_this_batch,ob_dim)
            assert ac_na[(i+1)].shape == (timesteps_this_batch,)
            assert re_n[(i+1)].shape == (timesteps_this_batch,)
            # assert log_prob_na[(i+1)].shape == torch.Size([timesteps_this_batch])
            assert next_ob_no[(i+1)].shape == (timesteps_this_batch,ob_dim)
            assert next_ac_na[(i+1)].shape == (timesteps_this_batch,)
        
        for i in range(num_users):
            returns[i,:] = [path[(i+1)]["reward"].sum(dtype=np.float32) for path in paths]   # (num_users, num episodes in batch)
            assert returns[i,:].shape == (timesteps_this_batch/400,)
        
        terminal_n = np.concatenate([path["terminal"] for path in paths])       # (batch_size,)
        
        r_comm1 = ([path["r_comm1"] for path in paths])  
        r_rad1 = ([path["r_rad1"] for path in paths])    
        throughput1 = ([path["throughput1"] for path in paths])                 # (batch,)
        throughput_av = ([path["throughput_av"] for path in paths])
        SINR1 = ([path["SINR1"] for path in paths])                      # (batch,)
        SINR_av = ([path["SINR_av"] for path in paths])
        num_transmits1 = ([path["num_transmits1"] for path in paths])
        num_transmits_av = ([path["num_transmits_av"] for path in paths])
        
        av_reward = np.mean(returns)
        
        # Log diagnostics
        ep_lengths = [pathlength(path[1]) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("Average Reward", av_reward)   # per agent per episode
        logz.log_tabular("StdReward", np.std(returns))
        logz.log_tabular("MaxReward", np.max(returns))
        logz.log_tabular("MinReward", np.min(returns))
        logz.log_tabular("r_comm1", np.mean(r_comm1))
        logz.log_tabular("r_rad1", np.mean(r_rad1))
        logz.log_tabular("Throughput1", np.mean(throughput1))
        logz.log_tabular("Average Throughput", np.mean(throughput_av))
        logz.log_tabular("SINR1", np.mean(SINR1))
        logz.log_tabular("SINR_av", np.mean(SINR_av))
        logz.log_tabular("Num Transmits 1", np.mean(num_transmits1))
        logz.log_tabular("Average Num Transmits", np.mean(num_transmits_av))
        
        for i in range(num_users):
            logz.log_tabular("Reward"+str(i+1), np.mean(returns, axis=1)[i])
            logz.log_tabular("StdReward"+str(i+1), np.std(returns, axis=1)[i])
            logz.log_tabular("MaxReward"+str(i+1), np.max(returns, axis=1)[i])
            logz.log_tabular("MinReward"+str(i+1), np.min(returns, axis=1)[i])
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        
        if test != True:
            adv_n = agent.estimate_decen_adv(ob_no, next_ob_no, re_n, terminal_n)
            agent.update(ob_no, ac_na, adv_n, log_prob_na, next_ob_no, re_n, terminal_n)
            
            reward1 = np.mean(returns, axis=1)[0]
        
            if best_av_reward == None:
                best_av_reward = reward1
            elif reward1 > best_av_reward:
                best_av_reward = reward1
                for i in range(env_config['num_agents']):
                    policy_model_file[(i+1)] = os.path.join(logdir,"NN"+str(i+1)+'_itr'+str(itr))
                    save_itr_info(f"{policy_model_file[(i+1)]}-{itr}.txt", itr, reward1)
                    save_variables(agent.net[(i+1)], policy_model_file[(i+1)])
        
        logz.dump_tabular(step=itr)
    
    
    if test != True:
        for i in range(env_config['num_agents']):
            policy_model_file[(i+1)] = os.path.join(logdir,"NN"+str(i+1))
            save_itr_info(f"{policy_model_file[(i+1)]}-{itr}.txt", itr, reward1)
            save_variables(agent.net[(i+1)], policy_model_file[(i+1)])
        
    

# In[]

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    # Env config
    parser.add_argument('env_name', type=str)
    parser.add_argument('--num_lanes', type=int, default=2)
    parser.add_argument('--num_users_NN', type=int)
    parser.add_argument('--num_users', type=int, nargs='+', default = [0])
    parser.add_argument('--num_intelligent', type=int, default = 0)
    parser.add_argument('--num_agents', type=int)
    parser.add_argument('--obj', choices=['peak','avg'], default='peak')
    parser.add_argument('--x_dim', type=int, nargs='+', default=[150, 150, 10])
    parser.add_argument('--age_max', type=float, default=3)
    parser.add_argument('--w_comm', type=float, default=1)
    parser.add_argument('--w_rad', type=float, default=1)
    parser.add_argument('--ang_range', type=float, default=45)
    parser.add_argument('--ob_time', action='store_true')
    parser.add_argument('--radar_interval', '-r_int', type=int, default=4)
    
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    
    # Algorithm hyperparameters
    parser.add_argument('--mode', choices=['unif_rand','urg5','best','csma-ca','rotate'], default='rotate') # policy for unintelligent agents
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--lamb', type=float, default=0.95)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--generalised_adv_est', '-gae', action='store_true')
    parser.add_argument('--n_step', '-ns', type=int, default=0)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--critic', type=str, default ='v')
    parser.add_argument('--decentralised_critic', '-dc', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--size', '-s', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--size_critic', '-sc', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--recurrent', '-r', action='store_true')
    parser.add_argument('--input_prev_ac','-ipa', action='store_true')
    parser.add_argument('--ppo_epochs', type=int, default=10)
    parser.add_argument('--minibatch_size', '-mbs', type=int, default=64)
    parser.add_argument('--ppo_clip', type=float, default=0.2) #0.05
    parser.add_argument('--v_coef', type=float, default=0.5)
    parser.add_argument('--entrop_loss_coef', type=float, default=0) #0.001) #0.0005
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--ac_dist', default='Gaussian')
    
    
    parser.add_argument('--policy_net_filename', '-p_file', type=str)
    parser.add_argument('--value_net_filename', '-v_file', type=str)
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--pre_uuid', '-uuid', type=str)
    args = parser.parse_args()
    
    
    if not(os.path.exists('data')):
        os.makedirs('data')
    
    if args.pre_uuid != None:
        logdir = args.exp_name + '_' + args.env_name + '_' + args.pre_uuid
        logdir = os.path.join('data/' + args.pre_uuid , logdir)
    else:
        logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join('data', logdir)
    
    print("------")
    print(logdir)
    print("------")
    
       
    max_path_length = args.ep_len if args.ep_len > 0 else None
    args.policy_net_dir = None
    args.value_net_dir = None
    
    if args.policy_net_filename != None:
        args.policy_net_dir = os.path.join(os.getcwd(),args.policy_net_filename)
    # if args.value_net_filename != None:
    #     args.value_net_dir = os.path.join(os.getcwd(),args.value_net_filename)
    
    if args.num_intelligent == 0:
        args.num_intelligent = args.num_users
    if args.num_users == [0]:
        args.num_users[0] = args.num_users_NN
    
    
    for num_users in args.num_users:
        for x_dim in range(args.x_dim[0], args.x_dim[1], args.x_dim[2]):
            for e in range(args.n_experiments):
                
                seed = args.seed + 10*e
                print('Running experiment with seed %d'%seed)
                
                test = False
                if args.policy_net_filename != None:
                    test = True
                
                env_config = {'num_users': num_users,
                              'num_agents': args.num_agents,
                              'num_users_NN': args.num_users_NN,
                              'age_max': args.age_max,
                              'x_dim': x_dim,
                              'num_lanes': args.num_lanes,
                              'w_comm': args.w_comm,
                              'w_rad': args.w_rad,
                              'radar_angular_range': args.ang_range,
                              'ob_time': args.ob_time,
                              }
                
                logdir_w_params = logdir + '_{}users_wcomm1{}_w_rad{}_maxage_{}_ang{}_obtime{}'.format(num_users,args.w_comm, args.w_rad,args.age_max,args.ang_range,args.ob_time)
                        
                train_AC(
                        exp_name=args.exp_name,
                        env_name=args.env_name,
                        env_config=env_config,
                        num_intelligent=args.num_intelligent,
                        mode = args.mode,
                        radar_interval = args.radar_interval,
                        n_iter=args.n_iter,
                        gamma=args.discount,
                        lamb=args.lamb,
                        min_timesteps_per_batch=args.batch_size,
                        max_path_length=max_path_length,
                        learning_rate=args.learning_rate,
                        reward_to_go=args.reward_to_go,
                        gae = args.generalised_adv_est,
                        n_step = args.n_step,
                        animate=args.render,
                        logdir=os.path.join(logdir_w_params,'%d'%seed),
                        normalize_advantages=not(args.dont_normalize_advantages),
                        critic=args.critic,
                        decentralised_critic= args.decentralised_critic,
                        seed=seed,
                        n_layers=args.n_layers,
                        size=args.size,
                        conv=args.conv,
                        size_critic=args.size_critic,
                        recurrent = args.recurrent,
                        input_prev_ac = args.input_prev_ac,
                        ppo_epochs = args.ppo_epochs,
                        minibatch_size = args.minibatch_size,
                        ppo_clip = args.ppo_clip,
                        v_coef = args.v_coef,
                        entrop_loss_coef = args.entrop_loss_coef,
                        max_grad_norm = args.max_grad_norm,
                        policy_net_dir = args.policy_net_filename,
                        value_net_dir = args.value_net_filename,
                        test = args.test
                        )