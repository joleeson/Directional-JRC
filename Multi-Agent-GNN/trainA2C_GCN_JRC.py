#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%pdb
"""

With this program, train multiple agents agent using Advantage Actor Critic (A2C) with a Graph Neural Network(GNN).
Running instructions in the README file.

J. Lee, Y. Cheng, D. Niyato, Y. L. Guan and D. González G.,
"Intelligent Resource Allocation in Joint Radar-Communication With Graph Neural Networks,"
in IEEE Transactions on Vehicular Technology, vol. 71, no. 10, pp. 11120-11135, Oct. 2022, doi: 10.1109/TVT.2022.3187377.

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
from torch.distributions import Categorical, Normal
# from typing import Callable, Union
import torch_geometric
from torch_geometric.nn import GCNConv, TransformerConv, knn_graph, knn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from trainPPO_GCN_JRC import Agent

from beamformJRCenv_multi import Beamform_JRC
from utils import *


# In[]

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)
    
def masking(edge_index, in_map, device, msg_ac=None):
    in_map = torch.tensor(in_map, device=device)
    adj = to_dense_adj(edge_index).squeeze(0)
    adj = adj.fill_diagonal_(0)
    adj = adj * (in_map * in_map.reshape(-1,1))
    if msg_ac is not None:
        adj = adj * msg_ac.reshape((-1,1))
    edge_index, _ = dense_to_sparse(adj)
    
    return edge_index


# In[]

class A2CAgent(Agent):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        Agent.__init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args)
    
    def update(self,ob_no, ac_na, adv_n, log_prob_na_old, next_ob_no, re_n, terminal_n, hidden=None, next_hidden=None, msg_ac=None, log_prob_msgs_old=None):
        
        batch_size = self.mb_size # len(ob_no)
        # num_mb = np.ceil(len(ob_no)/batch_size)
        ob_dataset = RLDataset(ob_no, next_ob_no)
        ob_loader = DataLoader(ob_dataset, batch_size = batch_size, shuffle = True)
        
        if self.ic3net:
            log_prob_na_old = log_prob_na_old + log_prob_msgs_old
        
        target_n = []
        for epoch in range(self.ppo_epochs):
            mb_idx = 0
            for ob_batch, next_ob_batch, idx in ob_loader:
                ''' Sample minibatch and learn '''
                # check
                # torch.sum(next_ob_no[idx[1]].x - next_ob_batch[1].x) == 0
                
                ob_batch.to(self.device)
                next_ob_batch.to(self.device)
                
                if epoch == 0:
                    ''' ob_batch.x: tensor (minibatch size * #agents, ob_size)
                    reshape hiddens: tensor (minibatch size * #agents, hidden_size) '''
                    next_hidden_mb = next_hidden[idx].reshape((-1,self.size[0])) if self.recurrent else None
                    target_n.append(self.eval_target(next_ob_batch, re_n[idx], terminal_n[idx], next_hidden_mb))    # torch.size([batch, #users])
                
                hidden_mb = hidden.clone()[idx].reshape((-1,self.size[0])) if self.recurrent else None
                msg_ac_mb = msg_ac[idx] if self.ic3net else None
                values, log_prob_na, entropy, log_prob_msg = self.eval_ac(ob_batch, ac_na[idx], hidden_mb, msg_ac_mb)       # [batch, #users], [batch, #users], []
                
                if self.ic3net:
                    log_prob_na = log_prob_na + log_prob_msg    # multiply both action probabilities, assuming independence
                
                policy_loss, policy_loss_after = 0, 0
                # Calc policy loss
                policy_loss = - torch.mul(log_prob_na, adv_n[idx]).mean()
                
                # Calc value loss
                v_loss = (values - target_n[mb_idx]).pow(2).mean()
                
                loss = policy_loss + self.v_coef * v_loss - self.entrop_loss_coef * entropy
                
                # Logging
                losses = loss.item()
                pg_losses = policy_loss.item()
                approx_kl_divs = torch.mean(log_prob_na_old[idx] - log_prob_na).item()
                value_loss = v_loss.item()
                entropy_loss = entropy.item()
                
                # Back propagate
                self.optimizer.zero_grad()
                loss.backward() #retain_graph = True)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Checking
                if (epoch == 0) & (mb_idx == 0):
                    logz.log_tabular("CriticLossBefore", v_loss.detach().cpu().numpy())
                    logz.log_tabular("LossBefore", loss.detach().cpu().numpy())
                
                if (epoch == (self.ppo_epochs-1)) & (mb_idx == 0):
                    values_after, log_prob_after, entropy_after, log_prob_msg_after = self.eval_ac(ob_batch, ac_na[idx], hidden_mb, msg_ac_mb)
                    
                    if self.ic3net:
                        log_prob_after = log_prob_after + log_prob_msg_after    # multiply both action probabilities, assuming independence
                    policy_loss_after = - torch.mul(log_prob_after, adv_n[idx]).mean()
                    
                    # Calc value loss
                    v_loss_after = (values_after - target_n[mb_idx]).pow(2).mean()
                    
                    loss_after = policy_loss_after + self.v_coef * v_loss_after + self.entrop_loss_coef * entropy_after
                    
                    logz.log_tabular("LossAfter", loss_after.detach().cpu().numpy())
                    logz.log_tabular("CriticLossAfter", v_loss_after.detach().cpu().numpy())
                    
                mb_idx += 1
                    
        # Log
        logz.log_tabular("Loss ", np.mean(losses))
        logz.log_tabular("Policy Gradient Loss ", np.mean(pg_losses))
        logz.log_tabular("KL Divergence ", np.mean(approx_kl_divs))
        logz.log_tabular("Value Loss ", np.mean(value_loss))
        logz.log_tabular("Entropy Loss ", np.mean(entropy_loss))
            

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
        gnn,
        gnn_pooling,
        gnn_knn,
        ic3net,
        size,
        size_critic,
        recurrent,
        detach_gap,
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
        'GNN': gnn,
        'GNN_pooling': gnn_pooling,
        'GNN_kNN': gnn_knn,
        'ic3net': ic3net,
        'size_critic': size_critic,
        'learning_rate': learning_rate,
        'recurrent': recurrent,
        'detach_gap': detach_gap,
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
    
    test = env_config['test']
    if policy_net_dir != None:
        load_variables(agent.net, policy_net_dir)
        test = True
    
        
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
        
        agent.update_current_progress_remaining(itr, 5/7*n_iter)
        agent.update_entropy_loss()

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        
        ob_no, next_ob_no = [], []
        for path in paths:
            ob_no = ob_no + path['observation']                     # list len batch_size
            next_ob_no = next_ob_no + path['next_observation']
        if recurrent:
            hidden = torch.cat([path["hidden"] for path in paths])              # shape (batch_size, #users, hidden_size)
            next_hidden = torch.cat([path["next_hidden"] for path in paths])    # shape (batch_size, #users, hidden_size)
        else:
            hidden, next_hidden = None, None
        if ic3net:
            msg_ac = torch.cat([path["msg_ac"] for path in paths])              
            log_prob_msg = torch.cat([path["log_prob_msg"] for path in paths])  # shape (batch_size, #users)
        else:
            msg_ac, log_prob_msg = None, None
        ac_na = np.concatenate([path["action"] for path in paths]) #, axis = -2)   # shape (batch_size, #users)
        re_n = np.concatenate([path["reward"] for path in paths])               # (batch_size, num_users)
        log_prob_na = torch.cat([path["log_prob"] for path in paths])          # torch.size([batcb_size, #agents])
        next_ac_na = np.concatenate([path["next_action"] for path in paths]) # shape (batch self.size /n/)
        returns = np.array([path["reward"].sum(axis=0, dtype=np.float32) for path in paths]).transpose()   # (num_users, num episodes in batch)
        terminal_n = np.concatenate([path["terminal"] for path in paths])       # (batch_size,)
        
        r_comm1 = ([path["r_comm1"] for path in paths])  
        r_rad1 = ([path["r_rad1"] for path in paths])    
        throughput1 = ([path["throughput1"] for path in paths])                 # (batch,)
        throughput_av = ([path["throughput_av"] for path in paths])
        SINR1 = ([path["SINR1"] for path in paths])                      # (batch,)
        SINR_av = ([path["SINR_av"] for path in paths])
        num_transmits1 = ([path["num_transmits1"] for path in paths])
        num_transmits_av = ([path["num_transmits_av"] for path in paths])
        
        if test:
            lane_action_map = np.sum([path["lane_action_map"] for path in paths], axis=0)
            comm_distance = np.concatenate([path["comm_distance"] for path in paths], axis=0)
            radar_NN_distance = np.concatenate([path["radar_NN_distance"] for path in paths], axis=0)
            NN_distance = np.concatenate([path["NN_distance"] for path in paths], axis=0)
        
        av_reward = np.mean(returns)
        
        # Log diagnostics
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
            adv_n = agent.estimate_decen_adv(ob_no, next_ob_no, re_n, terminal_n, hidden, next_hidden)
            agent.update(ob_no, ac_na, adv_n, log_prob_na, next_ob_no, re_n, terminal_n, hidden, next_hidden, msg_ac, log_prob_msg)
        
            if best_av_reward == None:
                best_av_reward = av_reward
            elif av_reward > best_av_reward:
                best_av_reward = av_reward
                policy_model_file = os.path.join(logdir,"NN"+'_itr'+str(itr))
                save_itr_info(f"{policy_model_file}-{itr}.txt", itr, av_reward)
                save_variables(agent.net, policy_model_file)
        
            logz.dump_tabular(step=itr)
        else:
            logz.log_tabular("Lane-Action Map", lane_action_map)
            logz.log_tabular("Comm Distance", comm_distance)
            logz.log_tabular('Radar NN Distance', radar_NN_distance)
            logz.log_tabular('NN Distance', NN_distance)
            logz.save_data()
    
    # At the end of the training loop,
    if test != True:
        policy_model_file = os.path.join(logdir,"NN")
        save_itr_info(f"{policy_model_file}-{itr}.txt", itr, av_reward)
        save_variables(agent.net, policy_model_file)
        
    

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
    parser.add_argument('--knn', type=int, default = -1)
    parser.add_argument('--obj', choices=['peak','avg'], default='peak')
    parser.add_argument('--x_dim', type=int, nargs='+', default=[150, 150, 10])
    parser.add_argument('--age_max', type=float, default=3)
    parser.add_argument('--w_comm', type=float, default=1)
    parser.add_argument('--w_rad', type=float, default=1)
    parser.add_argument('--ang_range', type=float, default=45)
    parser.add_argument('--ob_time', action='store_true')
    parser.add_argument('--radar_interval', '-r_int', type=int, default=4)
    parser.add_argument('--interference', action='store_true')
    parser.add_argument('--OFDM_comm', action='store_true')
    
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
    
    # PPO parameters
    parser.add_argument('--ppo_epochs', type=int, default=10)
    parser.add_argument('--minibatch_size', '-mbs', type=int, default=64)
    parser.add_argument('--ppo_clip', type=float, default=0.2) #0.05
    parser.add_argument('--v_coef', type=float, default=0.5)
    parser.add_argument('--entrop_loss_coef', type=float, default=0) #0.001) #0.0005
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    
    # NN architecture parameters
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--size', '-s', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--size_critic', '-sc', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--input_prev_ac','-ipa', action='store_true')
    parser.add_argument('--ac_dist', default='Gaussian')
    parser.add_argument('--recurrent', '-r', action='store_true')
    parser.add_argument('--detach', type=int, default=8)
    parser.add_argument('--ic3net', action='store_true')
    
    # GNN parameters
    parser.add_argument('--gnn', type=int, default = 1)
    parser.add_argument('--gnn_pooling', choices=['gcn','transformer'], default='gcn')
    
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
    if args.knn == -1:
        args.knn = args.num_users
    
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
                              'interference': args.interference,
                              'OFDM_comm': args.OFDM_comm,
                              'test': args.test,
                              }
                
                logdir_w_params = logdir + '_{}users_wcomm{}_w_rad{}_maxage_{}_ang{}_obtime{}'.format(num_users,args.w_comm, args.w_rad,args.age_max,args.ang_range,args.ob_time)
                
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
                        gnn=args.gnn,
                        gnn_pooling=args.gnn_pooling,
                        gnn_knn = args.knn,
                        ic3net = args.ic3net,
                        size_critic=args.size_critic,
                        recurrent = args.recurrent,
                        detach_gap = args.detach,
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