#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %pdb
"""

With this program, train multiple agents agent using Proximal Policy Optimisation (PPO) with a Graph Neural Network(GNN).
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

from beamformJRCenv_multi import Beamform_JRC
from utils import *


# In[]

# device = torch.device(0 if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(device)

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
        
class GCN(nn.Module):
    def __init__(self, ob_dim, policy_out_dim, policy_size, vf_size, discrete):
        super(GCN, self).__init__()
        self.out_dim = policy_out_dim
        self.discrete = discrete
        
        self.p_conv1 = GCNConv(ob_dim, policy_size[0])
        self.p_conv2 = GCNConv(policy_size[0], policy_size[1])
        self.p_out = GCNConv(policy_size[1], policy_out_dim)
        
        self.v_conv1 = GCNConv(ob_dim, vf_size[0])
        self.v_conv2 = GCNConv(vf_size[0], vf_size[1])
        self.v_out = GCNConv(vf_size[1], 1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        action_logits = self.p_conv1(x, edge_index)
        action_logits = F.relu(action_logits)
        action_logits = self.p_conv2(action_logits, edge_index)
        action_logits = F.relu(action_logits)
        action_logits = self.p_out(action_logits, edge_index)
        
        v, edge_index_v = data.x, data.edge_index
        v = self.v_conv1(v, edge_index_v)
        v = F.relu(v)
        v = self.v_conv2(v, edge_index_v)
        v = F.relu(v)
        v = self.v_out(v, edge_index_v)
        
        if self.discrete == False:
            mean, logstd = torch.split(action_logits, int(self.out_dim/2), dim = int(action_logits.dim()-1))
            dist = Normal(mean, logstd.exp())
        else:
            dist = Categorical(logits = action_logits)
        
        return dist, v

        
class GNN2(nn.Module):
    def __init__(self, ob_dim, policy_out_dim, policy_size, vf_size, discrete, recurrent, ic3net, pooling):
        super(GNN2, self).__init__()
        self.out_dim = policy_out_dim
        self.discrete = discrete
        self.recurrent = recurrent
        self.ic3net = ic3net
        
        self.encoder = nn.Linear(ob_dim, policy_size[0])
        if self.recurrent:
            self.p_lin1 = nn.GRUCell(policy_size[0], policy_size[1])
        else:
            self.p_lin1 = nn.Linear(policy_size[0], policy_size[1])
        if pooling == 'gcn':
            self.p_conv2 = GCNConv(policy_size[1], policy_size[1])
        elif pooling == 'transformer':
            self.p_conv2 = TransformerConv(policy_size[1], policy_size[1])
        self.p_out = nn.Linear(policy_size[1], policy_out_dim)
        if self.ic3net:
            self.p_comm_out = nn.Linear(policy_size[1], 2)
        
        self.v_out = nn.Linear(policy_size[1], 1)
        
    def forward(self, data, hidden=None):
        x, edge_index = data.x, data.edge_index
        
        features = self.encoder(x)
        if self.recurrent:
            features = self.p_lin1(features, hidden)
            next_hidden = features.clone()
        else:
            features = self.p_lin1(features)
            features = F.relu(features)
            next_hidden = None
        features = self.p_conv2(features, edge_index)
        features = F.relu(features)
        action_logits = self.p_out(features)
        
        if self.ic3net:
            comm_logits = self.p_comm_out(features)
            comm_dist = Categorical(logits=comm_logits)
        else:
            comm_dist = None
        
        v = self.v_out(features)
        
        if self.discrete == False:
            mean, logstd = torch.split(action_logits, int(self.out_dim/2), dim = int(action_logits.dim()-1))
            dist = Normal(mean, logstd.exp())
        else:
            dist = Categorical(logits = action_logits)
        
        
        return dist, v, next_hidden, comm_dist
        
# In[]

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.num_users = computation_graph_args['num_users']
        self.num_users_NN = computation_graph_args['num_users_NN']
        self.num_intelligent = computation_graph_args['num_intelligent']
        self.mode = computation_graph_args['mode']                       # policy of the unintelligent agents
        self.radar_interval = computation_graph_args['radar_interval']
        self.num_agents = computation_graph_args['num_agents']
        self.discrete = computation_graph_args['discrete']
        
        ''' NN architecture '''
        self.CNN = computation_graph_args['CNN']
        self.GNN = computation_graph_args['GNN']
        self.GNN_pooling = computation_graph_args['GNN_pooling']
        self.size = computation_graph_args['size']
        self.size_critic = computation_graph_args['size_critic']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.recurrent = computation_graph_args['recurrent']
        self.detach_gap = computation_graph_args['detach_gap']
        self.input_prev_ac = computation_graph_args['input_prev_ac']
        self.GNN_kNN = computation_graph_args['GNN_kNN']
        self.ic3net = computation_graph_args['ic3net']
        
        ''' learning parameters '''
        self.ppo_epochs = computation_graph_args['ppo_epochs']
        self.mb_size = computation_graph_args['minibatch_size']
        self.ppo_clip = computation_graph_args['ppo_clip']
        self.v_coef = computation_graph_args['v_coef']
        self.entrop_loss_coef_schedule = computation_graph_args['entrop_loss_coef']
        self.max_grad_norm = computation_graph_args['max_grad_norm']
        self.test = computation_graph_args['test']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']
        self.timesteps_this_batch = 0

        self.gamma = estimate_return_args['gamma']
        self.lamb = estimate_return_args['lambda']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.gae = estimate_return_args['gae']
        self.n_step = estimate_return_args['n_step']
        self.critic = estimate_return_args['critic']
        self.decentralised_critic = estimate_return_args['decentralised_critic']
        self.normalize_advantages = estimate_return_args['normalize_advantages']
        
        self.current_progress_remaining = 1.0
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(0 if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        policy_in_dim = int(self.ob_dim) # / self.num_users)
        policy_out_dim = int(self.ac_dim)
        if self.discrete == False:
            policy_out_dim = policy_out_dim * 2
        
        self.users_per_agent = int(self.num_users/self.num_agents)      # for parameter sharing
        
        self.edge_index = torch.empty((0,2), dtype=torch.long)
        for i in range(self.num_intelligent):
            for j in range(self.num_intelligent):
                if i != j:
                    self.edge_index = torch.cat((self.edge_index, torch.tensor([[i,j]], dtype=torch.long)))
        self.edge_index = self.edge_index.t()
        
        if self.GNN > 0:
            GNN_in = policy_in_dim #* self.num_intelligent
            GNN_out = policy_out_dim #* self.num_intelligent
            if self.GNN == 1:
                self.net = GNN(GNN_in, GNN_out, self.size, self.size_critic, discrete=self.discrete).to(self.device)
            elif self.GNN == 2:
                self.net = GNN2(GNN_in, GNN_out, self.size, self.size_critic, discrete=self.discrete, recurrent=self.recurrent, ic3net=self.ic3net, pooling=self.GNN_pooling).to(self.device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        
    def update_current_progress_remaining(self, num_iterations: int, total_iterations: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self.current_progress_remaining = 1.0 - float(num_iterations) / float(total_iterations)
        
    def update_entropy_loss(self):
        self.entrop_loss_coef = self.entrop_loss_coef_schedule(self.current_progress_remaining)
        
    """    
    Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)
            _uo - (num users, ob dim)
            _nuo- (batch size, num users, ob dim)
            
    """
    def act(self, ob, prev_comm_ac, hidden=None, steps=None):
        '''
        Parameters
        ----------
        ob : pytorch geometric Data(x=[8, 101], edge_index=[2, 28])
        prev_comm_ac : np array (8,1)
        steps : int

        Returns
        -------
        ac_np : np array (#users,)
        log_prob_ua : torch.Size([#users])
        prev_comm_ac : list len #users
        '''
        # prev_comm_ac = [0 for _ in range(self.num_users)]
        
        if self.GNN == 0:
            ob_uo = torch.tensor(ob[0 : self.num_intelligent],requires_grad=False,dtype=torch.float32, device=self.device) #.unsqueeze(0) 
        
        dist, value, next_hidden, comm_dist = self.net(ob, hidden)              # logits: torch.size([#agents, ac_dim]), value: torch.Size([#agents,1])
        
        ac_ua = dist.sample()                   # torch.size([#agents])
        ac_np = ac_ua.cpu().numpy()                   # (#agents,)
        log_prob_ua = dist.log_prob(ac_ua).detach()      # torch.size([#agents])
        
        if self.ic3net:
            msg_ac = comm_dist.sample()
            log_prob_msg = comm_dist.log_prob(msg_ac).detach()
        else:
            msg_ac = None
            log_prob_msg = None
        
        ''' for the remaining non-learning users '''
        for i in range(self.num_intelligent, self.num_users):
            if self.mode == 'unif_rand':
                np.append(ac_np, np.array(np.random.randint(int(self.ac_dim))))     # append to numpy array??
            elif self.mode == 'rotate':
                if steps == 0:
                    ac_np[i] = (steps + i) % int(self.ac_dim)
                    prev_comm_ac[i] = ac_np[i]
                elif ((steps + i) % self.radar_interval) == 0:        # choose radar every 4 steps
                    ac_np[i] = np.array(int(self.ac_dim) - 1)
                else:
                    if (prev_comm_ac[i] >= (int(self.ac_dim)-3)):
                        ac_np[i] = 0
                    else:
                        ac_np[i] = prev_comm_ac[i] + 1
                    prev_comm_ac[i] = ac_np[i]
        
        return ac_np, log_prob_ua, prev_comm_ac, next_hidden, msg_ac, log_prob_msg
    
    
    def eval_ac(self, ob, ac_ua, hidden=None, msg_ac=None):
        '''
        Parameters
        ----------
        ob : pytorch geometric 'Data' data type. num nodes=batch size * #agents
        ac_ua : numpy (batch size, #agents)

        Returns
        -------
        values : torch.Size([batch size, #agents])
        log_prob : torch.Size([batch size, #agents])
        entropy : torch.Size([])
        '''
        dist, values, next_hidden, comm_dist = self.net(ob, hidden)  # torch.Size([batch_size * #agents, ac_dim]), [batch_size * #agents, 1], [batch_size, hidden_dim], [batch, comm_dim]
        
        ac_ua2 = torch.tensor(ac_ua,requires_grad=False,dtype=torch.float32, device=self.device).reshape(-1)
        log_prob = dist.log_prob(ac_ua2)     # torch.Size([batch])
        entropy = dist.entropy().mean()         # torch.Size([])
        
        if self.ic3net:
            msg_ac = msg_ac.reshape(-1)
            log_prob_comm = comm_dist.log_prob(msg_ac).reshape((-1, self.num_users))
        else:
            log_prob_comm = None
        
        values = values.reshape((-1, self.num_users))
        log_prob = log_prob.reshape((-1, self.num_users))
        
        return values, log_prob, entropy, log_prob_comm
    
    
    def eval_target(self, next_ob, re_n, terminal_n, next_hidden=None):
        '''
        Parameters
        ----------
        next_ob : pytorch geometric 'Data' data type
        re_n : numpy (#users * batch size, 1)
        terminal_n : numpy (batch size, 1)

        Returns
        -------
        target_n : torch.Size([batch size, #users]), requires_grad = False

        '''
        terminal_n2 = torch.tensor(terminal_n,requires_grad=False,dtype=torch.float32, device=self.device).reshape((-1,1))
        terminal_n2 = torch.tile(terminal_n2, [self.num_users, 1])
        
        re_n2 = torch.tensor(re_n,requires_grad=False,dtype=torch.float32, device=self.device).reshape((-1,1))
        
        with torch.no_grad():
            _, next_v_n, _, _ = self.net(next_ob, next_hidden)
            target_n = (re_n2 + (1 - terminal_n2) * self.gamma * next_v_n).detach()
            target_n = target_n.reshape((-1, self.num_users))
        return target_n
    
    
    def sample_trajectories(self, itr, env):
        ''' Collect paths until we have enough timesteps '''
        self.timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            self.timesteps_this_batch += pathlength(path)
            if self.timesteps_this_batch >= self.min_timesteps_per_batch:
                break
        return paths, self.timesteps_this_batch
    
    
    def sample_trajectory(self, env, animate_this_episode):
        '''
        Sample a single episode in the Markov environment.
        
        Parameters
        ----------
        env : gym style environment
        animate_this_episode : bool

        Returns
        -------
        ob: list length 8, elements np.array (ob_dim,)
        ac: np.array (8,)
        rew: np.array (8,)
        log_prob, log_prob_msg: torch.Size([1,#agents])
        hidden, next_hidden: torch.Size([#agents, hidden_size])
        path : dictionary containing information on the current path/trajectory
        '''
        obs_data, acs, log_probs, rewards, next_obs_data, next_acs, entropys, terminals, values, next_values = \
        [], [], [], [], [], [], [], [], [], []
        hiddens, next_hiddens = [], []
        msg_acs, log_prob_msgs = [], []
        prev_comm_ac = np.zeros((self.num_users,1))
        steps = 0
        
        ob = env.reset()
        
        in_map = env.in_map.astype(bool)
        positions = env.positions.copy()    # set positions of agents not in map to high value (so not chosen as NN)
        positions[:,0] = env.positions[:,0] * in_map + 1e7 * np.invert(in_map)
        # For each element in y, find the k nearest points in x.
        edge_index = knn(x=torch.tensor(positions, device=self.device), y = torch.tensor(env.positions, device=self.device), k=self.GNN_kNN+1)
        msg_ac = torch.ones((self.num_users,), device=self.device) if self.ic3net else None
        edge_index = masking(edge_index, env.in_map, self.device, msg_ac)
        ob_data = Data(x = torch.tensor(ob, dtype=torch.float), edge_index = edge_index).to(self.device)  # x torch.Size([# agents, 1 step, ob_dim])
               
        hidden = torch.zeros(self.num_users, self.size[1], device=self.device) if self.recurrent else None     #agents, hidden_size
        
        ac, log_prob, prev_comm_ac, hidden, msg_ac, log_prob_msg = self.act(ob_data, prev_comm_ac, hidden, steps)
        
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
                if steps == 0:
                    input("Press any key to continue...")
            
            obs_data.append(ob_data)
            acs.append(ac)
            log_probs.append(log_prob.detach().reshape(-1,self.num_users))
            if self.recurrent:
                hiddens.append(hidden)
            if self.ic3net:
                msg_acs.append(msg_ac)
                log_prob_msgs.append(log_prob_msg.detach().reshape(-1,self.num_users))
            
            ob, rew, done, _ = env.step(ac)
            
            in_map = env.in_map.astype(bool)
            positions = env.positions.copy()    # set positions of agents not in map to high value (so not chosen as NN)
            positions[:,0] = env.positions[:,0] * in_map + 1e7 * np.invert(in_map)
            # For each element in y, find the k nearest points in x.
            edge_index = knn(x=torch.tensor(positions, device=self.device), y = torch.tensor(env.positions, device=self.device), k=self.GNN_kNN+1)
            edge_index = masking(edge_index, env.in_map, self.device, msg_ac)
            
            ob_data = Data(x = torch.tensor(ob, dtype=torch.float), edge_index = edge_index).to(self.device)   # x torch.Size([# agents, 1 step, ob_dim])
            
            # ac: (#agents,). log_prob: torch.Size([#agents]). prev_comm_ac: list size #agents
            ac, log_prob, prev_comm_ac, hidden, msg_ac, log_prob_msg = self.act(ob_data, prev_comm_ac, hidden, steps)
            
            if self.recurrent:
                next_hiddens.append(hidden.clone().detach())
                if ((steps+1) % self.detach_gap)==0:
                    hidden = hidden.detach()
            next_obs_data.append(ob_data)
            next_acs.append(ac)
            rewards.append(rew)     # most recent reward appended to END of list
            
            steps += 1
            if done or steps >= self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        
        path = {}
        path = {"observation" : obs_data,                   # list of length steps. (steps, ob_dim)
             "reward" : np.array(rewards, dtype=np.float32),} # (steps)}
        path["log_prob"] = torch.cat(log_probs, axis=0)     # torch.Size([#steps, #agents])
        path["next_observation"] = next_obs_data            # list of length steps
        path["next_action"] = np.array(next_acs, dtype=np.float32)  # np array (#steps, #agents)
        path["action"] = np.array(acs, dtype=np.float32)            # np array (#steps, #agents)
        path["terminal"] = np.array(terminals, dtype=np.float32)    # np array (#steps,)
        if self.recurrent:
            path["hidden"] = torch.stack(hiddens, axis=0)
            path["next_hidden"] = torch.stack(next_hiddens, axis=0)
        if self.ic3net:
            path["msg_ac"] = torch.stack(msg_acs, axis=0) 
            path["log_prob_msg"] = torch.cat(log_prob_msgs, axis=0)     # torch.Size([#steps, #agents])
        
        # Log additional statistics
        path['r_comm1'] = (env.episode_observation['r_comm1'])
        path['r_rad1'] = (env.episode_observation['r_rad1'])
        path['throughput1'] = (env.episode_observation['throughput1'] / 400)
        path['throughput_av'] = (env.episode_observation['throughput_av'] / 400)
        path['SINR1'] = (env.episode_observation['SINR1'] / 400)
        path['SINR_av'] = (env.episode_observation['SINR_av'] / 400)
        path['num_transmits1'] = (env.episode_observation['num_transmits1'] / 400)
        path['num_transmits_av'] = (env.episode_observation['num_transmits_av'] / 400)
        
        if self.test:
            path['lane_action_map'] = (env.episode_observation['lane_action_map'])
            path['comm_distance'] = (env.episode_observation['comm_distance'])
            path['all_distances'] = (env.episode_observation['all_distances'])
            path['radar_NN_distance'] = (env.episode_observation['radar_NN_distance'])
            path['NN_distance'] = (env.episode_observation['NN_distance'])
            path['radar_NN_position'] = (env.episode_observation['radar_NN_position'])
            path['NN_position'] = (env.episode_observation['NN_position'])
        
        return path
    
    def estimate_decen_adv(self, ob_no, next_ob_no, re_n, terminal_n, hidden=None, next_hidden=None):
        '''
        Parameters
        ----------
        ob_no : List of Pytorch Geometric Data() datatype
        next_ob_no : List of Pytorch Geometric Data() datatype
        re_n : np array (batch_size, num_users)
        terminal_n : np array (batch size,)

        Returns
        -------
        adv_n : np array (batch_size, num_users)
        '''
        lastgaelam = torch.zeros((1, self.num_users), device = self.device)
        re_n2 = torch.tensor(re_n, requires_grad=False,dtype=torch.float32, device=self.device)
        adv_n = torch.zeros_like(re_n2).to(self.device)
        
        for t in reversed(range(self.min_timesteps_per_batch)):
            if t == self.min_timesteps_per_batch-1:
                nextnonterminal = 1 - 1 #terminal_n[t]
            else:
                nextnonterminal = 1 - terminal_n[t+1]
                
            with torch.no_grad():
                if self.recurrent:
                    _, v_next, _, _ = self.net(next_ob_no[t], hidden[t])
                    _, v, _, _ = self.net(ob_no[t], next_hidden[t])
                else:
                    _, v_next, _, _ = self.net(next_ob_no[t])     # torch.Size([8,1])
                    _, v, _, _ = self.net(ob_no[t])
            q = re_n2[t].unsqueeze(-1) + self.gamma * v_next * nextnonterminal  # torch.Size([])
            delta = q - v
            adv_n[t] = lastgaelam = delta.t() + self.gamma*self.lamb*nextnonterminal*lastgaelam
        if self.normalize_advantages:
            adv_n = (adv_n - torch.mean(adv_n, axis=0)) / (torch.std(adv_n, axis=0) + 1e-7)
        
        return adv_n
    
    
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
                ratio = (log_prob_na - log_prob_na_old[idx].detach()).exp()
                obj = torch.mul(ratio, adv_n[idx])
                obj_clipped = torch.mul(torch.clamp(ratio, 1-self.ppo_clip, 1+self.ppo_clip), adv_n[idx])     # torch.size([batch])
                policy_loss = - torch.min(obj, obj_clipped).mean()
                
                # Calc value loss
                v_loss = (values - target_n[mb_idx]).pow(2).mean()
                
                loss = policy_loss + self.v_coef * v_loss - self.entrop_loss_coef * entropy
                
                # Logging
                losses = loss.item()
                pg_losses = policy_loss.item()
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.ppo_clip).float()).item()
                approx_kl_divs = torch.mean(log_prob_na_old[idx] - log_prob_na).item()
                value_loss = v_loss.item()
                entropy_loss = entropy.item()
                
                # Back propagate
                self.optimizer.zero_grad()
                loss.backward() #retain_graph = True)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Checking
                # if (epoch == 0) & (mb_idx == 0):
                #     logz.log_tabular("CriticLossBefore", v_loss.detach().cpu().numpy())
                #     logz.log_tabular("LossBefore", loss.detach().cpu().numpy())
                
                # if (epoch == (self.ppo_epochs-1)) & (mb_idx == 0):
                #     values_after, log_prob_after, entropy_after, log_prob_msg_after = self.eval_ac(ob_batch, ac_na[idx], hidden_mb, msg_ac_mb)
                    
                #     if self.ic3net:
                #         log_prob_after = log_prob_after + log_prob_msg_after    # multiply both action probabilities, assuming independence
                #     ratio = (log_prob_after - log_prob_na_old[idx]).exp().detach().requires_grad_(False)
                #     obj = torch.mul(ratio, adv_n[idx])
                #     obj_clipped = torch.mul(torch.clamp(ratio, 1-self.ppo_clip, 1+self.ppo_clip), adv_n[idx])     # torch.size([batch])
                #     policy_loss_after = - torch.min(obj, obj_clipped).mean()
                    
                #     # Calc value loss
                #     v_loss_after = (values_after - target_n[mb_idx]).pow(2).mean()
                    
                #     loss_after = policy_loss_after + self.v_coef * v_loss_after + self.entrop_loss_coef * entropy_after
                    
                #     logz.log_tabular("LossAfter", loss_after.detach().cpu().numpy())
                #     logz.log_tabular("CriticLossAfter", v_loss_after.detach().cpu().numpy())
                    
                mb_idx += 1
                    
        # Log
        logz.log_tabular("Loss ", np.mean(losses))
        logz.log_tabular("Policy Gradient Loss ", np.mean(pg_losses))
        logz.log_tabular("Clip Fraction ", np.mean(clip_fraction))
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
    
    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)
    
    # policy_net_dir = os.path.join('data',
    #                               'PPO_8NN_Attn_IC3_bs1200_n7000_beamform_JRC_multi_vi_03-02-2022_23-44-13_8users_wcomm1.0_w_rad6.0_maxage_8.0_ang45.0_obtimeTrue',
    #                               '1',
    #                               'NN_itr5925')
    
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
            all_distances = np.concatenate([path["all_distances"] for path in paths], axis=0)
            radar_NN_distance = np.concatenate([path["radar_NN_distance"] for path in paths], axis=0)
            NN_distance = np.concatenate([path["NN_distance"] for path in paths], axis=0)
            radar_NN_position = np.concatenate([path["radar_NN_position"] for path in paths], axis=0)
            NN_position = np.concatenate([path["NN_position"] for path in paths], axis=0)
        
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
            logz.log_tabular("All Distances", all_distances)
            logz.log_tabular('Radar NN Distance', radar_NN_distance)
            logz.log_tabular('NN Distance', NN_distance)
            logz.log_tabular('Radar NN Position', radar_NN_position)
            logz.log_tabular('NN Position', NN_position)
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
                              'test': args.test,
                              'OFDM_comm': args.OFDM_comm,
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