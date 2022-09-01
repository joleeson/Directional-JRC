# %pdb on

"""

Run an agent that uses the "Heuristic" or "Round Robin" policies. Running instructions in the README file.

J. Lee, Y. Cheng, D. Niyato, Y. L. Guan and G. David González,
"Deep Reinforcement Learning for Time Allocation and Directional Transmission in Joint Radar-Communication,"
2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2559-2564, doi: 10.1109/WCNC51071.2022.9771580.


"""

from __future__ import division

# from beamformJRCenv_1lane import Beamform_JRC
from beamformJRCenv_vd import Beamform_JRC
from gym.wrappers import Monitor
import numpy as np
import numpy.random as random
import random as python_random
# import json
import time
import os
import argparse
import logz
import inspect


def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(test)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

def alternative_switch_action(t, num_actions):
    """
    Alternates between communication '0' and a choice of communications actions.
    Cycles between the communication actions

    Parameters
    ----------
    t : time step
    num_actions : Tnumber of communication actions available

    Returns
    -------
    action num

    """
    if t % 2 == 1:
        return 0
    else:
        r = t % (num_actions*2)
        return int(r/2 + 1)

def alt_switch_action5(t, comm_action):
    """
    Alternates between communication '0' and communicating packets with urgency level 'comm_action'.

    Parameters
    ----------
    t : time step
    num_actions : Tnumber of communication actions available

    Returns
    -------
    action num

    """
    if t % 2 == 1:
        return 0
    else:
        return comm_action

class Agent(object):
    def __init__(self, policy_config, sample_trajectory_args, env_args):
        
        self.num_users = env_args['num_users']
        
        self.mode = policy_config['mode']
        self.ac_dim = policy_config['ac_dim']
        self.CW_min = policy_config['CW'][0]
        self.CW_max = policy_config['CW'][1]
        self.CW = np.ones((self.num_users,)) * self.CW_min
        self.counter = np.zeros((self.num_users,))
        
        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']
        
        self.timesteps_this_batch = 0
        
        self.counter = np.random.randint(self.CW, size=self.num_users)
    
    
    def act(self, ob):
        # if counter is zero select actions
        actions = (self.counter==0) * np.random.randint(self.ac_dim)
        action_reqs = actions
        
        priorities = np.empty((0),dtype=int)
        for n in range(self.num_users):
            priorities = np.concatenate((priorities, ob[(n+1)][state_space_size['data_size']*2].reshape(1)))
        
        if np.sum(actions>0) > 1:
            actions = (priorities == 1) * action_reqs   # Choose agent that transmits based on priority
        unsuccessful_ac_reqs = (actions==0) * (action_reqs!=0)         # agents that act but unsuccessful
        
        # Halve CW for successful transmission, double for unsuccessful transmission
        self.CW = np.clip(((actions>0) * self.CW / 2) + ((actions==0) * self.CW), 2, self.CW_max)
        self.CW = np.clip((unsuccessful_ac_reqs * self.CW * 2) + (unsuccessful_ac_reqs==0 * self.CW), 2, self.CW_max)
        
        # decrement counter
        self.counter = np.clip(self.counter - 1, a_min=0, a_max=self.CW_max)
        # reset counter for agents that attempted to take action
        self.counter = ((action_reqs>0) * np.random.randint(self.CW, size=self.num_users)) + ((action_reqs==0) * self.counter)
        
        ac = {}
        for n in range(self.num_users):
            ac[(n+1)] = actions[n]
            
        return ac
    
    
    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        self.timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            self.timesteps_this_batch += pathlength(path[1])
            if self.timesteps_this_batch >= self.min_timesteps_per_batch:
                break
        return paths, self.timesteps_this_batch
    
    
    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()                    # returns ob['agent_no'] = 
        obs, acs, log_probs, rewards, next_obs, next_acs, hiddens, entropys = {}, {}, {}, {}, {}, {}, {}, {}
        prev_acs = {}
        terminals = []
        for i in range(self.num_users):
            obs[(i+1)], acs[(i+1)], log_probs[(i+1)], rewards[(i+1)], next_obs[(i+1)], next_acs[(i+1)], hiddens[(i+1)], entropys[(i+1)] = \
            [], [], [], [], [], [], [], []
        
        steps = 0
        
        for i in range(self.num_users):
            if self.mode == 'unif_rand':
                acs[(i+1)] = np.array(np.random.randint(env.action_space.n))
            elif self.mode == 'rotate':
                acs[(i+1)] = np.array((steps + i) % env.action_space.n)
            elif self.mode == 'urg5':
                acs[(i+1)] = alt_switch_action5(steps, 5)
            elif self.mode == 'heuristic':
                if i==0:
                    acs[(i+1)] = np.array(env.heuristic_nn_action(i))
                else:
                    acs[(i+1)] = np.array((steps + i) % env.action_space.n)
        if self.mode == 'csma-ca':
            acs = self.act(ob)

        
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            ob, rew, done, _ = env.step(acs)
            
            for i in range(self.num_users):
                rewards[(i+1)].append(rew[(i+1)])     # most recent reward appended to END of list
                
                if self.mode == 'unif_rand':
                    acs[(i+1)] = np.array(np.random.randint(env.action_space.n))
                elif self.mode == 'rotate':
                    acs[(i+1)] = np.array((steps + i) % env.action_space.n)
                elif self.mode == 'urg5':
                    acs[(i+1)] = alt_switch_action5(steps, 5)
                elif self.mode == 'heuristic':
                    if i==0:
                        acs[(i+1)] = np.array(env.heuristic_nn_action(i))
                    else:
                        acs[(i+1)] = np.array((steps + i) % env.action_space.n)
            if self.mode == 'csma-ca':
                acs = self.act(ob)
            
            steps += 1
            if done or steps >= self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        
        path = {}
        for i in range(self.num_users):
            path[(i+1)] = {"reward" : np.array(rewards[(i+1)], dtype=np.float32),                  # (steps)
                 }
            path[(i+1)]["action"] = np.array(acs[(i+1)], dtype=np.float32)
        
        path["terminal"] = np.array(terminals, dtype=np.float32)
        
        # Log additional statistics
        path['r_comm1'] = (env.episode_observation['r_comm1'])
        path['r_rad1'] = (env.episode_observation['r_rad1'])
        path['throughput1'] = (env.episode_observation['throughput1'] / 400)
        path['throughput_av'] = (env.episode_observation['throughput_av'] / 400)
        path['SINR1'] = (env.episode_observation['SINR1'] / 400)
        path['SINR_av'] = (env.episode_observation['SINR_av'] / 400)
        path['num_transmits1'] = (env.episode_observation['num_transmits1'] / 400)
        path['num_transmits_av'] = (env.episode_observation['num_transmits_av'] / 400)
        
        return path


def test(
        exp_name,
        env_name,
        env_config,
        n_iter, 
        min_timesteps_per_batch, 
        max_path_length,
        animate,
        seed,
        mode,
        CW,
        logdir,
        ):
    
    start = time.time()
    setup_logger(logdir, locals())  # setup logger for results    
    
    env = Beamform_JRC(env_config)
    # if animate == True:
    #     env = Monitor(env, './video', force=True)
    env.seed(seed)
    num_users = int(env.N)
    
    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps
    policy_config = {'mode': args.mode,
                     'CW': CW,
                     'ac_dim': env.action_space.n,
                         }
    env_args = {'num_users': env_config['num_users']}
    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }
    agent = Agent(policy_config, sample_trajectory_args, env_args)
    
    total_timesteps = 0
    
    for itr in range(n_iter):
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch
    
        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no, ac_na, re_n, log_prob_na, next_ob_no, next_ac_na, h_ns1, entropy = {}, {}, {}, {}, {}, {}, {}, {}
        returns = np.zeros((num_users,len(paths)))
        for i in range(num_users):
            re_n[(i+1)] = np.concatenate([path[(i+1)]["reward"] for path in paths])               # (batch_size, num_users)
            returns[i,:] = [path[(i+1)]["reward"].sum(dtype=np.float32) for path in paths]   # (num_users, num episodes in batch)
            assert re_n[(i+1)].shape == (timesteps_this_batch,)
            assert returns[i,:].shape == (timesteps_this_batch/400,)
                
        terminal_n = np.concatenate([path["terminal"] for path in paths])       # (batch_size,)
        
        # Log additional statistics
        r_comm1 = ([path["r_comm1"] for path in paths])  
        r_rad1 = ([path["r_rad1"] for path in paths])   
        throughput1 = ([path["throughput1"] for path in paths])                      # (batch,)
        throughput_av = ([path["throughput_av"] for path in paths])
        SINR1 = ([path["SINR1"] for path in paths])                      # (batch,)
        SINR_av = ([path["SINR_av"] for path in paths])
        num_transmits1 = ([path["num_transmits1"] for path in paths])
        num_transmits_av = ([path["num_transmits_av"] for path in paths])
        
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("Average Reward", np.mean(returns))   # per agent per episode
        logz.log_tabular("StdReward", np.std(returns))
        logz.log_tabular("MaxReward", np.max(returns))
        logz.log_tabular("MinReward", np.min(returns))
        # logz.log_tabular("throughput1", np.mean(throughput1))
        # logz.log_tabular("throughput_av", np.mean(throughput_av))
        logz.log_tabular("r_comm1", np.mean(r_comm1))
        logz.log_tabular("r_rad1", np.mean(r_rad1))
        logz.log_tabular("Throughput1", np.mean(throughput1))
        logz.log_tabular("Average Throughput", np.mean(throughput_av))
        logz.log_tabular("SINR1", np.mean(SINR1))
        logz.log_tabular("SINR_av", np.mean(SINR_av))
        logz.log_tabular("Num Transmits 1", np.mean(num_transmits1))
        logz.log_tabular("Average Num Transmits", np.mean(num_transmits_av))
        
        for i in range(env_config['num_users']):
            logz.log_tabular("Reward"+str(i+1), np.mean(returns, axis=1)[i])
            logz.log_tabular("StdReward"+str(i+1), np.std(returns, axis=1)[i])
            logz.log_tabular("MaxReward"+str(i+1), np.max(returns, axis=1)[i])
            logz.log_tabular("MinReward"+str(i+1), np.min(returns, axis=1)[i])
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        
        logz.dump_tabular(step=itr)

parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str, default='beamform_JRC-v0')
parser.add_argument('--num_lanes', type=int, default=2)
parser.add_argument('--num_users_NN', type=int)
parser.add_argument('--num_users', type=int, nargs='+', default = [0])
parser.add_argument('--num_agents', type=int)
parser.add_argument('--x_dim', type=int, nargs='+', default=[150, 150, 10])
parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
parser.add_argument('--obj', choices=['peak','avg'], default='avg')
parser.add_argument('--age_max', type=float, default=3)
parser.add_argument('--w_comm', type=float, default=1)
parser.add_argument('--w_rad', type=float, default=1)
parser.add_argument('--ang_range', type=float, default=45)

# Algorithm hyperparameters
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_experiments', '-e', type=int, default=1)
parser.add_argument('--mode', choices=['heuristic','urg5','best','csma-ca','rotate'], default='rotate')
parser.add_argument('--CW', type=int, nargs='+', default=[2,16])

parser.add_argument('--exp_name', type=str, default='vpg')
parser.add_argument('--render', action='store_true')

parser.add_argument('--n_iter', '-n', type=int, default=100)
parser.add_argument('--batch_size', '-b', type=int, default=1000)
args = parser.parse_args()


logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logdir = os.path.join('data', logdir)

print("------")
print(logdir)
print("------")

max_path_length = args.ep_len if args.ep_len > 0 else None

if args.num_users == [0]:
        args.num_users[0] = args.num_users_NN

for num_users in args.num_users:
    for x_dim in range(args.x_dim[0], args.x_dim[1], args.x_dim[2]):
                
        for e in range(args.n_experiments):
            """ Set random seeds 
            https://keras.io/getting_started/faq/
            """
            seed = args.seed + e*10
            # The below is necessary for starting Numpy and Python generated random numbers in a well-defined initial state.
            np.random.seed(seed)
            python_random.seed(seed)
            
            env_config = {'num_users': num_users,
                          'num_users_NN': args.num_users_NN,
                          'num_agents': args.num_agents,
                          'age_max': args.age_max,
                          'x_dim': x_dim,
                          'num_lanes': args.num_lanes,
                          'w_comm': args.w_comm,
                          'w_rad': args.w_rad,
                          'radar_angular_range': args.ang_range,
                          }
            
            logdir_w_params = logdir + "_{}usrs_{}_wcomm1{}_w_rad{}_maxage_{}_ang{}".format(num_users,args.mode,args.w_comm, args.w_rad, args.age_max,args.ang_range)
            
            test(
                exp_name = args.exp_name,
                env_name = args.env_name,
                env_config = env_config,
                n_iter = args.n_iter, 
                min_timesteps_per_batch = args.batch_size, 
                max_path_length = max_path_length,
                animate = args.render,
                seed = args.seed,
                mode = args.mode,
                CW = args.CW,
                logdir = os.path.join(logdir_w_params,'%d'%seed),
                )
                        