# -*- coding: utf-8 -*-
"""

Dual carriageway JRC game environment.

J. Lee, Y. Cheng, D. Niyato, Y. L. Guan and G. David González,
"Deep Reinforcement Learning for Time Allocation and Directional Transmission in Joint Radar-Communication,"
2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2559-2564, doi: 10.1109/WCNC51071.2022.9771580.


"""

import gym
import numpy as np
from scipy.stats import norm
from gym import spaces
from gym.utils import seeding


def SINR(P_signal, noise, P_interferences):
    sum_P_interference = np.sum(P_interferences)
    ans = P_signal / (np.power(noise, 2) + sum_P_interference + 1e-7)
    
    return ans

def power_received(P_transmit, R, wavelength=1, gain=1):
    P_received = (P_transmit * (gain**2) * (wavelength**2)) / (np.power(4*np.pi, 2) * np.power(R, 2) + 1e-7)
    
    return P_received

def success_rate(SINR):
    BER = 1- norm.cdf(np.sqrt(2*SINR))     # Bit Error Rate
    return (1 - BER) * (SINR > 0)

def vector_angle(vector1, vector2):
    ''' returns angle between vector1 and vector2 in degrees
    vector 1: vector from ego vehicle
    vector 2: comparison vectors from other N-1 vehicles
    '''
    unit_vector1 = vector1 / (np.linalg.norm(vector1) + 1e-7)
    unit_vector2 = vector2 / (np.linalg.norm(vector2, axis=1) + 1e-7)[:,None]
    cos = np.dot(unit_vector2, unit_vector1)        # commutative: a.b = b.a
    sin = np.cross(unit_vector2, unit_vector1)      # anti-clockwise angle from vector 2 to vector 1
    angle = np.arctan2(sin,cos)
    
    return (angle / np.pi * 180)

def rotation(vector, direction):
    '''
    Input: vector to be rotated, (direction X pi) to be rotated by
    

    Parameters
    ----------
    vector : TYPE
        DESCRIPTION.
    direction : TYPE
        DESCRIPTION.

    Returns
    -------
    rotated_vector : TYPE
        DESCRIPTION.

    '''
    theta = direction * (np.pi/2)
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([[cos, -sin], [sin, cos]])     # rotation matrix
    rotated_vector = np.matmul(R, vector)
    return rotated_vector

class Beamform_JRC(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    
    def __init__(self, env_config):
        self.seed(123)
        self.viewer = None
        
        self.timestep = 0.5     # 0.5 seconds per time step
        self.N = env_config['num_users']      # num users
        self.N_obs = env_config['num_users_NN']       # num users observed by NN
        self.N_RSU = 8  # num roadside units
        self.v = 15     # 14 m/s
        self.v_diff = 7.5
        self.noise = 0.0007     # standard variation of noise (sigma)
        self.ob_time = env_config['ob_time']
        
        ''' env dimensions '''
        self.max_x_dim = env_config['x_dim']    # length of road
        self.min_x = 0
        self.min_x_gap = 5
        self.lane_width = 4
        self.num_lanes = env_config['num_lanes']  # number of lanes per direction
        self.max_y_dim = (self.lane_width * self.num_lanes) * 2    # width of env
        
        self.x_width = 7.5    # metres per unit for data grid
        self.y_width = 4 # 2
        
        self.w_comm = env_config['w_comm']
        self.w_rad = env_config['w_rad']
        
        self.veh_enter_rate = (self.N * self.v) / self.max_x_dim
        self.new_veh_queue = 0
        
        
        '''
        new data legend:
            1 - front
            2 - left
            3 - back
            4 - right

        '''
        self.received_mask = np.array([[2,2,2,2,2,2,2],
                                       [2,2,2,2,2,2,2],
                                       [3,3,3,1,1,1,1],
                                       [4,4,4,4,4,4,4],
                                       [4,4,4,4,4,4,4]])
        self.new_data_mask = np.array([[0,0,0,2,0,0,0],
                                       [0,0,0,2,0,0,0],
                                       [0,0,3,0,1,1,0],
                                       [0,0,0,4,0,0,0],
                                       [0,0,0,4,0,0,0]])
        self.new_data_mask_flipped = np.flip(self.new_data_mask, axis=(0,1))
        self.new_data_vector = np.array([[2],[2],[3],[4],[4]])
        self.new_data_vector_flipped = np.flip(self.new_data_vector, axis=(0,1))
        self.weight_importance = np.array([[0,0,0,1,1,1,1],
                                           [0,0,0,1,1,1,1],
                                           [0,0,1,0,1,1,1],
                                           [0,0,0,1,1,1,1],
                                           [0,0,0,1,1,1,1]])
        self.age_max = env_config['age_max']
        
        
        ''' action space '''
        self.radar_directions_n = 4
        self.data_directions_n = 4
        # communication direction X sensor direction + 1 null action + 1 radar action
        self.action_space = spaces.Discrete(self.radar_directions_n * self.data_directions_n + 2)
        
        self.radar_angular_range = env_config['radar_angular_range']   # beam 40 X 2 degrees wide
        self.radar_range = 100          # 100 m range
        
        ''' observation space '''
        self.data_map_shape = self.new_data_mask.shape
        
        if self.ob_time:
            self.high = np.concatenate((np.ones((1,))*(self.N*4), # num vehicles * radar interval
                                        np.ones((1,)),               # in map
                                        np.ones((1,)) * self.v,               # velocity
                                        self.max_x_dim * np.ones((2*(self.N_obs-1), )),
                                        self.v * np.ones((2*(self.N_obs-1), )),
                                        (self.age_max + 1) * np.ones( ((2 * self.data_map_shape[0] * self.data_map_shape[1]), ))
                                        ), axis=0)
        else:
            self.high = np.concatenate((np.ones((1,)),
                                        np.ones((1,)) * self.v,               # velocity
                                        self.max_x_dim * np.ones((2*(self.N_obs-1), )),
                                        self.v * np.ones((2*(self.N_obs-1), )),
                                        (self.age_max + 1) * np.ones( ((2 * self.data_map_shape[0] * self.data_map_shape[1]), ))
                                        ), axis=0)
        self.observation_space = spaces.Box(low=0, high=self.high, shape = (self.high.shape[0],))
        
        
    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]
    
    def state_transition(self, state, f_sent, f_num_transmits):
        
        state = {}
        
        # update position
        for n in range(self.N):
            # If vehicle n is in the map
            
            if (0 <= self.positions[n,0] <= self.max_x_dim):
                delta_position = self.velocities[(n)] * self.timestep
                self.positions[n] = self.positions[n] + (delta_position)
                if not (0 <= self.positions[n,0] <= self.max_x_dim):
                    self.remove_vehicle(n)
                
                # add received data
                #new_data = newer data + new data
                received_data = (self.data_age[(n+1)] > f_sent[n]) * (f_sent[n]>0) + (f_sent[n]>0) * (self.data_age[(n+1)] == 0)
                self.data_transmits[(n+1)] = (received_data * (f_num_transmits[n] + 1)) + (np.invert(received_data) * self.data_transmits[(n+1)])
                self.data_age[(n+1)] = (received_data * f_sent[n]) + (np.invert(received_data) * self.data_age[(n+1)])
                self.data[(n+1)] = (received_data * self.received_mask) + (np.invert(received_data) * self.data[(n+1)])
                
                # shift data according to new position
                shift_x = (delta_position[0] / self.x_width).astype(int)
                shift_y = (delta_position[1] / self.y_width).astype(int)
                
                if shift_x > 0:
                    self.data[(n+1)] = np.pad(self.data[(n+1)], ((0,0), (0, shift_x)), mode='constant')[:,shift_x:]
                    self.data_age[(n+1)] = np.pad(self.data_age[(n+1)], ((0,0), (0, shift_x)), mode='constant')[:,shift_x:]
                    self.data_transmits[(n+1)] = np.pad(self.data_transmits[(n+1)], ((0,0), (0, shift_x)), mode='constant')[:,shift_x:]
                    
                    # update with new data
                    new_data = np.tile(self.new_data_vector, shift_x-1)
                    new_data = np.insert(self.new_data_mask, [self.new_data_mask.shape[0]//2 + 1] * shift_x, new_data, axis=1)[:,-self.new_data_mask.shape[1]:]
                
                if shift_x < 0:
                    self.data[(n+1)] = np.pad(self.data[(n+1)], ((0,0), (abs(shift_x), 0)), mode='constant')[:,:shift_x]
                    self.data_age[(n+1)] = np.pad(self.data_age[(n+1)], ((0,0), (abs(shift_x), 0)), mode='constant')[:,:shift_x]
                    
                    # update with new data
                    new_data = np.tile(self.new_data_vector_flipped, abs(shift_x)-1)
                    new_data = np.insert(self.new_data_mask_flipped, [self.new_data_mask.shape[0]//2 + 2] * shift_x, new_data, axis=1)[:,-self.new_data_mask.shape[1]:]
                
                # increment data age
                self.data_age[(n+1)] = self.data_age[(n+1)] + (self.data_age[(n+1)] > 0)
                self.data[(n+1)][self.data_age[(n+1)] > self.age_max] = 0
                self.data_age[(n+1)][self.data_age[(n+1)] > self.age_max] = 0
                # add new data
                self.data[(n+1)] = new_data + (self.data[(n+1)] * (new_data == 0))
                self.data_age[(n+1)] = (self.data_age[(n+1)] * (new_data == 0)) + (new_data > 0)
                self.data_transmits[(n+1)] = (self.data_transmits[(n+1)] * (new_data == 0))
                
            else:
                self.remove_vehicle(n)
            
            state[(n+1)] = self._get_obs(n)
            
        idx_out = np.where(self.in_map == 0)[0]
        self.new_veh_queue += np.random.poisson(self.veh_enter_rate)
        
        if (len(idx_out) > 0) & (self.new_veh_queue > 0) :
            new_idx = np.random.choice(idx_out)
            self.reset_vehicle(new_idx)
            self.new_veh_queue -= 1
            state[new_idx+1] = self._get_obs(new_idx)
        
        return state
        
        
    def get_reward(self, state, action):
        relative_distance = np.zeros((self.N, self.N))
        relative_positions = np.zeros((self.N, self.N, 2))
        targeted = np.zeros((self.N, self.N))
        targeted_front = np.zeros((self.N, self.N))
        targeted_back = np.zeros((self.N, self.N))
        targeted_left = np.zeros((self.N, self.N))
        targeted_right = np.zeros((self.N, self.N))
        radar_targeted = np.zeros((self.N, self.N))
        noise_power = np.zeros((self.N, self.N))
        f_sends = np.zeros((self.N, self.N, self.new_data_mask.shape[0], self.new_data_mask.shape[1]))
        r_rad = np.zeros((self.N,))
        f_num_transmits = np.zeros((self.N, self.new_data_mask.shape[0], self.new_data_mask.shape[1]))
        
        # find vehicles targeted
        for i in range(self.N):
            if self.in_map[i] == 0:
                r_rad[i] = 0
                continue
            
            if action[(i+1)] == (self.action_space.n - 2):
                ac = 'null'
                self.beam_ac[i] = 4
            elif action[(i+1)] == (self.action_space.n - 1):
                ac = 'radar'
                radar_targeted[i] = np.ones((1,self.N))
                radar_targeted[i][i] = 0
            else:
                ac = np.unravel_index(action[(i+1)], (self.radar_directions_n, self.data_directions_n))
                self.beam_ac[i] = ac[0]
            
            r_rad[i] = np.exp(np.abs(self.velocities[i,0]*.05)) if ac != 'radar' else 0
            
            # Euclidean distance between ego vehicle and other vehicles
            relative_distance[i] = np.linalg.norm(self.positions - self.positions[i], axis=1)
            relative_positions[i] = self.positions - self.positions[i]
            
            if (ac != 'null') and (ac != 'radar'):
                ac_vector = rotation(self.velocities[i], ac[0])
                f_send = (self.data[(i+1)] == (ac[1] + 1)) * np.clip((-self.data_age[(i+1)] + self.age_max), 0, self.age_max)
                f_num_transmits[i] = (self.data[(i+1)] == (ac[1] + 1)) * self.data_transmits[(i+1)]
                

                targeted[i] = (abs(vector_angle(ac_vector, relative_positions[i])) < self.radar_angular_range) * self.in_map
                targeted[i][i] = 0
                
                incident_angle = vector_angle(ac_vector, self.velocities)             # angle between beam and velocities
                targeted_front[i] = (abs(incident_angle) < self.radar_angular_range) * targeted[i]
                targeted_back[i] = (abs(incident_angle) > (90+self.radar_angular_range)) * targeted[i]
                targeted_left[i] = ((incident_angle >= 45) & (incident_angle <= (90+self.radar_angular_range))) * targeted[i]
                targeted_right[i] = ((incident_angle <= -45) & (incident_angle >= -(90+self.radar_angular_range))) * targeted[i]
                
                # number of units in x dimension each other vehicle is away from ego vehicle n
                x_unit_distance = np.round(relative_positions[i,:,0] / self.x_width).astype(int)
                y_unit_distance = np.round(relative_positions[i,:,1] / self.y_width).astype(int)
                
                for j in range(self.N):
                    if targeted[i][j] > 0:
                        if x_unit_distance[j] > 0:
                            f_sends[i][j] = np.pad(f_send, ((0,0), (x_unit_distance[j], 0)), mode='constant')[:, 0:-x_unit_distance[j]]
                        else:
                            shift = np.abs(x_unit_distance[j])
                            f_sends[i][j] = np.pad(f_send, ((0,0), (0, shift)), mode='constant')[:, shift:]
                        if y_unit_distance[j] > 0:
                            f_sends[i][j] = np.pad(f_sends[i][j], ((y_unit_distance[j],0), (0, 0)), mode='constant')[0:-y_unit_distance[j], :]
                        else:
                            shift = np.abs(y_unit_distance[j])
                            f_sends[i][j] = np.pad(f_sends[i][j], ((0, shift), (0, 0)), mode='constant')[shift:, :]
        
        power_received_front = power_received(targeted_front, relative_distance)
        power_received_back = power_received(targeted_back, relative_distance)
        power_received_left = power_received(targeted_left, relative_distance)
        power_received_right = power_received(targeted_right, relative_distance)
        
        SINR_front = power_received_front / (noise_power +1e-7)
        SINR_back = power_received_back / (noise_power + 1e-7)
        SINR_left = power_received_left / (noise_power + 1e-7)
        SINR_right = power_received_right / (noise_power + 1e-7)
        
        success_rate_front = success_rate(SINR_front)
        success_rate_back = success_rate(SINR_back)
        success_rate_left = success_rate(SINR_left)
        success_rate_right = success_rate(SINR_right)
        combined_success_rate = success_rate_front + success_rate_back + success_rate_left + success_rate_right
        
        # Log SINR and throughput
        SINR_combined = np.sum(SINR_front + SINR_back + SINR_left + SINR_right, axis=0)
        self.episode_observation['SINR1'] += SINR_combined[0]
        self.episode_observation['SINR_av'] += np.average(SINR_combined)
        
        throughput = np.sum(np.sum(f_sends, axis=(2,3)) * combined_success_rate, axis=1)
        self.episode_observation['throughput1'] += throughput[0]
        self.episode_observation['throughput_av'] += np.average(throughput)
        self.episode_observation['num_transmits1'] += np.sum(f_num_transmits[0])
        self.episode_observation['num_transmits_av'] += np.sum(f_num_transmits) / self.N
        
        # Data transmitted
        f_sent = np.sum(f_sends, axis=(0))
        
        # Compute rewards
        r_comm = np.sum(np.sum(self.weight_importance * f_sends, axis=(2,3)) * combined_success_rate, axis=1)
        
        r = (self.w_comm * r_comm) - (self.w_rad * r_rad)
        
        # Log rewards
        self.episode_observation['r_comm1'] += (r_comm[0])
        self.episode_observation['r_rad1'] += (r_rad[0])
        
        return r, f_sent, f_num_transmits
    
    def heuristic_nn_action(self, n):
        if self.in_map[n] == True:
            ac = np.zeros((2,),dtype=int)
            targeted = np.zeros((self.N,))
            relative_distance = np.linalg.norm(self.positions - self.positions[n], axis=1)
            relative_positions = (self.positions - self.positions[n]) / np.array([self.max_x_dim, self.max_y_dim])
            
            # if other vehicle is in exactly the same position, may be targeted in multiple directions. update targeted only if not previously targeted.
            for d in range(4):
                ac_vector = rotation(self.velocities[n], d)
                targeted = targeted + (targeted==0)*(abs(vector_angle(ac_vector, relative_positions)) < self.radar_angular_range) * self.in_map * (d +1)
            
            mask = np.zeros((self.N,))
            mask[n] = True
            mask[targeted==0] = True    # mask vehicles if they are not targeted
            targeted_vehicle = np.argmin(np.ma.array(relative_distance, mask=mask))
            
            # If no vehicles are targeted, choose null action
            if np.sum(mask) == self.N:
                heuristic_ac = self.action_space.n - 1
            else:
                ac[0] = ac[1] = targeted[targeted_vehicle] - 1
                assert(ac[0] >= 0)
                assert(ac[0] < self.radar_directions_n)
                
                heuristic_ac = np.ravel_multi_index(ac, (self.radar_directions_n, self.data_directions_n))
        
        else:
            heuristic_ac = (self.action_space.n - 1)
        
        return heuristic_ac
        
    def step(self, action):
        r, f_sent, f_num_transmits = self.get_reward(self.state, action)
        r = dict(enumerate(r, 1))
        
        next_state = self.state.copy()
        next_state = self.state_transition(next_state, f_sent, f_num_transmits)
        self.state = next_state
        
        self.episode_observation['step_counter'] += 1
        if self.episode_observation['step_counter'] == 400:
            done = True
            print('End of episode')
        else:
            done = False
        
        return self.state, r, done, {}
        
    def reset(self):
        self.episode_observation = {
            'step_counter': 0,
            'throughput1': 0,
            'throughput_av': 0,
            'SINR1': 0,
            'SINR_av': 0,
            'r_comm1': 0,
            'r_rad1': 0,
            'num_transmits1': 0,
            'num_transmits_av': 0,
        }
        
        """ Generate RSU positions """
        x_pos_RSU = np.random.choice(self.max_x_dim , size=self.N_RSU, replace = False)
        self.RSU_positions = x_pos_RSU
        
        """ Generate vehicle positions 
        - position_idx generates places without replacement from dimensions (road length) x (num lanes)
        """
        position_idx = np.random.choice(self.max_x_dim * self.num_lanes, size=self.N, replace = False)
        x_position = position_idx - position_idx//self.max_x_dim * self.max_x_dim
        lane = (position_idx // self.max_x_dim)
        y_position = ((lane - self.num_lanes//2 + 0.5) * self.lane_width)
        lane_num = lane - self.num_lanes//2
        lane_num[lane_num<0] = lane_num[lane_num<0] + 1
        
        self.positions = np.stack((x_position, y_position)).transpose().astype('float32')
        self.velocities = np.zeros_like(self.positions)
        self.velocities[:,0] = self.v * np.sign(self.positions[:,1]) + lane_num * self.v_diff
        self.in_map = np.ones((self.N,))
        self.beam_ac = 5*np.ones((self.N,))
        
        self.data, self.data_age, state = {}, {}, {}
        self.data_transmits = {}
        for n in range(self.N):
            self.data[(n+1)] = (lane[n]==0) * self.new_data_mask_flipped + (lane[n]==1)*self.new_data_mask
            self.data_age[(n+1)] = (self.data[(n+1)] > 0) * 1
            self.data_transmits[(n+1)] = np.zeros_like(self.data[(n+1)])
            state[(n+1)] = self._get_obs(n)
        
        self.state = state.copy()
        return state
    
    def remove_vehicle(self, n):
        self.in_map[n] = 0
        self.data[(n+1)] = np.zeros_like(self.new_data_mask)
        self.data_age[(n+1)] = np.zeros_like(self.new_data_mask)
    
    def reset_vehicle(self, n):
        self.in_map[n] = 1
        lane = np.random.randint(self.num_lanes)
        lane_num = lane - self.num_lanes//2
        lane_num = lane_num + 1*(lane_num < 0)
        
        y_position = ((lane - self.num_lanes//2 + 0.5) * self.lane_width)
        x_position = (y_position < 0) *  self.max_x_dim
        self.positions[n] = np.array([x_position, y_position]).astype('float32')
        self.velocities[n,0] = self.v * np.sign(self.positions[n,1]) + lane_num * self.v_diff
        self.data[(n+1)] = (y_position < 0) * np.flip(self.new_data_mask, axis=(0,1)) + (y_position > 0)*self.new_data_mask
        self.data_age[(n+1)] = (self.data[(n+1)] > 0) * 1
        self.data_transmits[(n+1)] = np.zeros_like(self.data[(n+1)])
    
    def _get_obs(self, n):
        # normalise positions and velocities by range
        relative_positions = (np.delete(self.positions, n, axis=0) - self.positions[n]) / np.array([self.max_x_dim, self.max_y_dim])
        relative_velocities = (np.delete(self.velocities, n, axis=0) - self.velocities[n]) / self.v
        
        if self.N_obs < self.N:
            l2_dist = np.linalg.norm(relative_positions, axis=1)
            l2_dist_idx = l2_dist.argsort()
            relative_positions = relative_positions[l2_dist_idx]
            relative_velocities = relative_velocities[l2_dist_idx]
            relative_positions = relative_positions[0:self.N_obs-1]
            relative_velocities = relative_velocities[0:self.N_obs-1]
        elif self.N_obs > self.N:
            zeros = np.zeros((self.N_obs-self.N,2))
            relative_positions = np.concatenate((relative_positions, zeros))
            relative_velocities = np.concatenate((relative_velocities, zeros))
        
        # normalise data by num directions (i.e. 4)
        if self.ob_time:
            state = np.concatenate((np.array(self.episode_observation['step_counter'] % (4 * self.N)).reshape(-1),
                                    self.in_map[n].reshape(-1),
                                    self.velocities[n,0].reshape(-1),
                                    relative_positions.reshape(-1),
                                    relative_velocities.reshape(-1),
                                    (self.data[(n+1)] / 4).reshape(-1),
                                    (self.data_age[(n+1)] / self.age_max).reshape(-1)
                                    ))
        else:
            state = np.concatenate((self.in_map[n].reshape(-1),
                                    self.velocities[n,0].reshape(-1),
                                    relative_positions.reshape(-1),
                                    relative_velocities.reshape(-1),
                                    (self.data[(n+1)] / 4).reshape(-1),
                                    (self.data_age[(n+1)] / self.age_max).reshape(-1)
                                    ))
        state = np.divide(state, self.high)
        state = np.expand_dims(state, axis=0)
        return state
    
    def render(self, mode='human'):
        """
        
        Produces a rendering of the dual carriageway JRC game environment.

        """
        
        screen_width = 600
        screen_height = 400
        
        scale_x = screen_width/self.max_x_dim
        scale_y = screen_height/self.max_y_dim
        carwidth = 40
        carheight = 20
        
        # Geoemetry for beam triangle
        triangle_height = carheight
        triangle_halfwidth = triangle_height * np.tan(30/180*np.pi)
        v1 = np.array([carheight/2,0])
        v2 = v1 + np.array([triangle_height, -triangle_halfwidth])
        v3 = v1 + np.array([triangle_height, triangle_halfwidth,])
        vertices = np.transpose(np.array([v1,v2,v3]))
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            clearance = -carheight/2
            self.cartrans = {}
            
            for n in range(self.N):
                l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
                car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                if n == 0:
                    car.set_color(1,0,0)
                car.add_attr(rendering.Transform(translation=(0, clearance)))
                self.cartrans[n+1] = rendering.Transform()
                car.add_attr(self.cartrans[n+1])
                self.viewer.add_geom(car)
                frontwheel = rendering.make_circle(carheight / 2.5)
                frontwheel.set_color(.5, .5, .5)
                frontwheel.add_attr(
                    rendering.Transform(translation=(carwidth / 4, clearance))
                )
                frontwheel.add_attr(self.cartrans[n+1])
                self.viewer.add_geom(frontwheel)
                backwheel = rendering.make_circle(carheight / 2.5)
                backwheel.add_attr(
                    rendering.Transform(translation=(-carwidth / 4, clearance))
                )
                backwheel.add_attr(self.cartrans[n+1])
                backwheel.set_color(.5, .5, .5)
                self.viewer.add_geom(backwheel)
            
        for n in range(self.N):
            pos = self.positions[n]
            x = pos[0] * scale_x
            y = (pos[1] * scale_y) + (screen_height / 2)
            self.cartrans[n+1].set_translation(x, y)
            
            # draw beams
            if self.beam_ac[n] != 5:
                vertex_direction = vertices * np.sign(self.velocities[n,0])
                v1, v2, v3 = np.transpose(rotation(vertex_direction, self.beam_ac[n])) + np.array([x,y])
                self.viewer.draw_polygon([tuple(v1), tuple(v2), tuple(v3)], color=(0,0,1))
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
                
if __name__ == "__main__":
    
    env_config = {'num_users': 8,
                  'num_agents': 1,
                  'age_max': 3,
                  'x_dim': 150,
                  'num_lanes': 2,
                  }
    env = Beamform_JRC(env_config)
    env.reset()
    
    ac = {}
    while True:
        for n in range(env.N):
            ac[(n+1)] = np.random.randint(env.action_space.n)
        state, reward, done, _ = env.step(ac)