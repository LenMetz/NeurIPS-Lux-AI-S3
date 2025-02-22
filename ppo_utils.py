import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from IPython.display import display, Javascript
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
import time
from maps import *
from astar import *
from agent import Agent
from my_agent.lux.utils import direction_to, direction_to_change
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete, Tuple, Box


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_ens = env.observation_space[1].shape[0]
        self.n_maps = len(env.single_observation_space[0])
        self.n_state_params = env.single_observation_space[1].shape[0]
        self.n_action = env.single_action_space.shape[0]
        self.action_dim = env.single_action_space.shape[-1]
        self.n_unit_states = env.single_observation_space[2].shape[1]
        self.transformer_embedding_dim = env.get_attr("transformer_embedding_dim")[0]
        self.state_param_embedding_dim = env.get_attr("state_param_embedding_dim")[0]
        
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(self.n_maps, 8, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.AvgPool2d(2),
            layer_init(nn.Conv2d(8, 4, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            layer_init(nn.Linear(6*6*4, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1)),
        )
        
        self.unit_net = nn.Sequential(
            layer_init(nn.Linear(self.n_unit_states, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1)),
        )
    def get_value(self, x):
        maps, state_params, unit_params = x
        return torch.sum(self.unit_net(unit_params), dim=1) + self.cnn(maps)

        
# TODO network design
class Actor(nn.Module):
    def __init__(self, env, std_clip=1.0):
        super().__init__()
        self.n_ens = env.observation_space[1].shape[0]
        self.n_maps = len(env.single_observation_space[0])
        self.n_state_params = env.single_observation_space[1].shape[0]
        self.n_action = env.single_action_space.shape[0]
        #self.action_dim = env.single_action_space.nvec[-1]
        self.n_unit_states = env.single_observation_space[2].shape[1]
        self.transformer_embedding_dim = env.get_attr("transformer_embedding_dim")[0]
        self.state_param_embedding_dim = env.get_attr("state_param_embedding_dim")[0]
        self.std_clip=std_clip
        
        self.state_params_to_hidden = nn.Sequential(
            layer_init(nn.Linear(self.n_state_params, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.state_param_embedding_dim)),
            nn.ReLU(),
        )
                
        self.cnn_map = nn.Sequential(
            layer_init(nn.Conv2d(self.n_maps, 16, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #layer_init(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            #nn.Tanh(),
            #layer_init(nn.Conv2d(32, 16, kernel_size=3, padding=1)),
            #nn.Tanh(),
            layer_init(nn.Conv2d(16, 4, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            layer_init(nn.Linear(6*6*4, 16)),
            nn.ReLU(),
            
            #nn.Softmax(),
        )
        self.cnn_std = nn.Sequential(
            layer_init(nn.Conv2d(self.n_maps, 8, kernel_size=3, padding=1)),
            nn.Tanh(),
            nn.MaxPool2d(2),
            #layer_init(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            #nn.Tanh(),
            #layer_init(nn.Conv2d(32, 16, kernel_size=3, padding=1)),
            #nn.Tanh(),
            layer_init(nn.Conv2d(8, 1, kernel_size=3, padding=1)),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            layer_init(nn.Linear(6*6*1, self.n_action)),
        )

        self.combine_net = nn.Sequential(
            layer_init(nn.Linear(self.state_param_embedding_dim + 16, 5)),
            nn.ReLU(),
        )
            

        
    def get_action(self, x, action=None, verbose=0):
        maps, state_params, unit_params = x
        batch_size, n_units = unit_params.shape[0], unit_params.shape[1]
        # map of tile values
        cnn_map_out = self.cnn_map(maps)
        state_params_hidden = self.state_params_to_hidden(state_params)
        means = self.combine_net(torch.cat((cnn_map_out,state_params_hidden),dim=1))
        stds = self.cnn_std(maps)#.repeat(1,means.shape[1])
        probs = Normal(means, torch.exp(stds).clamp(max=self.std_clip))
        if action is None:
            action= probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

def env_fn():
    return ProxyEnvironment()

class ProxyEnvironment(gym.Env):
    def __init__(self):
        self.n_maps = 6
        self.n_state_params = 3
        self.transformer_embedding_dim = 16
        self.state_param_embedding_dim = 8
        self.map_space = Tuple((
            Box(np.full((24,24),-10),np.full((24,24),10)),
            Box(np.full((24,24),-10),np.full((24,24),10)),
            Box(np.full((24,24),-10),np.full((24,24),10)),
            Box(np.full((24,24),-10),np.full((24,24),10)),
            Box(np.full((24,24),-10),np.full((24,24),10)),
            Box(np.full((24,24),-10),np.full((24,24),10)),
            Box(np.full((24,24),-10),np.full((24,24),10)),
            Box(np.full((24,24),-10),np.full((24,24),10)),
        ))
        self.unit_param_space = MultiDiscrete(np.repeat(np.expand_dims(np.array([2,576,576,401,11,2]),0),16,axis=0),
                                              start=np.repeat(np.expand_dims(np.array([0,0,0,0,-10,0]),0),16,axis=0))
        self.param_space = Box(np.array([-10 for i in range(11)]),np.array([10 for i in range(11)]))
        self.observation_space = Tuple((self.map_space, self.param_space, self.unit_param_space))
        self.action_space = MultiDiscrete(np.repeat(np.expand_dims(np.array([576*2]),0),16,axis=0),
                                            start=np.repeat(np.expand_dims(np.array([0]),0),16,axis=0))
        self.current_step = 0
        self.curriculum_step = 0
        self.env = LuxAIS3GymEnv(numpy_output=True)
        self.obs, info  = self.env.reset()
        self.agent1 = ProxyAgent("player_0", info["params"], 0)
        self.agent2 = Agent("player_1", info["params"])

    def close(self):
        self.env.close()
        
    def reset(self, seed=42, options=0):
        self.current_step = 0
        self.obs, info = self.env.reset(seed=seed)
        self.agent1 = ProxyAgent("player_0", info["params"], self.curriculum_step)
        self.agent2 = Agent("player_1", info["params"])
        self.proxy_obs = self.agent1.get_init_proxy_obs(self.obs)
        return self.proxy_obs, info

    def step(self, proxy_action):
        actions = dict()
        actions["player_0"] = self.agent1.proxy_to_act(proxy_action)
        actions["player_1"] = np.zeros((16,3),dtype=int)# self.agent2.act(step=self.current_step, obs=self.obs[self.agent2.player])
        #print(self.obs[self.agent1.player])
        self.obs, reward, terminated, truncated, info = self.env.step(actions)
        terminated = terminated["player_0"]
        truncated = truncated["player_0"]
        #print(self.obs[self.agent1.player]["units_mask"])
        self.proxy_obs, self.proxy_reward = self.agent1.step(self.obs[self.agent1.player], self.obs[self.agent1.player]["steps"])
        self.current_step += 1
        return self.proxy_obs, self.proxy_reward, terminated, truncated, info

class ProxyAgent():
    def __init__(self, player: str, env_cfg, cr, model_name=None, inference=False, args=None, actor=None) -> None:
        self.cr = cr
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        if self.player=="player_0":
            self.start_pos = [0,0]
            self.pnum = 1
        else:
            self.start_pos = [23,23]
            self.pnum = 0
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.n_units = self.env_cfg["max_units"]
        self.match_num = 1
        self.relic_map = RelicMap(self.n_units)
        self.tile_map = TileMap()
        self.energy_map = EnergyMap()
        self.move_cost = 3.0
        self.nebula_drain = 5.0
        self.move_check = 0
        self.nebula_check = 0
        
        self.range = self.env_cfg["unit_sensor_range"]
        self.sap_range = self.env_cfg["unit_sap_range"]
        self.sap_cost = self.env_cfg["unit_sap_cost"]
        self.width = self.env_cfg["map_width"]
        self.height = self.env_cfg["map_height"]
        
        self.unit_has_target = -np.ones((self.n_units))
        self.unit_targets = np.zeros((self.n_units,2))
        self.unit_targets_previous = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_path = dict(zip(range(0,self.n_units), [[] for i in range(0,self.n_units)]))
        self.unit_energys = np.full((self.n_units),100)
        self.unit_positions = -np.ones((self.n_units,2))
        self.enemy_positions = -np.ones((self.n_units,2))
        self.available_unit_ids = []
        self.unit_mask = np.zeros((self.n_units),dtype=int)
        self.unit_moved = np.zeros((self.n_units))
        self.prev_points = 0
        self.prev_point_diff = 0
        self.prev_points_increase = 0
        self.wins = 0
        self.losses = 0
        self.prev_actions = np.zeros((self.n_units,3))
        self.prev_proxy_actions = np.zeros((self.n_units,5))
        self.previous_energys = 100*np.ones((self.n_units))
        self.previous_positions = -np.ones((self.n_units,2))
        self.prev_value_map = np.zeros((24,24))
        self.step_num = 0
        if inference:
            if actor is not None:
                self.actor = actor
            else:
                envs = gym.vector.SyncVectorEnv([env_fn for i in range(1)],)
                self.actor = Actor(envs, args.clip_std)
                self.critic = Critic(envs)
                envs.close()
                if model_name:
                    checkpoint = torch.load(model_name, weights_only=True)
                    self.actor.load_state_dict(checkpoint["actor"])
                    self.actor.eval()

        
        a = torch.tensor(np.stack((np.repeat(np.arange(24),24,axis=0).reshape((24,24)), np.repeat(np.arange(24),24,axis=0).reshape((24,24)).T),axis=2))
        self.map_2d_indices = torch.cat((a[:,:,0].view(576,1), a[:,:,1].view(576,1)),dim=1).numpy()
        self.start_distance_map = self.map_2d_indices.sum(-1).reshape((24,24))/np.max(self.map_2d_indices.sum(-1).reshape((24,24)))
        self.map_1d_indices = np.arange(576).reshape((24,24))
        self.dist_indices = p_pos = np.array([np.concatenate((np.arange(16)[i:],np.arange(16)[:i]),axis=0) for i in range(16)])
        self.static_dist_map = np.minimum(self.map_2d_indices.sum(axis=1).reshape((24,24)),np.flip(self.map_2d_indices.sum(axis=1).reshape((24,24))))
        
    def reset(self):
        self.match_num += 1
        self.unit_has_target = -np.ones((self.n_units))
        self.unit_targets = np.zeros((self.n_units,2))
        self.unit_targets_previous = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_path = dict(zip(range(0,self.n_units), [[] for i in range(0,self.n_units)]))
        self.available_unit_ids = []
        self.unit_moved = np.zeros((self.n_units))
        self.prev_points = 0
        self.prev_point_diff = 0
        self.prev_energy_total = self.n_units*100
        self.prev_points_increase = 0
        self.prev_actions = np.zeros((self.n_units,3))
        self.prev_proxy_actions = np.zeros((self.n_units,5))
        self.prev_value_map = np.zeros((24,24))
        self.prev_energys = 100*np.ones((self.n_units))
        self.previous_positions = -np.ones((self.n_units,2))

    def compare_positions(self, pos1, pos2):
        return pos1[0]==pos2[0] and pos1[1]==pos2[1]
        
    # bunnyhop mechanic (maximize points by avoiding doubling on fragment)
    def bunnyhop(self, unit, unit_positions):
        counter = 0
        unit_pos = unit_positions[unit]
        for unit2 in range(self.n_units):            
            if self.unit_has_target[unit2]==2 and self.tile_map.map[unit_positions[unit2][0],unit_positions[unit2][1]]!=2 and len(self.unit_path[unit])>1 and self.compare_positions(self.unit_path[unit][0],unit_positions[unit2]):
                self.unit_path[unit2] = self.unit_path[unit][1:]
                self.unit_targets[unit2] = self.unit_targets[unit]
                self.unit_has_target[unit2] = 1#self.unit_has_target[unit]
                self.unit_path[unit] = [unit_positions[unit2]]
                self.unit_targets[unit] = unit_positions[unit2]
                self.unit_has_target[unit] = 1
                counter +=1
                if counter<10:
                    self.bunnyhop(unit2, unit_positions)

    def in_bounds(self, point):
        return point[0]>0 and point[0]<24 and point[1]>0 and point[1]<24
    
    def positions_to_map(self, unit_positions):
        if type(unit_positions)==dict:
            unit_positions = np.array(list(unit_positions.items()))
        unit_map = np.zeros((24,24))
        for unit in unit_positions:
            if unit[0]!=-1 and unit[1]!=-1:
                unit_map[int(unit[0]),int(unit[1])] = 1
        return unit_map

    # adjust for not only direct hits, but adjacent hits
    def check_hit(self, target):
        for pos in self.enemy_positions:
            if pos[0]!=-1 and pos[1]!=-1:
                if pos[0]==target[0] and pos[1]==target[1]:
                    return 1
        else:
            return 0
    def get_explore_score(self, t):
        score = 0
        for x in range(-2,3):
            for y in range(-2,3):
                if self.in_bounds([t[0]+x,t[1]+y]):
                    if self.tile_map.map[t[0]+x,t[1]+y]==-1:
                        score +=1
        #print(t)
        return score

    def get_close_known_score(self, pos_map):
        weight = torch.tensor(np.array([[0.25,0.5,0.25],
                  [0.5,1,0.5],
                  [0.25,0.5,0.25]])).unsqueeze(0).unsqueeze(0)
        b = torch.nn.functional.conv2d(torch.tensor(self.relic_map.map_knowns+self.relic_map.map_possibles).unsqueeze(0).unsqueeze(0), weight,padding=1)
        c = torch.nn.functional.conv2d(b, weight,padding=1).squeeze()
        return np.sum(pos_map * np.clip(c.numpy().round(2),a_min=None,a_max=3))
    
    def get_init_proxy_obs(self, obs):
         return (np.array([np.zeros((24,24),dtype=float) for i in range(8)]),
                 np.array([0.0 for i in range(11)]), 
                 np.zeros((self.n_units,6),dtype=int))
     
    def step(self, obs, step):        
        #print("\n\n\n", step)
        self.step_num = step
        if step in [101,202,303,404,505]:
            #print("reset")
            self.reset()
        reward = 0
        self.unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        #print(step, unit_mask)
        self.unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        self.enemy_positions = np.array(obs["units"]["position"][abs(self.team_id-1)]).tolist()
        my_unit_map = self.positions_to_map(self.unit_positions)
        enemy_unit_map = self.positions_to_map(self.enemy_positions)
        self.unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        increase = team_points[self.team_id]-self.prev_points
        diff = team_points[self.team_id] - team_points[abs(self.team_id-1)]
        diff_change = diff-self.prev_point_diff
        self.prev_point_diff = diff
        # ids of units you can control at this timestep
        current_tile_map = obs["map_features"]["tile_type"]
        current_energy_map = obs["map_features"]["energy"]
        #print(team_points, increase)
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        # save any new relic nodes that we discover for the rest of the game.
        for ii in visible_relic_node_ids:
            if ii not in self.discovered_relic_nodes_ids:
                # explore units switch to relic collection
                #print(observed_relic_node_positions[ii])
                self.relic_map.new_relic(observed_relic_node_positions[ii])
                self.discovered_relic_nodes_ids.add(ii)
                self.discovered_relic_nodes_ids.add((ii+3)%6)
                self.relic_node_positions.append(observed_relic_node_positions[ii])
        '''if step==400:
            print("knowns and possibl")
            plt.imshow(self.relic_map.map_knowns.T)
            plt.show()
            plt.imshow(self.relic_map.map_possibles.T)
            plt.show()'''

        n_known_old = np.sum(1*(self.tile_map.map[1:23,1:23]!=-1))
        age_sum_old = self.tile_map.map_age.sum()
        
        # update maps
        self.available_unit_ids = np.where(self.unit_mask)[0].tolist()
        for unit in self.available_unit_ids.copy():
            if self.unit_energys[unit]<self.move_cost:
                self.available_unit_ids.remove(unit)
        self.relic_map.step(self.unit_positions, increase)
        tile_shift = self.tile_map.update(current_tile_map, step)
        energy_shift = self.energy_map.update(current_energy_map)
        
        n_known = np.sum(1*(self.tile_map.map[1:23,1:23]!=-1))
        age_sum = (self.tile_map.map_age-1).sum()
        #reward += 0.0001*(age_sum_old-age_sum)
        #reward += increase
        #reward += 0.01*self.get_close_known_score(self.positions_to_map(self.unit_positions))
        #reward += 0.001*self.get_close_known_score(self.positions_to_map(self.unit_positions))
        #reward += 0.01*self.get_close_known_score(self.positions_to_map(self.unit_targets))
        reward += 0.001*max(0,n_known)
        '''for unit in range(self.n_units):
            if self.prev_actions[unit,0]==5:
                reward -= 0.1
            pos = self.unit_positions[unit]
            t = self.unit_targets[unit]
            if pos[0]!=-1 and pos[1]!=-1:
                if self.relic_map.map_knowns[int(t[0]),int(t[1])]==1:
                    reward += 1
                if self.relic_map.map_possibles[int(t[0]),int(t[1])]==1:
                    reward += 0.1'''
        #reward += (self.relic_map.map_knowns*self.prev_value_map).sum()
        #reward += (0.1*self.relic_map.map_possibles*self.prev_value_map).sum()
                
            
        

        # find out move cost
        if step>2 and not self.move_check and self.tile_map.map[self.unit_positions[0][0],self.unit_positions[0][1]]!=1 and self.unit_moved[0]:
            self.move_cost=self.previous_energys[0]-self.unit_energys[0]+self.energy_map.map[self.unit_positions[0][0],self.unit_positions[0][1]]
            self.move_check=1
        # find out nebula drain
        if not self.nebula_check and self.move_check:
            for unit in self.available_unit_ids:
                if self.unit_moved[unit] and  self.tile_map.map[self.unit_positions[unit][0],self.unit_positions[unit][1]]==1:
                    self.nebula_check=1
                    self.nebula_drain = -(self.unit_energys[unit]-self.previous_energys[unit]-self.energy_map.map[self.unit_positions[unit][0],self.unit_positions[unit][1]]+self.move_cost)
                    break

        
        self.previous_energys = self.unit_energys
        self.prev_points = team_points[self.team_id]
        self.prev_points_increase = increase
        self.previous_positions = self.unit_positions

        # TODO explore map
        tiles = np.zeros((24,24))
        tiles[self.tile_map.map==-1] = 1
        tile_age = np.abs(self.tile_map.map_age-step)
        tile_age = (tile_age-tile_age.mean())/(tile_age.std()+1e-10)
        energy = self.energy_map.map.copy()
        energy[self.tile_map.map==1] = energy[self.tile_map.map==1] - self.nebula_drain
        energy = (energy-np.mean(energy))/(np.std(energy)+1e-10)
        on_known = np.zeros((self.n_units,1))
        tile_energys = np.zeros((self.n_units,1))
        target_map = self.positions_to_map(self.unit_targets)
        for ii, p in enumerate(self.unit_positions):
            if self.relic_map.map_knowns[p[0],p[1]]==1:
                on_known[ii] = 1
            tile_energys[ii] = energy[p[0],p[1]]
        # constructing observations
        # maps: unknown tile, energy, possibles, knowns, unit, enemy units
        obs_maps = np.array([target_map.astype(float), tiles.astype(float), tile_age, energy, self.relic_map.map_possibles.astype(float), 
                             self.relic_map.map_knowns.astype(float), my_unit_map.astype(float), enemy_unit_map.astype(float)])
        # param: episode 1 hot, epi step, p diff, unit e, living units
        episode = [0,0,0,0,0]
        episode[int(max(0,(step-1)//101))] = 1
        obs_params = np.array(episode+[((step-1)%101)/101, increase/16, diff/1000, np.sum(self.unit_mask*self.unit_energys)/(16*400), np.sum(1*(self.unit_mask))/16, len(self.discovered_relic_nodes_ids)/6])
        obs_units = np.concatenate((np.expand_dims(np.array(self.unit_mask),-1).astype(int), np.expand_dims(self.map_1d_indices[self.unit_positions[:,0],self.unit_positions[:,1]],-1).astype(int), 
                                    np.expand_dims(self.map_1d_indices[self.unit_targets[:,0].astype(int),self.unit_targets[:,1].astype(int)],-1).astype(int), 
                                    np.expand_dims(self.unit_energys,-1).astype(int), tile_energys.astype(int), on_known.astype(int)), axis=-1)
        proxy_obs = (obs_maps, 
                     obs_params, 
                     obs_units,
                    )
        return proxy_obs, reward
        
        
    def act(self, obs, step):
        proxy_obs, _ = self.step(obs, step)
        proxy_obs = (torch.tensor(proxy_obs[0]).to(torch.float32).unsqueeze(0),torch.tensor(proxy_obs[1]).to(torch.float32).unsqueeze(0),torch.tensor(proxy_obs[2]).to(torch.float32).unsqueeze(0))
        proxy_action,_,_ = self.actor.get_action(proxy_obs)
        return self.proxy_to_act(proxy_action)
        
        
        
    def proxy_to_act(self, proxy_action):
        if torch.is_tensor(proxy_action):
            proxy_action = proxy_action.squeeze().cpu().detach().numpy()
        else:
            proxy_action = proxy_action.squeeze()
        actions = np.zeros((self.n_units, 3), dtype=int)
        discover_flag = 0
        for ii, unit in enumerate(self.available_unit_ids):
            unit_pos = self.unit_positions[unit]
            if proxy_action[unit]>=576:
                t = self.map_2d_indices[proxy_action[unit]-576]
                actions[unit] = [5,int(t[0]),int(t[1])]
            else:
                t = self.map_2d_indices[proxy_action[unit]]
                self.unit_targets[unit] = [int(t[0]),int(t[1])]
                '''if not self.compare_positions(self.unit_targets[unit], self.unit_targets_previous[unit]):
                    path, _ = a_star(unit_positions[unit], self.unit_targets[unit], self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, self.move_cost, self.nebula_drain, use_energy=False)
                    self.unit_path[unit] = path[1:]'''
                direction = direction_to(self.unit_positions[unit], self.unit_targets[unit])
                change = direction_to_change(direction)
                self.unit_path[unit] = [[int(self.unit_positions[unit][0]+change[0]),int(self.unit_positions[unit][1]+change[1])]]
                if self.unit_energys[unit]<self.move_cost:
                        actions[unit]=[0,0,0]
                elif self.unit_path[unit]:
                    if self.relic_map.map_possibles[self.unit_path[unit][0][0],self.unit_path[unit][0][1]]==1:
                        if discover_flag:
                            if self.relic_map.map_possibles[unit_pos[0],unit_pos[1]]==1:
                                actions[unit] = self.relic_map.move_away(self.tile_map.map, [unit_pos[0],unit_pos[1]])
                                self.unit_path[unit].insert(0, unit_pos)
                            else:
                                actions[unit]=[0,0,0]
                        else:
                            actions[unit] = [direction_to(unit_pos, self.unit_path[unit].pop(0)), 0, 0]
                            discover_flag=1
                    else:
                        actions[unit] = [direction_to(unit_pos, self.unit_path[unit].pop(0)), 0, 0]
                else:
                    if self.relic_map.map_possibles[unit_pos[0],unit_pos[1]]==1:
                        if discover_flag:
                            actions[unit] = self.relic_map.move_away(self.tile_map.map, [unit_pos[0],unit_pos[1]])
                            self.unit_path[unit].insert(0, unit_pos)
                        else:
                            actions[unit]=[0,0,0]
                            discover_flag = 1
                    else:
                        actions[unit]=[0,0,0]
        #self.prev_value_map = target_map
        self.prev_proxy_actions = proxy_action
        self.prev_actions = actions
        self.unit_targets_previous = self.unit_targets
        return actions

def pretrain_to_move(actor, envs, device="cpu"):
    optimizer = optim.Adam(actor.parameters(), lr=0.001, eps=1e-5)
    loss_fn = nn.MSELoss()
    mask = torch.zeros((16,5))
    mask[:,0] = 1
    for i in range(1000):
        x = (torch.randint(0,100,((16,)+np.array(envs.single_observation_space[0]).shape)).to(torch.float).to(device),
           torch.randint(0,100,((16,)+envs.single_observation_space[1].shape)).to(torch.float).to(device),
          torch.randint(0,100,((16,)+np.array(envs.single_observation_space[2]).shape)).to(torch.float).to(device))
        y = torch.rand((16,16,5))
        y[:,:,0] = 0
        y = torch.log(y)
        action, pred,_ = actor.get_action(x)
        y[action[:,:,0]==0] = 0
        y[action[:,:,0]==1] = -1e10
        loss = loss_fn(mask*y,mask*pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return actor

'''obs_maps = np.array([target_map.astype(float), tiles.astype(float), tile_age, energy, self.relic_map.map_possibles.astype(float), 
                     self.relic_map.map_knowns.astype(float), my_unit_map.astype(float), enemy_unit_map.astype(float), self.prev_value_map.astype(float)])
# param: episode 1 hot, epi step, p diff, unit e, living units
episode = [0,0,0,0,0]
episode[int(max(0,(step-1)//101))] = 1
obs_params = np.array(episode+[(step-1)%101, increase, diff, np.sum(self.unit_mask*self.unit_energys), np.sum(1*(self.unit_mask)), len(self.discovered_relic_nodes_ids)])
obs_units = np.concatenate((np.expand_dims(np.array(self.unit_mask),-1).astype(int), np.expand_dims(self.map_1d_indices[self.unit_positions[:,0],self.unit_positions[:,1]],-1).astype(int), 
                            np.expand_dims(self.map_1d_indices[self.unit_targets[:,0].astype(int),self.unit_targets[:,1].astype(int)],-1).astype(int), 
                            np.expand_dims(self.unit_energys,-1).astype(int), tile_energys.astype(int), on_known.astype(int)), axis=-1)'''


def pretrain_value_map(actor, envs, device="cpu"):
    optimizer = optim.Adam(actor.parameters(), lr=0.0001, eps=1e-8)
    bsize = 128
    loss_fn = nn.MSELoss()
    losses = []
    for i in range(1000):
        x = (torch.rand((bsize,9,24,24)).to(torch.float).to(device),
            torch.rand(((bsize,)+envs.single_observation_space[1].shape)).to(torch.float).to(device),
            torch.rand(((bsize,)+np.array(envs.single_observation_space[2]).shape)).to(torch.float).to(device))
        x[0][:,4] =  torch.randint(2,(24,24))*torch.randint(2,(24,24))*torch.randint(2,(24,24))
        x[0][:,5] =  torch.randint(2,(24,24))*torch.randint(2,(24,24))*torch.randint(2,(24,24))
        y = 0.01*x[0][:,2] + 0.1*x[0][:,4] + x[0][:,5]
        y = y.view(bsize,-1)
        action, pred,_ = actor.get_action(x)
        diff = -torch.abs(action[:,:576]-y)
        #print(diff.shape, )
        loss = loss_fn(diff,pred[:,:576])
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    plt.plot(losses)
    plt.show()
    return actor


def render_episode(episode: RecordEpisode) -> None:
    data = json.dumps(episode.serialize_episode_data(), separators=(",", ":"))
    display(Javascript(f"""
var iframe = document.createElement('iframe');
iframe.src = 'https://s3vis.lux-ai.org/#/kaggle';
iframe.width = '100%';
iframe.scrolling = 'no';

iframe.addEventListener('load', event => {{
    event.target.contentWindow.postMessage({data}, 'https://s3vis.lux-ai.org');
}});

new ResizeObserver(entries => {{
    for (const entry of entries) {{
        entry.target.height = `${{Math.round(320 + 0.3 * entry.contentRect.width)}}px`;
    }}
}}).observe(iframe);

element.append(iframe);
    """))

def evaluate_model(actor=None, seed=42, args = None, games_to_play=1, replay_save_dir="replays", render=True, model_name=None):
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
    )
    obs, info = env.reset()
    total_wins = np.zeros((2))
    for i in range(games_to_play):
        start = time.time()
        obs, info = env.reset(seed=np.random.randint(100000))
        env_cfg = info["params"] 
        agent = ProxyAgent("player_0", env_cfg, 0, actor=actor, model_name=model_name, inference=True, args=args)
        # main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        while not game_done:
            actions = dict()
            actions["player_0"] = agent.act(step=step, obs=obs["player_0"])
            actions["player_1"] = np.zeros((16,3),dtype=int)
            obs, reward, terminated, truncated, info = env.step(actions)
            # info["state"] is the environment state object, you can inspect/play around with it to e.g. print
            # unobservable game data that agents can't see
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1
        total_wins += np.array([reward["player_0"],reward["player_1"]])
        if render:
            render_episode(env)
        print("Runtime: ", time.time()-start)
    print(total_wins/games_to_play)
    env.close() # free up resources and save final replay

'''energy = self.energy_map.map.copy()
energy[self.tile_map.map==1] = energy[self.tile_map.map==1] - self.nebula_drain
energy = (energy-np.mean(energy))/(np.std(energy)+1e-10)
tile_age = np.abs(self.tile_map.map_age-self.step_num)
age_map = (tile_age-tile_age.mean())/(tile_age.std()+1e-10)
target_map = (proxy_action[0]*age_map
              +proxy_action[1]*energy
              +proxy_action[2]*self.relic_map.map_possibles
              +proxy_action[3]*self.relic_map.map_knowns
              #+proxy_action[4]*self.positions_to_map(self.enemy_positions)
             )
#target_map += self.relic_map.map_possibles + 1.5*self.relic_map.map_knowns
agf = 0
flat_indices = np.argsort(target_map.ravel())[-16:][::-1]
max_indices = np.array(np.unravel_index(flat_indices, target_map.shape)).T
# assign highest value targets to closest units
p = np.full((self.n_units,2),100)
p[self.available_unit_ids] = self.unit_positions[self.available_unit_ids]
dists = np.array([np.sum(np.abs(max_indices-np.concatenate((p[i:,:],p[:i,:]),axis=0)),axis=1) for i in range(16)])
targets = dict()
for i in range(len(self.available_unit_ids)):
    closest = int(self.dist_indices[np.argmin(dists[:,i]),i])
    dists[self.dist_indices==closest] = 1e5
    targets[closest] = [int(max_indices[i,0]),int(max_indices[i,1])]
if self.step_num==400:
    plt.imshow(target_map.T)
    plt.show()
    plt.imshow(self.relic_map.map_knowns.T)
    plt.show()
    plt.imshow(self.relic_map.map_possibles.T)
    plt.show()
    print(targets)'''