import json
from IPython.display import display, Javascript
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import os
from my_agent.lux.utils import direction_to, direction_to_change
import matplotlib.pyplot as plt
import numpy as np
import random
from maps import EnergyMap, RelicMap, TileMap
from astar import *
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete, Tuple
from agent import Agent

def env_fn():
    return ProxyEnvironment()
    
class ProxyEnvironment(gym.Env):
    def __init__(self):
        self.n_maps = 6
        self.n_state_params = 3
        self.transformer_embedding_dim = 8
        self.state_param_embedding_dim = 8
        self.map_space = Tuple((
            MultiDiscrete(np.full((24,24),24)),
            MultiDiscrete(np.full((24,24),24)),
            MultiDiscrete(np.full((24,24),24)),
            MultiDiscrete(np.full((24,24),24)),
            MultiDiscrete(np.full((24,24),24)),
            MultiDiscrete(np.full((24,24),24)),
        ))
        self.unit_param_space = MultiDiscrete(np.repeat(np.expand_dims(np.array([24,24,24,24,401,11,2]),0),16,axis=0),
                                              start=np.repeat(np.expand_dims(np.array([0,0,0,0,0,-10,0]),0),16,axis=0))
        self.param_space = MultiDiscrete(np.array([505, 1000, 16*400]))
        self.observation_space = Tuple((self.map_space, self.param_space, self.unit_param_space))
        #print(self.observation_space)
        self.action_space = MultiDiscrete(np.repeat(np.expand_dims(np.array([2,24,24,24,24]),0),16,axis=0))

class ProxyAgent():
    def __init__(self, player: str, env_cfg, model_name=None) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
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
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_targets_previous = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_path = dict(zip(range(0,self.n_units), [[] for i in range(0,self.n_units)]))
        self.unit_energys = np.full((self.n_units),100)
        self.unit_positions = -np.ones((self.n_units,2))
        self.available_unit_ids = []
        self.unit_moved = np.zeros((self.n_units))
        self.prev_points = 0
        self.prev_point_diff = 0
        self.prev_points_increase = 0
        self.prev_actions = np.zeros((self.env_cfg["max_units"], 5), dtype=int)
        self.previous_energys = 100*np.zeros((self.n_units))
        self.previous_positions = -np.ones((self.n_units,2))
        envs = gym.vector.SyncVectorEnv([env_fn for i in range(1)],)
        self.model = ActorCritic(envs)
        envs.close()
        if model_name:
            self.model.load_state_dict(torch.load(model_name, weights_only=True))
            self.model.eval()


    def get_explore(self, current):
        a = np.stack((np.repeat(np.arange(24),24,axis=0).reshape((24,24)), np.repeat(np.arange(24),24,axis=0).reshape((24,24)).T),axis=2)
        a[current!=-1] = [100,100]
        self.explore_choices = a[np.sum(np.abs(a-np.array(self.start_pos)),axis=-1)<24-self.range].tolist()
        if self.explore_choices:
            return random.choice(self.explore_choices)
        else:
            x = np.random.randint(0,24)
            y = np.random.randint(0,24-x)
            return [abs(x-self.start_pos[0]), abs(y-self.start_pos[1])]

    def get_moves(self, obs, unit_id, unit_pos):
        prev_pos = [unit_pos[0] - direction_to_change(self.prev_actions[unit_id][0])[0], unit_pos[1] - direction_to_change(self.prev_actions[unit_id][0])[1]]
        new_pos = [[unit_pos[0], unit_pos[1]-1],
                  [unit_pos[0]+1, unit_pos[1]],
                  [unit_pos[0], unit_pos[1]+1],
                  [unit_pos[0]-1, unit_pos[1]]]
        moves = [0]
        for ii, pos in enumerate(new_pos):
            if pos[0]<0 or pos[1]<0 or pos[0]>=self.width or pos[1]>=self.height or (pos[0]==prev_pos[0] and pos[1]==prev_pos[1]) or obs["map_features"]["tile_type"][pos[0], pos[1]]==2 :
            #if pos[0]<0 or pos[1]<0 or pos[0]>23 or pos[1]>23 or obs["map_features"]["tile_type"][pos[0], pos[1]]==2:
                pass
            else:
                moves.append(direction_to(unit_pos, pos))
        #print(moves)
        return moves
        
    def reset(self):
        self.match_num += 1
        self.unit_has_target = -np.ones((self.n_units))
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_targets_previous = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_path = dict(zip(range(0,self.n_units), [[] for i in range(0,self.n_units)]))
        self.available_unit_ids = []
        self.unit_moved = np.zeros((self.n_units))
        self.prev_points = 0
        self.prev_point_diff = 0
        self.prev_points_increase = 0
        self.prev_actions = np.zeros((self.env_cfg["max_units"], 5), dtype=int)
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

    def positions_to_map(self, unit_positions):
        unit_map = np.zeros((24,24))
        for unit in unit_positions:
            if unit[0]!=-1 and unit[1]!=-1:
                unit_map[unit[0],unit[1]] = 1
        return unit_map

    # adjust for not only direct hits, but adjacent hits
    def check_hit(self, target):
        for pos in self.enemy_positions:
            if pos[0]!=-1 and pos[1]!=-1:
                if pos[0]==target[0] and pos[1]==target[1]:
                    return 1
        else:
            return 0

    def get_init_proxy_obs(self, obs):
         return (np.array([np.zeros((24,24),dtype=int) for i in range(6)]),np.array([0,0,0]), np.zeros((self.n_units,7),dtype=int))
     
    def step(self, obs, step):
        reward = 0
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        self.unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        self.enemy_positions = np.array(obs["units"]["position"][abs(self.team_id-1)]).tolist()
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
        ### proxy reward calculation ###
        # change in point difference
        '''reward += diff
        # units on known fragment tiles
        for unit in range(self.n_units):
            pos = self.unit_positions[unit]
            if pos[0]!=-1 and pos[1]!=-1:
                if self.relic_map.map_knowns[pos[0],pos[1]]==1:
                    reward += 1
                # units targeting possibles/known fragments
                #t = self.unit_targets[unit]
                #print(t[0], t[1])
                #if self.relic_map.map_knowns[t[0],t[1]]==1 or self.relic_map.map_possibles[t[0],t[1]]==1:
                #    reward += 1
            # unit dies (negative reward)
            else: 
                if self.unit_moved[unit]:
                    reward += -1
            # hit enemy
            action = self.prev_actions[unit]
            if action[0]==5:
                reward += self.check_hit(action[1:])'''
            
            
        
        if step in [102,203,304,405]:
            self.reset()
            
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        # save any new relic nodes that we discover for the rest of the game.
        for ii in visible_relic_node_ids:
            if ii not in self.discovered_relic_nodes_ids:
                # explore units switch to relic collection
                self.relic_map.new_relic(observed_relic_node_positions[ii])
                self.discovered_relic_nodes_ids.add(ii)
                self.discovered_relic_nodes_ids.add((ii+3)%6)
                self.relic_node_positions.append(observed_relic_node_positions[ii])
        # update maps
        self.available_unit_ids = np.where(unit_mask)[0].tolist()
        self.relic_map.step(self.unit_positions, increase)
        tile_shift = self.tile_map.update(current_tile_map)
        energy_shift = self.energy_map.update(current_energy_map)        

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
        tiles = np.ones((24,24))
        tiles[self.tile_map.map==2] = 0
        energy = self.energy_map.map.copy()
        energy[self.tile_map.map==1] = energy[self.tile_map.map==1] - self.nebula_drain
        my_unit_map = self.positions_to_map(self.unit_positions)
        enemy_unit_map = self.positions_to_map(self.enemy_positions)
        on_known = np.zeros((self.n_units,1))
        tile_energys = np.zeros((self.n_units,1))
        for ii, p in enumerate(self.unit_positions):
            if self.relic_map.map_knowns[p[0],p[1]]==1:
                on_known[ii] = 1
            tile_energys[ii] = energy[p[0],p[1]]
                

        proxy_obs = (np.array([tiles.astype(int), energy.astype(int), self.relic_map.map_possibles.astype(int), self.relic_map.map_knowns.astype(int), my_unit_map.astype(int), enemy_unit_map.astype(int)]), 
                     np.array([step, diff, np.sum(self.unit_energys)]), 
                     np.concatenate((np.array(self.unit_positions).astype(int), np.array(list(self.unit_targets.values())).astype(int), np.expand_dims(self.unit_energys,-1).astype(int), tile_energys.astype(int), on_known.astype(int)), axis=-1))
       # print(self.available_unit_ids)
        return proxy_obs, reward
        
    def act(self, obs, step):
        proxy_obs, _ = self.step(obs, step)
        proxy_obs = (torch.tensor(proxy_obs[0]).to(torch.float32).unsqueeze(0),torch.tensor(proxy_obs[1]).to(torch.float32).unsqueeze(0),torch.tensor(proxy_obs[2]).to(torch.float32).unsqueeze(0))
        proxy_action,_ ,_ ,_ = self.model.get_action_and_value(proxy_obs)
        return self.proxy_to_act(proxy_action)
        
        
        
    def proxy_to_act(self, proxy_action):
        if torch.is_tensor(proxy_action):
            proxy_action = proxy_action.squeeze().cpu().detach().numpy()

        #print(proxy_action)
        actions = np.zeros((self.n_units, 3), dtype=int)
        for unit in self.available_unit_ids:
            if proxy_action[unit,0]==1:
                actions[unit] = [5, proxy_action[unit,3], proxy_action[unit,4]]
            else:
                self.unit_targets[unit] = [proxy_action[unit,1],proxy_action[unit,2]]
                '''if not self.compare_positions(self.unit_targets[unit], self.unit_targets_previous[unit]):
                    path, _ = a_star(unit_positions[unit], self.unit_targets[unit], self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, self.move_cost, self.nebula_drain, use_energy=False)
                    self.unit_path[unit] = path[1:]'''
                direction = direction_to(self.unit_positions[unit], self.unit_targets[unit])
                change = direction_to_change(direction)
                self.unit_path[unit] = [self.unit_positions[unit][0]+change[0],self.unit_positions[unit][1]+change[1]]
                actions[unit] = [direction, 0, 0]

        self.prev_actions = actions
        self.unit_targets_previous = self.unit_targets
        return actions

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_ens = env.observation_space[1].shape[0]
        self.n_maps = len(env.single_observation_space[0])
        self.n_state_params = env.single_observation_space[1].shape[0]
        self.n_action = env.single_action_space.shape[0]
        self.action_dim = env.single_action_space.nvec[-1,-1]
        self.n_unit_states = env.single_observation_space[2].shape[1]
        self.transformer_embedding_dim = env.get_attr("transformer_embedding_dim")[0]
        self.state_param_embedding_dim = env.get_attr("state_param_embedding_dim")[0]
        #self.transformer_embedding_dim = env[0].transformer_embedding_dim
        
        self.state_params_to_hidden = nn.Sequential(
            layer_init(nn.Linear(self.n_state_params, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.state_param_embedding_dim)),
            nn.ReLU(),
        )
        
        self.embedding_maps = nn.Sequential(
            layer_init(nn.Linear(self.n_maps, 16)),
            layer_init(nn.Linear(16, self.transformer_embedding_dim)),
        )
        
        self.embedding_unit_params = nn.Sequential(
            layer_init(nn.Linear(self.n_unit_states, 16)),
            layer_init(nn.Linear(16, self.transformer_embedding_dim)),
        )
        
        self.actor_encoder = torch.nn.TransformerEncoderLayer(self.transformer_embedding_dim,4,64, batch_first=True)
        self.actor_decoder = torch.nn.TransformerDecoderLayer(self.transformer_embedding_dim,4,64, batch_first=True)
        
        self.critic_encoder = torch.nn.TransformerEncoderLayer(self.transformer_embedding_dim,4,64, batch_first=True)
        self.critic_decoder = torch.nn.TransformerDecoderLayer(self.transformer_embedding_dim,4,64, batch_first=True)
        self.out_to_logits = nn.Sequential(
            layer_init(nn.Linear(self.transformer_embedding_dim+self.state_param_embedding_dim, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 2+4*24)),
            nn.ReLU(),
        )
        
        self.encoder_out_to_critic = nn.Sequential(
            layer_init(nn.Linear(8, 1)),
            nn.ReLU(),
        )

        self.critic_out_old = nn.Sequential(
            layer_init(nn.Linear(self.transformer_embedding_dim, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1)),
        )
        self.critic_out = nn.Sequential(
            layer_init(nn.Linear(self.n_unit_states, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1)),
        )
        

    def critic_old(self, maps, unit_params):
        encoder_out = self.critic_encoder(self.embedding_maps(maps)) # B x 576 x 8
        decoder_out = self.critic_decoder(self.embedding_unit_params(unit_params), encoder_out) # B x N x 8
        return torch.sum(self.critic_out_old(decoder_out).squeeze(), dim=-1)

    def critic(self, maps, unit_params):
        return torch.sum(self.critic_out(unit_params).squeeze(), dim=-1)
        
        
    def get_value(self, x):
        maps, state_params, unit_params = x
        maps = torch.flatten(maps,start_dim=-2).permute(0,2,1)
        value = self.critic(maps, unit_params)
        return value

    def get_action_and_value(self, x, action=None):
        maps, state_params, unit_params = x
        maps = torch.flatten(maps,start_dim=-2).permute(0,2,1)
        batch_size, n_units = unit_params.shape[0], unit_params.shape[1] # B, N
        
        encoder_out = self.actor_encoder(self.embedding_maps(maps)) # B x 576 x 16
        decoder_out = self.actor_decoder(self.embedding_unit_params(unit_params), encoder_out) # B x N x 16
        
        state_params_hidden = self.state_params_to_hidden(state_params) # B x 8
        decoder_out_state_params_combined = torch.cat((decoder_out, torch.stack([state_params_hidden for i in range(n_units)],dim=1)),dim=-1) # B x N x 24
        all_logits = self.out_to_logits(decoder_out_state_params_combined) # B x N x 2+4*24
        
        move_type_logits = all_logits[:,:,:2].reshape(batch_size, n_units, 1, 2) # B x N x 1 x 2
        target_logits = all_logits[:,:,2:].reshape(batch_size, n_units, 4, self.action_dim) # B x N x 4 x 24
        move_type_probs = Categorical(logits=move_type_logits)
        target_probs = Categorical(target_logits)

        value = self.critic(maps, unit_params)
        
        if action is None:
            action_type = move_type_probs.sample()
            action_target = target_probs.sample()
            action = torch.cat((action_type,action_target),dim=-1) # B x N x 5
        else:
            action_type = action[:,:,0].unsqueeze(dim=-1)
            action_target = action[:,:,1:]
        probs = torch.cat((move_type_probs.log_prob(action_type), target_probs.log_prob(action_target)),dim=-1) # B x N x 5
        return action, probs, move_type_probs.entropy() + target_probs.entropy(), value

## NOT DONE
class ActorCritic2(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_ens = env.observation_space[1].shape[0]
        self.n_maps = len(env.single_observation_space[0])
        self.n_state_params = env.single_observation_space[1].shape[0]
        self.n_action = env.single_action_space.shape[0]
        self.action_dim = env.single_action_space.nvec[0]
        self.n_unit_states = env.single_observation_space[2].shape[0]
        
        self.state_params_to_hidden = nn.Sequential(
            layer_init(nn.Linear(self.n_state_params, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 8)),
            nn.ReLU(),
        )
        self.embedding_maps = nn.Sequential(
            layer_init(nn.Linear(self.n_maps, 32)),
            layer_init(nn.Linear(32, 16)),
        )
        self.embedding_unit_params = nn.Sequential(
            layer_init(nn.Linear(self.n_unit_states, 32)),
            layer_init(nn.Linear(32, 16)),
        )
        self.encoder = torch.nn.TransformerEncoderLayer(16,4,128)
        self.decoder = torch.nn.TransformerDecoderLayer(16,4,128)
        self.out_to_logits = nn.Sequential(
            layer_init(nn.Linear(24, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 2+4*24)),
            nn.ReLU(),
        )
        
        self.encoder_out_to_critic = nn.Sequential(
            layer_init(nn.Linear(16, 1)),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(576 + 8, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1)),
        )
    
    def get_value(self, x):
        return self.critic(self.combine(x))

    def get_action_and_value(self, x, action=None):
        maps, state_params, unit_params = x
        batch_size, n_units = unit_params.shape[0], unit_params.shape[1] # B, N
        encoder_out = self.encoder(self.self.embedding_maps(maps)) # B x 576 x 16
        decoder_out = self.decoder(self.embedding_unit_params(unit_params), encoder_out) # B x N x 16
        state_params_hidden = self.state_params_to_hidden(state_params) # B x 8
        decoder_out_state_params_combined = torch.cat((decoder_out, torch.stack([state_params_hidden for i in range(n_units)],dim=1)),dim=-1) # B x N x 24
        all_logits = self.out_to_logits(decoder_out_state_params_combined) # B x N x 2+4*24
        
        move_type_logits = all_logits[:,:,:2].reshape(batch_size, n_units, 1, 2) # B x N x 1 x 2
        target_logits = all_logits[:,:,2:].reshape(batch_size, n_units, 4, self.action_dim) # B x N x 4 x 24
        move_type_probs = Categorical(logits=move_type_logits)
        target_probs = Categorical(logitstarget_logits)

        encoder_out_critic = self.encoder_out_to_critic(encoder_out).squeeze()
        value = self.critic(torch.cat(encoder_out_critic, state_params_hidden),dim=-1)
        
        if action is None:
            action_type = move_type_probs.sample()
            action_target = target_probs.sample()
            action = torch.cat((action_type,action_target),dim=-1) # B x N x 5
        #probs = torch.cat((move_type_probs.log_prob(action_type).reshape((batch_size, n_units, 2)),
                          #target_probs.log_prob(action_target).reshape((batch_size, n_units, 4*24))),dim=-1) # B x N x 2+4*24
        probs = torch.cat((move_type_probs.log_prob(action_type), target_probs.log_prob(action_target)),dim=-1) # B x N x 5
        return action, probs, move_type_probs.entropy() + target_probs.entropy(), value




        