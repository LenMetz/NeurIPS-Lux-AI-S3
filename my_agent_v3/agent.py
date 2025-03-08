from lux.utils import direction_to, direction_to_change
import numpy as np
import random
from maps import RelicMap, TileMap, EnergyMap
from astar import *
import torch

class Agent():
    def __init__(self, player: str, env_cfg, model_name=None, predict_mode=1) -> None:
        self.predict_mode=predict_mode
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
        self.sap_t = 1.3 - self.predict_mode*0.35
        self.range = self.env_cfg["unit_sensor_range"]
        self.sap_range = self.env_cfg["unit_sap_range"]
        self.sap_cost = self.env_cfg["unit_sap_cost"]
        self.width = self.env_cfg["map_width"]
        self.height = self.env_cfg["map_height"]
        q = torch.tensor(np.stack((np.repeat(np.arange(5),5,axis=0).reshape((5,5)), np.repeat(np.arange(5),5,axis=0).reshape((5,5)).T),axis=2))
        self.aim_weight = ((q-np.array([2,2])).sum(-1).to(float))
        self.aim_weight = self.aim_weight+1
        self.dist_map = np.stack((np.repeat(np.arange(24),24,axis=0).reshape((24,24)), np.repeat(np.arange(24),24,axis=0).reshape((24,24)).T),axis=2).sum(axis=-1)
        self.a = np.stack((np.repeat(np.arange(24),24,axis=0).reshape((24,24)), np.repeat(np.arange(24),24,axis=0).reshape((24,24)).T),axis=2)
        self.explore_border = np.ones((24,24))
        self.explore_border[self.range:24-self.range,self.range:24-self.range] = np.zeros((24-2*self.range,24-2*self.range))
        self.explore_choices = self.a[np.sum(np.abs(self.a-np.array(self.start_pos)),axis=-1)<24-self.range]
        self.explore_targets = []
        a = [[i,abs(i-23+self.range)] for i in range(23-self.range+1)]
        if self.team_id==1:
            a = [[abs(t[1]-23),abs(t[0]-23)] for t in a]
        for ii, t in enumerate(a):
            if ii%(self.range+2)==0:
                self.explore_targets.append(t)
        self.relic_targets = []
        self.fragment_targets = []
        self.locked_relic_targets = []
        self.locked_fragment_targets = []
        self.fragment_locations = []
        self.occupied_fragments = []

        self.n_explore_units = 0
        self.unit_has_target = -np.ones((self.n_units)) # -1=no target; 0=explore target; 1=relic target; 2=on relic
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_path = dict(zip(range(0,self.n_units), [[] for i in range(0,self.n_units)]))
        self.unit_moved = np.zeros((self.n_units))
        self.prev_points = 0
        self.prev_points_increase = 0
        self.prev_actions = None
        self.previous_energys = 100*np.zeros((self.n_units))
        self.previous_positions = -np.ones((self.n_units,2))
        self.previous_predictions = np.zeros((24,24))
        self.accs = []


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
        
    # moves around asteroids
    def move_obstacle_avoid(self, obs, unit_id, unit_pos, direction):
        moves = self.get_moves(obs, unit_id, unit_pos)
        if direction in moves:
            return direction
        elif moves:
            return random.choice(moves)
        else:
            return 0
            
    def relic_to_targets(self, pos):
        targets = []
        for i in range(-2,3,1):
            for j in range(-2,3,1):
                if pos[0]+i>=0 and pos[0]+i<=23 and pos[1]+j>=0 and pos[1]+j<=23:
                    new_target = np.array([pos[0]+i, pos[1]+j])
                    mirrored_target = np.abs(new_target-np.array([self.width, self.height]))
                    targets.append(new_target)
                    #targets.append(mirrored_target)
        return targets
        
    def reset(self):
        self.match_num += 1
        self.relic_map.reset()
        self.explore_targets = []
        self.unit_has_target = -np.ones((self.n_units)) # -1=no target; 0=explore target; 1=relic target; 2=on relic, 3=known fragment
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_path = dict(zip(range(0,self.n_units), [[] for i in range(0,self.n_units)]))
        self.unit_moved = np.zeros((self.n_units))
        self.prev_points = 0
        self.prev_points_increase = 0
        self.prev_actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        self.prev_energys = 100*np.ones((self.n_units))
        self.previous_positions = -np.ones((self.n_units,2))
        self.previous_predictions = np.zeros((24,24))
        self.occupied_fragments = []
        self.n_explore_units = 0
        if self.match_num==2:
            self.n_explore_units = 3
        elif self.match_num==3:
            self.n_explore_units = 2

    def find_best_unit(self, goal, available_unit_ids, unit_positions, unit_energys, use_energy=True, energy_surplus=0):
        best_unit = 0
        best_pos = [-1,-1]
        best_dist = np.inf
        best_path = [goal]
        if len(available_unit_ids)>0:
            for ii, unit_id in enumerate(available_unit_ids):
                unit_pos = unit_positions[unit_id]
                path, dist = a_star(unit_pos, goal, self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, self.move_cost, 
                                    self.nebula_drain, use_energy=use_energy, budget=unit_energys[ii]-energy_surplus)
                if dist<best_dist:
                    best_dist = dist
                    best_path = path
                    best_unit = unit_id
                    best_pos = unit_pos
        return best_unit, best_path

    def compare_positions(self, pos1, pos2):
        return pos1[0]==pos2[0] and pos1[1]==pos2[1]

    def get_attack_targets(self):
        fragments = self.relic_map.get_fragments(self.start_pos)
        targets = []
        for frag in fragments:
            if self.tile_map.map[frag[0],frag[1]]!=2:
                if self.start_pos[0]==0:
                    if frag[0]+frag[1]>23:
                        if frag in self.occupied_fragments: 
                            targets.append(frag)
                        else:
                            targets.insert(0,frag)
                else:
                    if frag[0]+frag[1]<23:
                        if frag in self.occupied_fragments: 
                            targets.append(frag)
                        else:
                            targets.insert(0,frag)
        return targets

    def get_defend_targets(self, unit_map, n):
        weight = torch.ones((3,3),dtype=float)
        if self.team_id==0:
            unit_map[self.a.sum(-1)>=23] = 0
        else:
            unit_map[self.a.sum(-1)<=23] = 0
        b = torch.nn.functional.conv2d(torch.tensor(unit_map).unsqueeze(0).unsqueeze(0), weight.unsqueeze(0).unsqueeze(0), padding="same").squeeze().numpy()
        c = torch.nn.functional.conv2d(torch.tensor(unit_map).unsqueeze(0).unsqueeze(0), weight.unsqueeze(0).unsqueeze(0), padding="same")
        c = torch.nn.functional.conv2d(c, weight.unsqueeze(0).unsqueeze(0), padding="same")
        c = torch.nn.functional.conv2d(c, weight.unsqueeze(0).unsqueeze(0), padding="same")
        d = torch.nn.functional.conv2d(c, weight.unsqueeze(0).unsqueeze(0), padding="same").squeeze().numpy()/100
        d[np.clip(b,-1,1)==1] = 0
        d = d*self.energy_map.map + 0.1*np.abs(self.a-self.start_pos).sum(-1)
        flat_indices = np.argsort(d.ravel())[-n:]
        max_indices = np.array(np.unravel_index(flat_indices, d.shape)).T
        return max_indices.tolist()
        

    def get_enemy_targets(self, pos, enemy_positions, relative=True):
        targets = []
        for dx in range(-self.sap_range,self.sap_range):
            for dy in range(-self.sap_range,self.sap_range):
                if [pos[0]+dx,pos[1]+dy] in enemy_positions:
                    if relative:
                        targets.append([dx,dy])
                    else:
                        targets.append([pos[0]+dx,pos[1]+dy])
        return targets
        
    def repath(self, unit_positions):
        for unit in range(self.n_units):
            pos = unit_positions[unit]
            if self.compare_positions(pos, [-1,-1]):
                pos = self.start_pos
            if self.unit_path[unit]:
                self.unit_path[unit],_ = a_star(pos, self.unit_targets[unit], self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                self.unit_path[unit].pop(0)

    def free_target(self, unit, pos):
        if self.unit_has_target[unit]==1:
            self.relic_targets.append([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
        if self.unit_has_target[unit]==2:
            self.fragment_targets.append([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
            if [int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])] in self.occupied_fragments:
                self.occupied_fragments.remove([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])

    def sort_targets(self, targets, positions):
        
        if targets and positions:
                T = np.repeat(np.expand_dims(np.array(targets),axis=0),len(positions),axis=0)
                P = np.tile(np.array(positions), len(targets)).reshape(len(positions),len(targets),2)
                dists = np.sum(np.abs(T-P),axis=2)
                best = np.min(dists,axis=0)
                return (np.array(targets)[np.argsort(best)]).tolist()
        else:
            return []

    def preaim(self, pos):
        frags = self.relic_map.get_fragments(self.start_pos)
        if pos in frags:
            return pos
        frags = self.relic_map.get_fragments(self.start_pos, own=True)
        dist = np.sum(np.abs(np.array(frags)-np.array(pos)),axis=-1)
        target = frags[np.argsort(dist)[0]]
        path, _ = a_star(pos, target, self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, self.move_cost, self.nebula_drain, use_energy=True)
        return path[1]

    def positions_to_map(self, unit_positions, unit_energys=None):
        if type(unit_positions)==dict:
            unit_positions = np.array(list(unit_positions.items()))
        unit_map = np.zeros((24,24))
        for ii,unit in enumerate(unit_positions):
            if unit[0]!=-1 and unit[1]!=-1:
                if unit_energys is None or unit_energys[ii]>0:
                    unit_map[int(unit[0]),int(unit[1])] = 1
        return unit_map

    # bunnyhop mechanic (maximize points by avoiding doubling on fragment)
    def bunnyhop(self, unit, unit_positions, unit_energys):
        counter = 0
        unit_pos = unit_positions[unit]
        for unit2 in range(self.n_units):            
            if unit_energys[unit2]>self.move_cost and self.unit_has_target[unit2]==2 and self.tile_map.map[unit_positions[unit2][0],unit_positions[unit2][1]]!=2 and len(self.unit_path[unit])>1 and self.compare_positions(self.unit_path[unit][0],unit_positions[unit2]):
                self.unit_path[unit2] = self.unit_path[unit][1:]
                self.unit_targets[unit2] = self.unit_targets[unit]
                self.unit_has_target[unit2] = 1#self.unit_has_target[unit]
                self.unit_path[unit] = [unit_positions[unit2]]
                self.unit_targets[unit] = unit_positions[unit2]
                self.unit_has_target[unit] = 1
                counter +=1
                if counter<10:
                    self.bunnyhop(unit2, unit_positions, unit_energys)
                    
    def model_action(self, p, e, own_map, enemy_map, energy_map):
        input_params = torch.tensor([e/400, (self.env_cfg["unit_sap_range"]-3)/5, self.env_cfg["unit_move_cost"]/4, (self.env_cfg["unit_sap_cost"]-30)/21, 
                                     0.5,0.25], dtype=torch.float32)
        sap_range_map = torch.zeros((1,24,24))
        sap_range_map[0,p[0]-self.sap_range:p[0]+self.sap_range+1,p[1]-self.sap_range:p[1]+self.sap_range+1] = 1
        in_pos_map = torch.zeros((1,24,24))
        in_pos_map[0,p[0],p[1]] = 1
        X = torch.tensor(np.concatenate((in_pos_map, sap_range_map, np.expand_dims(own_map, axis=0), np.expand_dims(enemy_map, axis=0), np.expand_dims(self.relic_map.map_knowns, axis=0), 
                                         np.expand_dims(self.tile_map.get_asteroid_map(), axis=0), np.expand_dims(energy_map, axis=0)),axis=0), dtype=torch.float32)
        if self.team_id==0:
            X = torch.flip(X,dims=[2])
            X = torch.rot90(X,k=3,dims=[1,2])
        y = self.model(X.unsqueeze(0),input_params.unsqueeze(0)).detach()
        if self.team_id==0:
            y_flip = y.detach().clone()
            y_flip[0,1] = y[0,2]
            y_flip[0,2] = y[0,1]
            y_flip[0,3] = y[0,4]
            y_flip[0,4] = y[0,3]
            y = y_flip
        return y
        
    def predict_enemies(self, own_positions, enemy_positions, enemy_energys, energy_map):
        energy_map = (energy_map-energy_map.mean())/(energy_map.std()+1e-8)
        own_map = self.positions_to_map(own_positions)
        enemy_map = self.positions_to_map(enemy_positions)
        weight = torch.ones((2*self.range+1,2*self.range+1),dtype=float)
        vis_map = torch.nn.functional.conv2d(input=torch.tensor(enemy_map,dtype=float).unsqueeze(0).unsqueeze(0),weight=weight.unsqueeze(0).unsqueeze(0), stride=1, padding="same").squeeze().numpy()
        vis_map[vis_map>0] = 1
        vis_own_map = own_map*vis_map
        new_enemy_map = torch.zeros((24,24))
        for unit, pos in enumerate(enemy_positions):
            if pos[0]!=-1 and pos[1]!=-1:
                out = self.model_action(pos, enemy_energys[unit], enemy_map, vis_own_map, energy_map)[0]
                new_enemy_map[pos[0],pos[1]] += out[0]
                new_enemy_map[pos[0],self.p_clip(pos[1]-1)] += out[1]
                new_enemy_map[self.p_clip(pos[0]+1),pos[1]] += out[2]
                new_enemy_map[pos[0],self.p_clip(pos[1]+1)] += out[3]
                new_enemy_map[self.p_clip(pos[0]-1),pos[1]] += out[4]
        
        return new_enemy_map

    
    def predict_enemies_rule(self, own_positions, enemy_positions, enemy_energys):
        weight = self.aim_weight.clone()
        ast_map = self.tile_map.map.copy()
        ast_map[ast_map==0] = 1
        ast_map[ast_map==2] = 0
        a1 = self.relic_map.map_knowns.copy()*ast_map
        a2 = self.relic_map.map_knowns.copy()*ast_map
        a1[self.dist_map>=23] = 0
        a2[self.dist_map<=23] = 0
        if self.team_id==0:
            weight = torch.rot90(weight,2)
            d = torch.nn.functional.conv2d(torch.tensor(a2).unsqueeze(0).unsqueeze(0),weight.unsqueeze(0).unsqueeze(0),padding="same")
        else:
            d = torch.nn.functional.conv2d(torch.tensor(a1).unsqueeze(0).unsqueeze(0),weight.unsqueeze(0).unsqueeze(0),padding="same")
        weight = torch.ones((5,5),dtype=float)
        its = 0
        while (d==0).any() and its<10:
            its +=1
            e = torch.nn.functional.conv2d(d,weight.unsqueeze(0).unsqueeze(0),padding="same")
            d[d==0] = e[d==0]
        d = np.clip(d.squeeze().numpy(),-1,1)
        new_enemy_map = np.zeros((24,24))
        for unit, pos in enumerate(enemy_positions):
            if enemy_energys[unit]<0:
                pass
            elif enemy_energys[unit]<self.move_cost:
                new_enemy_map[pos[0],pos[1]] += 1
            elif self.relic_map.map_knowns[pos[0],pos[1]]==1:
                new_enemy_map[pos[0],pos[1]] += 1
            elif a1.sum()>0:
                if self.team_id==0:
                    if d[pos[0],pos[1]]<0:
                        frags = np.transpose((a1==1).nonzero())
                    else:
                        frags = np.transpose((a2==1).nonzero())
                else:
                    if d[pos[0],pos[1]]>0:
                        frags = np.transpose((a1==1).nonzero())
                    else:
                        frags = np.transpose((a2==1).nonzero())
                t = np.argmin(np.abs(frags-np.array(pos)).sum(1))
                path, _ = a_star(pos, frags[t], self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, self.move_cost, self.nebula_drain, use_energy=True)
                new_enemy_map[path[1][0],path[1][1]] += 1
            else:
                new_enemy_map[pos[0],pos[1]] += 1
        return torch.tensor(new_enemy_map,dtype=torch.float32)
                
                
    def explore_oldest(self, p):
        age_map = self.current_age_map.copy()
        dist_map = np.abs(p-self.a).sum(-1)
        max_index = np.unravel_index(np.argmax(self.current_age_map-dist_map-50*self.explore_border), self.current_age_map.shape)
        self.current_age_map[max_index[0]-self.range:max_index[0]+self.range+1,max_index[0]-self.range:max_index[0]+self.range+1] = 0
        return max_index

    def sap_range_map(self, p):
        sap_range_map = torch.zeros((24,24))
        sap_range_map[p[0]-self.sap_range:p[0]+self.sap_range+1,p[1]-self.sap_range:p[1]+self.sap_range+1] = 1
        return sap_range_map
        
    def p_clip(self, p):
        return min(23,max(0,p))
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        #print("Step: ", step)
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        enemy_positions = np.array(obs["units"]["position"][abs(self.team_id-1)])
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        enemy_energys = np.array(obs["units"]["energy"][abs(self.team_id-1)]) # shape (max_units, 1)
        enemy_map = self.positions_to_map(enemy_positions, enemy_energys)
        if self.previous_predictions.sum()>0:
            self.accs.append((self.previous_predictions*enemy_map).sum()/self.previous_predictions.sum())
        if step==504:
            print(np.mean(self.accs))
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        increase = team_points[self.team_id]-self.prev_points
        # ids of units you can control at this timestep
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        current_tile_map = obs["map_features"]["tile_type"]
        current_energy_map = obs["map_features"]["energy"]
        vision_mask = obs["sensor_mask"]
        
        if step in [102,203,304,405]:
            self.reset()
            
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        # save any new relic nodes that we discover for the rest of the game.
        for ii in visible_relic_node_ids:
            if ii not in self.discovered_relic_nodes_ids:
                # explore units switch to relic collection
                self.relic_map.new_relic(observed_relic_node_positions[ii])
                self.unit_has_target[self.unit_has_target==0]=-1
                self.unit_has_target[self.unit_has_target==3]=-1
                self.discovered_relic_nodes_ids.add(ii)
                self.discovered_relic_nodes_ids.add((ii+3)%6)
                self.relic_node_positions.append(observed_relic_node_positions[ii])
                self.relic_targets.extend(self.relic_map.get_possibles(self.start_pos, own=True))
                # remove duplicates from relic targets
                self.relic_targets = np.array(list({array.tobytes(): array for array in np.array(self.relic_targets)}.values())).tolist()
        # update maps
        available_unit_ids = np.where(unit_mask)[0].tolist()
        self.relic_map.step(unit_positions, increase)
        tile_shift = self.tile_map.update(current_tile_map, step)
        energy_shift = self.energy_map.update(current_energy_map)        

        # find out move cost
        if step>2 and not self.move_check and self.tile_map.map[unit_positions[0][0],unit_positions[0][1]]!=1 and self.unit_moved[0]:
            self.move_cost=self.previous_energys[0]-unit_energys[0]+self.energy_map.map[unit_positions[0][0],unit_positions[0][1]]
            self.move_check=1
        # find out nebula drain
        if not self.nebula_check and self.move_check:
            for unit in available_unit_ids:
                if self.unit_moved[unit] and  self.tile_map.map[unit_positions[unit][0],unit_positions[unit][1]]==1:
                    self.nebula_check=1
                    self.nebula_drain = -(unit_energys[unit]-self.previous_energys[unit]-self.energy_map.map[unit_positions[unit][0],unit_positions[unit][1]]+self.move_cost)
                    break
            
        if tile_shift or energy_shift:
            self.fragment_targets.extend(self.locked_fragment_targets)
            self.relic_targets.extend(self.locked_relic_targets)
            self.locked_relic_targets = []
            self.locked_fragment_targets = []

        # collision detection ->repath
        for unit in available_unit_ids:
            if self.prev_actions[unit][0] in [1,2,3,4] and self.compare_positions(self.previous_positions[unit], unit_positions[unit]) and self.previous_energys[unit]>self.move_cost:
                    self.unit_path[unit],_ = a_star(unit_positions[unit], self.unit_targets[unit], self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, 
                                                    self.move_cost, self.nebula_drain, use_energy=True, budget=unit_energys[unit])
                    self.unit_path[unit].pop(0)
        
        self.occupied_fragments = []
        for unit in available_unit_ids.copy():
            pos = [int(unit_positions[unit][0]), int(unit_positions[unit][1])]
            # set moved flag necessary for kill detection
            if not self.compare_positions(pos, self.start_pos):
                self.unit_moved[unit] = 1
            # if unit is on unoccupied fragment, stay and remove this tile as target from other units, but append to possible/fragment targets if necessary
            if self.relic_map.map_knowns[pos[0], pos[1]]==1 and [pos[0],pos[1]] not in self.occupied_fragments and self.compare_positions(self.unit_targets[unit],pos):
                self.unit_has_target[unit] = 2
                self.unit_targets[unit] = pos
                self.unit_path[unit] = []
                self.occupied_fragments.append([int(pos[0]), int(pos[1])])
                remain_units = available_unit_ids.copy()
                remain_units.remove(unit)
                for unit2 in remain_units:
                    if self.unit_targets[unit2][0]==pos[0] and self.unit_targets[unit2][1]==pos[1]:# and self.unit_has_target[unit]!=3:
                        # decide who goes and who stays based on energy
                        if unit_energys[unit]>unit_energys[unit2]:
                            self.unit_has_target[unit2] = -1
                        else:
                            self.unit_has_target[unit2] = 2
                            self.unit_has_target[unit] = -1
                            self.unit_targets[unit2] = pos
                            self.unit_path[unit2] = []
            if self.unit_has_target[unit]==3 and self.compare_positions(self.unit_targets[unit],pos):
                self.unit_has_target[unit]==4
            # remove target if possible fragment has been cleared (by other unit)
            if self.unit_has_target[unit]==1 and self.relic_map.map_possibles[self.unit_targets[unit][0], self.unit_targets[unit][1]]==0 and self.relic_map.map_knowns[self.unit_targets[unit][0], self.unit_targets[unit][1]]!=1:
                self.unit_has_target[unit] = -1
            # retarget def units
            if self.unit_has_target[unit]==3 and self.compare_positions(self.unit_targets[unit],pos):
                if self.fragment_locations:
                    target = self.get_defend_targets(self.fragment_locations[unit%len(self.fragment_locations)])
                    self.unit_targets[unit] = target
                    self.unit_path[unit],_ = a_star(pos, target, self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, self.move_cost, self.nebula_drain, use_energy=True, budget=unit_energys[unit])
                    self.unit_path[unit].pop(0)
                else:
                    self.unit_has_target[unit] = -1
            # free unit that has been killed and reissue targets if necessary
            if self.unit_moved[unit] and self.compare_positions(pos, self.start_pos):
                self.free_target(unit, pos)
                self.unit_has_target[unit] = -1
                self.unit_moved[unit] = 0
            # untarget if target is asteroid, keep track of blocked targets to reuse when shift happens
            if self.tile_map.map[int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])]==2 and not self.compare_positions(pos,self.unit_targets[unit]):
                if self.unit_has_target[unit]==1:
                    self.locked_relic_targets.append([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
                if self.unit_has_target[unit]==2:
                    self.locked_fragment_targets.append([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
                self.unit_has_target[unit]=-1
            # remove if path empty (can happen if blocked by asteroid or pathing mistakes happens)
            if not self.unit_path[unit] and self.unit_has_target[unit]<2:
                if self.unit_has_target[unit]==0:
                    self.n_explore_units += 1
                self.unit_has_target[unit]=-1
                if not self.compare_positions(pos, self.unit_targets[unit]):
                    self.free_target(unit, pos)
            # if unit has possible or known as target or no energy remove from available
            if self.unit_has_target[unit]==2 or unit_energys[unit]<self.move_cost:
                available_unit_ids.remove(unit)
            if unit_energys[unit]<self.move_cost and not self.compare_positions(pos, self.unit_targets[unit]):
                self.free_target(unit, pos)
            
        self.fragment_targets = self.relic_map.get_fragments(self.start_pos, own=True)
        self.relic_targets = self.relic_map.get_possibles(self.start_pos, own=True)
        for f in self.fragment_targets.copy():
            if self.tile_map.map[f[0],f[1]]==2 or f in self.occupied_fragments:
                self.fragment_targets.remove(f)
        for r in self.relic_targets.copy():
            if self.tile_map.map[r[0],r[1]]==2:
                self.relic_targets.remove(r)
         
        positions = []
        for u in available_unit_ids:
            positions.append(unit_positions[u])
        all_targets = self.sort_targets(self.fragment_targets, positions) + self.sort_targets(self.relic_targets, positions) 

        for ii, goal in enumerate(all_targets):
            if available_unit_ids:
                unit, path = self.find_best_unit(goal, available_unit_ids, unit_positions, unit_energys, use_energy=np.abs(self.start_pos-np.array([goal[0],goal[1]])).sum()>13) # max energy if contested, else shortest
                available_unit_ids.remove(unit)
                self.unit_path[unit] = path[1:]
                self.unit_has_target[unit] = 1
                self.unit_targets[unit] = goal
        for ii, goal in enumerate(self.explore_targets):
            if available_unit_ids:
                unit, path = self.find_best_unit(goal, available_unit_ids, unit_positions, unit_energys, use_energy=True)
                available_unit_ids.remove(unit)
                self.unit_path[unit] = path[1:]
                self.unit_has_target[unit] = 0
                self.unit_targets[unit] = goal
                self.explore_targets.remove(goal)
                
        # only keep targets that aren't exploring
        for unit in available_unit_ids.copy():
            if self.unit_has_target[unit]==0 or self.unit_has_target[unit]==3:
                if not self.compare_positions(unit_positions[unit], self.unit_targets[unit]):
                    available_unit_ids.remove(unit)
        
        
        if available_unit_ids and self.relic_map.map_knowns.sum()>0:
            if self.predict_mode==1:
                defense_targets = self.get_attack_targets()
            else:
                defense_targets = self.get_defend_targets(self.relic_map.map_knowns,len(available_unit_ids))
        else:
            attack_targets = []
            defense_targets = []
        self.current_age_map = self.tile_map.map_age
        # send remaining units to explore (first match) or target 
        for ii, unit in enumerate(available_unit_ids.copy()):
            if not self.n_explore_units>0 and defense_targets:
                path, _ = a_star(unit_positions[unit], defense_targets[min(len(defense_targets)-1,ii)], self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, 
                                 self.move_cost, self.nebula_drain, use_energy=True, budget=unit_energys[unit])
                self.unit_path[unit] = path[1:]
                self.unit_has_target[unit] = 4
                self.unit_targets[unit] = defense_targets[min(len(defense_targets)-1,ii)]
            else:
                if self.predict_mode==0:
                    target = self.get_explore(current_tile_map)
                else:
                    target = self.explore_oldest(unit_positions[unit]) #
                path, _ = a_star(unit_positions[unit], target, self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, self.move_cost, 
                                 self.nebula_drain, use_energy=True, budget=unit_energys[unit])
                available_unit_ids.remove(unit)
                self.unit_path[unit] = path[1:]
                self.unit_has_target[unit] = 0
                self.unit_targets[unit] = target
                self.n_explore_units -= 1
                
        sap_map = np.zeros((24,24))
        enemy_prediction = np.zeros((24,24))
        if not (np.array(enemy_positions)==-1).all():
            enemy_prediction = self.predict_enemies_rule(unit_positions, enemy_positions, enemy_energys)
            weight = torch.full((3,3),0.58)
            weight[1,1] = 1
            sap_map = torch.nn.functional.conv2d(enemy_prediction.unsqueeze(0).unsqueeze(0), weight.unsqueeze(0).unsqueeze(0),padding="same").squeeze().numpy()
        
        unseen_frag = self.relic_map.map_knowns.copy()
        if self.team_id==0:
            unseen_frag[self.dist_map<=23] = 0
        else:
            unseen_frag[self.dist_map>=23] = 0
        if self.predict_mode==1:
            sap_map += unseen_frag*(1*np.logical_not(vision_mask))
        sap_count = np.zeros((24,24))
        discover_flag = 0
        # Decide on action. Follow path, if multiple units want to move to possible fragment only let one through, if attacking fire on enemy instead of moving
        for unit in range(self.n_units):
            unit_pos = unit_positions[unit]
            self.bunnyhop(unit, unit_positions, unit_energys)
        for unit in range(self.n_units):
            unit_pos = unit_positions[unit]
            if unit_mask[unit]:
                for node in self.unit_path[unit]: # repath if blocked by asteroid
                    if self.tile_map.map[node[0],node[1]]==2:
                        path, _ = a_star(unit_positions[unit], self.unit_targets[unit], self.tile_map.map, self.energy_map.map, self.relic_map.map_knowns, 
                                         self.move_cost, self.nebula_drain, use_energy=True, budget=unit_energys[unit])
                        self.unit_path[unit] = path[1:]
                        break
                max_value = 0
                if sap_map.sum()>1e-10:
                    range_map = self.sap_range_map(unit_pos).numpy()
                    unit_sap_map = sap_map*range_map
                    max_index = np.unravel_index(np.argmax(unit_sap_map), unit_sap_map.shape)
                    max_value = sap_map[max_index[0],max_index[1]]
                if self.unit_has_target[unit]==0:
                    thresh=1
                if self.unit_has_target[unit]==1:
                    thresh=1.5
                if self.unit_has_target[unit]==2:
                    thresh=1
                if self.unit_has_target[unit]==3:
                    thresh=0.75
                if self.unit_has_target[unit]==4:
                    thresh=0.5
                # sap
                if self.predict_mode==0:
                    thresh = 0.9
                if unit_energys[unit]>self.sap_cost and max_value>thresh and (self.unit_has_target[unit]==3 or self.unit_has_target[unit]==2 or self.unit_has_target[unit]==4):
                    actions[unit]=[5,max_index[0]-unit_pos[0],max_index[1]-unit_pos[1]]
                else:
                    if unit_energys[unit]<self.move_cost:
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
        self.previous_predictions = enemy_prediction
        self.previous_energys = unit_energys
        self.relic_map.map_occupied = np.zeros((24,24))
        self.prev_points = team_points[self.team_id]
        self.prev_points_increase = increase
        self.prev_actions = actions
        self.previous_positions = unit_positions
        return actions