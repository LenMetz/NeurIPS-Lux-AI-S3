from lux.utils import direction_to, direction_to_change
import numpy as np
import random
from maps import RelicMap, TileMap, EnergyMap
from astar import *

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
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
        self.relic_map = RelicMap(self.n_units)
        self.tile_map = TileMap()
        self.energy_map = EnergyMap()
        self.move_cost = 3.0
        self.nebula_drain = 5.0
        
        self.range = self.env_cfg["unit_sensor_range"]
        self.width = self.env_cfg["map_width"]
        self.height = self.env_cfg["map_height"]
        self.explore_targets = [(self.range+1, self.height-self.range-1), 
                                (self.width-self.range-1, self.range+1), 
                                (abs(self.width-self.start_pos[0])+self.range*(-1)**self.pnum, abs(self.height-self.start_pos[0])+self.range*(-1)**self.pnum)
                               ]
        self.relic_targets = []
        self.fragment_targets = []
        self.locked_relic_targets = []
        self.locked_fragment_targets = []
        self.fragment_locations = []
        self.occupied_fragments = []
        
        self.unit_has_target = -np.ones((self.n_units)) # -1=no target; 0=explore target; 1=relic target; 2=on relic
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_path = dict(zip(range(0,self.n_units), [[] for i in range(0,self.n_units)]))
        self.unit_moved = np.zeros((self.n_units))
        self.prev_points = 0
        self.prev_points_increase = 0
        self.prev_actions = None
        self.previous_positions = -np.ones((self.n_units,2))


    def get_explore(self):
        options = np.transpose((self.tile_map.map==-1).nonzero()).tolist()
        if options:
            return random.choice(options)
        else:
            return [0,0]
            
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

    def sort_closest(self, targets, pos):
        distances = []
        for target in targets:
            _, d = a_star(pos, target, self.tile_map.map, np.zeros((24,24)), self.move_cost, 0)
            distances.append(d)
        return np.array(targets)[np.argsort(distances)].tolist()
        
    
    def reset(self):
        self.relic_map.reset()
        self.explore_targets = []
        self.unit_has_target = -np.ones((self.n_units)) # -1=no target; 0=explore target; 1=relic target; 2=on relic, 3=known fragment
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.unit_path = dict(zip(range(0,self.n_units), [[] for i in range(0,self.n_units)]))
        self.unit_moved = np.zeros((self.n_units))
        self.prev_points = 0
        self.prev_points_increase = 0
        self.prev_actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        self.previous_positions = -np.ones((self.n_units,2))
        self.fragment_locations = self.relic_map.get_fragments()
        self.fragment_locations = self.sort_closest(self.fragment_locations, self.start_pos)
        self.possible_locations = self.relic_map.get_possibles()
        self.possible_locations = self.sort_closest(self.possible_locations, self.start_pos)
        self.occupied_fragments = []
        #print(self.fragment_locations, self.possible_locations)
        # TODO target closet first
        for unit, target in enumerate(self.fragment_locations + self.possible_locations):
            if unit<self.n_units:
                self.unit_has_target[unit] = 1
                self.unit_targets[unit] = target
                self.unit_path[unit],_ = a_star(self.start_pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                self.unit_path[unit].pop(0)
        #print(self.unit_targets, "\n", self.unit_path)

    def find_best_unit(self, goal, available_unit_ids, unit_positions):
        best_unit = 0
        best_pos = [-1,-1]
        best_dist = np.inf
        best_path = [goal]
        if len(available_unit_ids)>0:
            for ii, unit_id in enumerate(available_unit_ids):
                unit_pos = unit_positions[unit_id]
                path, dist = a_star(unit_pos, goal, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                if dist<best_dist:
                    best_dist = dist
                    best_path = path
                    best_unit = unit_id
                    best_pos = unit_pos
        return best_unit, best_path

    def compare_positions(self, pos1, pos2):
        return pos1[0]==pos2[0] and pos1[1]==pos2[1]

    def get_attack_targets(self):
        fragments = self.relic_map.get_fragments()
        targets = []
        for frag in fragments:
            if self.tile_map.map[frag[0],frag[1]]!=2:
                if self.start_pos[0]==0:
                    if frag[0]+frag[1]>23:
                        targets.append(frag)
                else:
                    if frag[0]+frag[1]<23:
                        targets.append(frag)
        return targets

    def get_enemy_targets(self, pos, enemy_positions):
        targets = []
        for dx in range(-self.range,self.range):
            for dy in range(-self.range,self.range):
                if [pos[0]+dx,pos[1]+dy] in enemy_positions:
                    targets.append([dx,dy])
        #print(pos, enemy_positions, targets)
        return targets
    
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        #print("Step: ", step)
        if step in [102,203,304,405]:
            self.reset()
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        enemy_positions = []
        for pos in np.array(obs["units"]["position"][abs(self.team_id-1)]).tolist():
            if pos[0]!=-1 and pos[1]!=-1:
                enemy_positions.append(pos)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        increase = team_points[self.team_id]-self.prev_points
        # ids of units you can control at this timestep
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        current_tile_map = obs["map_features"]["tile_type"]
        current_energy_map = obs["map_features"]["energy"]
        
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                # explore units switch to relic collection
                self.relic_map.new_relic(observed_relic_node_positions[id])
                self.unit_has_target[self.unit_has_target==0]=-1
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
                self.relic_targets.extend(self.relic_to_targets(observed_relic_node_positions[id]))
                # remove duplicates from relic targets
                self.relic_targets = list({array.tobytes(): array for array in np.array(self.relic_targets)}.values())
        # update maps
        available_unit_ids = np.where(unit_mask)[0].tolist()
        self.relic_map.step(unit_positions, increase)
        tile_shift = self.tile_map.update(current_tile_map)
        self.energy_map.update(current_energy_map)
        if tile_shift:
            self.fragment_targets.extend(self.locked_fragment_targets)
            self.relic_targets.extend(self.locked_relic_targets)
            self.locked_relic_targets = []
            self.locked_fragment_targets = []
            for unit in range(self.n_units):
                pos = unit_positions[unit]
                if self.compare_positions(pos, [-1,-1]):
                    pos = self.start_pos
                if self.unit_path[unit]:
                    self.unit_path[unit],_ = a_star(pos, self.unit_targets[unit], self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                    self.unit_path[unit].pop(0)
        
        all_targets = self.fragment_targets + self.relic_targets + self.explore_targets
        release_count = len(all_targets)
        for unit in available_unit_ids.copy():
            pos = unit_positions[unit]
            #print("before ", pos, unit, self.unit_has_target[unit], self.unit_targets[unit], self.unit_path[unit])
            # set moved flag necessary for kill detection
            if not self.compare_positions(pos, self.start_pos):
                self.unit_moved[unit] = 1
            # if unit is on unoccupied fragment, stay and remove this tile as target from other units, but append to possible/fragment targets if necessary
            if self.relic_map.map_knowns[pos[0], pos[1]]==1 and [pos[0],pos[1]] not in self.occupied_fragments:
                # if occupied tile is not original target, free target if not explore
                if not (self.unit_targets[unit][0]==pos[0] and self.unit_targets[unit][1]==pos[1]):
                    if self.unit_has_target[unit]==1:
                        self.relic_targets.append(self.unit_targets[unit])
                    if self.unit_has_target[unit]==2:
                        self.fragment_targets.append(self.unit_targets[unit])
                self.unit_has_target[unit] = 2
                self.unit_targets[unit] = pos
                self.unit_path[unit] = []
                self.occupied_fragments.append([pos[0], pos[1]])
                remain_units = available_unit_ids.copy()
                remain_units.remove(unit)
                for unit2 in remain_units:
                    if self.unit_targets[unit2][0]==pos[0] and self.unit_targets[unit2][1]==pos[1] and self.unit_has_target[unit]!=3:
                        self.unit_has_target[unit2] = -1

            # free unit that is exploring if higher priority target is available
            if (self.unit_has_target[unit]==0 or self.unit_has_target[unit]==3) and release_count>0:
                release_count -= 1
                self.unit_has_target[unit]=-1
            # free unit that has been killed and reissue targets if necessary
            if self.unit_moved[unit] and self.compare_positions(pos, self.start_pos):
                if self.unit_has_target[unit]==1:
                    self.relic_targets.append([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
                if self.unit_has_target[unit]==2:
                    self.fragment_targets.append([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
                    if self.compare_positions(pos, self.unit_targets[unit]):
                        self.occupied_fragments.remove([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
                self.unit_has_target[unit] = -1
                self.unit_moved[unit] = 0
            # free unit that's on target and not occupying fragment
            if self.unit_has_target[unit]<2 and self.unit_targets[unit][0]==pos[0] and self.unit_targets[unit][1]==pos[1]:
                #print("unit at target", unit)
                self.unit_has_target[unit]=-1
            # untarget if target is asteroid, keep track of blocked targets to reuse when shift happens
            if self.tile_map.map[int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])]==2 and not self.compare_positions(pos,self.unit_targets[unit]):
            #print("blocked by asteroid")
                if self.unit_has_target[unit]==1:
                    self.locked_relic_targets.append([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
                if self.unit_has_target[unit]==2:
                    self.locked_fragment_targets.append([int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])])
                self.unit_has_target[unit]=-1
            # remove if path empty (THIS SHOULDN'T HAPPEN)
            if not self.unit_path[unit] and self.unit_has_target[unit]<2:
                #print("path empty")
                self.unit_has_target[unit]=-1
            # if unit has possible or known as target or no energy remove from available
            if self.unit_has_target[unit]>0 or unit_energys[unit]<self.move_cost:
                #print("remove unit", unit)
                available_unit_ids.remove(unit)
            #print("after ", unit, pos, self.unit_has_target[unit], self.unit_targets[unit], self.unit_path[unit])
            # if unit is available set path to nothing
            #if self.unit_has_target[unit]==-1:
            #    self.unit_path[unit] = []

        #print(available_unit_ids)
        # send available units to fragments, possibles, explore targets
        all_targets = self.fragment_targets + self.relic_targets# + self.explore_targets
        target_type = [2 for i in range(len(self.fragment_targets))] + [1 for i in range(len(self.relic_targets))]# + [0 for i in range(len(self.explore_targets))]
        for ii, goal in enumerate(all_targets):
            if available_unit_ids:
                unit, path = self.find_best_unit(goal, available_unit_ids, unit_positions)
                available_unit_ids.remove(unit)
                self.unit_path[unit] = path[1:]
                self.unit_has_target[unit] = target_type[ii]
                self.unit_targets[unit] = goal
                if target_type[ii]==2:
                    self.fragment_targets.remove(goal)
                if target_type[ii]==1:
                    self.relic_targets.remove(goal)
        
        # only keep targets that aren't exploring
        for unit in available_unit_ids.copy():
            if self.unit_has_target[unit]==0 or self.unit_has_target[unit]==3:
                available_unit_ids.remove(unit)
                
        # send remaining units to explore (first match) or target 
        if step<102:
            for unit in available_unit_ids.copy():
                target = self.get_explore()
                path, _ = a_star(unit_positions[unit], target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                available_unit_ids.remove(unit)
                self.unit_path[unit] = path[1:]
                self.unit_has_target[unit] = 0
                self.unit_targets[unit] = target
        else:
            targets = self.get_attack_targets()
            if targets:
                for unit in available_unit_ids:
                    target_id, path = self.find_best_unit(unit_positions[unit], np.arange(len(targets)), targets)
                    if path:
                        self.unit_path[unit] = path[::-1]
                    self.unit_path[unit].append(targets[target_id])
                    self.unit_has_target[unit] = 3
                    self.unit_targets[unit] = targets[target_id]

        
        discover_flag = 0
        # Decide on action. Follow path, if multiple units want to move to possible fragment only let one through, if attacking fire on enemy instead of moving
        for unit in range(self.n_units):
            if unit_mask[unit]:
                unit_pos = unit_positions[unit]
                if self.get_enemy_targets(unit_pos.tolist(), enemy_positions):
                    targets = self.get_enemy_targets(unit_pos.tolist(), enemy_positions)
                    actions[unit]=[5,targets[0][0],targets[0][1]]
                else:
                    if self.unit_path[unit]:
                        if self.relic_map.map_possibles[self.unit_path[unit][0][0],self.unit_path[unit][0][1]]==1:
                            if discover_flag:
                                actions[unit]=[0,0,0]
                            else:
                                actions[unit] = [direction_to(unit_pos, self.unit_path[unit].pop(0)), 0, 0]
                                discover_flag=1
                        else:
                            actions[unit] = [direction_to(unit_pos, self.unit_path[unit].pop(0)), 0, 0]
                    else:
                        actions[unit]=[0,0,0]
                        
        self.relic_map.map_occupied = np.zeros((24,24))
        self.prev_points = team_points[self.team_id]
        self.prev_points_increase = increase
        self.prev_actions = actions
        self.previous_positions = unit_positions
        return actions