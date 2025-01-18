from lux.utils import direction_to, direction_to_change
import numpy as np
import random
from maps import RelicMap

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
            self.pnum = 0
        else:
            self.start_pos = [self.env_cfg["map_width"], self.env_cfg["map_height"]]
            self.pnum = 1
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.n_units = self.env_cfg["max_units"]
        self.relic_map = RelicMap(self.n_units)
        self.range = self.env_cfg["unit_sensor_range"]
        self.width = self.env_cfg["map_width"]
        self.height = self.env_cfg["map_height"]
        self.explore_targets = [(self.range+1, self.height-self.range-1), 
                                (self.width-self.range-1, self.range+1), 
                                (self.width-self.start_pos[0]+self.range*(-1)**self.pnum, self.height-self.start_pos[0]+self.range*(-1)**self.pnum)
                               ]
        self.relic_targets = []
        self.explore_targets = []
        self.fragment_locations = []
        
        self.unit_has_target = -np.ones((self.n_units)) # -1=no target; 0=explore target; 1=relic target; 2=on relic
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.prev_points = 0
        self.prev_points_increase = 0
        self.prev_actions = None


    
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
                new_target = np.array([pos[0]+i, pos[1]+j])
                mirrored_target = np.abs(new_target-np.array([self.width, self.height]))
                targets.append(new_target)
                #targets.append(mirrored_target)
        return targets

    def reset(self):
        self.relic_map.reset()
        self.explore_targets = []
        self.unit_has_target = -np.ones((self.n_units)) # -1=no target; 0=explore target; 1=relic target; 2=on relic, 3=known fragment
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros((self.n_units,2))))
        self.prev_points = 0
        self.prev_points_increase = 0
        self.prev_actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        self.fragment_locations = self.relic_map.get_fragments()
        #if self.fragment_locations:
        #    frag_dist = np.sum(np.abs(np.array(self.start_pos) - np.array(self.fragment_locations)), axis=1)
        #    print(self.fragment_locations)
        #    print(frag_dist)
        #    self.fragment_locations = list(self.fragment_locations[np.argsort(frag_dist)])
        #print(self.fragment_locations)
        self.possible_locations = list(self.relic_map.get_possibles())
        free_frags = len(self.fragment_locations)
        free_pos = len(self.possible_locations)
        # TODO target closet first
        for unit_id in range (self.n_units):
            if free_frags>0:
                self.unit_has_target[unit_id] = 3
                self.unit_targets[unit_id] = self.fragment_locations[len(self.fragment_locations)-free_frags]
                free_frags -= 1
            elif free_pos>0:
                self.unit_has_target[unit_id] = 1
                self.unit_targets[unit_id] = self.possible_locations[len(self.possible_locations)-free_pos]
                free_pos -=1
        
    
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        #print("Step: ", step)
        if step in [102,203,304,405]:
            #print("Step: ", step, np.round(self.relic_map.map_possibles.T,2), np.round(self.relic_map.map_knowns.T,2))
            self.reset()
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        increase = team_points[self.team_id]-self.prev_points
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        #print(obs)
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        #print(observed_relic_nodes_mask)
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
                self.relic_targets = list({array.tobytes(): array for array in self.relic_targets}.values())
        self.relic_map.step(unit_positions, increase)
        #print("Step: ", step, np.round(self.relic_map.map_possibles.T,2), "\n", np.round(self.relic_map.map_knowns.T,2), np.round(self.relic_map.map_confidence.T,2), "\n", )
        #print("Step: ", step, np.round(self.relic_map.map_confidence.T,2))
        #print(np.round(self.relic_map.map_confidence.T,2))
        # unit ids range from 0 to max_units - 1
        
        poss = self.relic_map.get_possibles()
        knowns = self.relic_map.get_fragments()

        # set first unit in list as on known fragment
        for known in knowns:
            for ii, unit_id in enumerate(available_unit_ids):
                unit_pos = unit_positions[unit_id]
                if unit_pos.tolist()==known.tolist():
                    self.unit_targets[unit_id] = unit_pos
                    self.unit_has_target[unit_id] = 2
                    break
            
        for ii, unit_id in enumerate(available_unit_ids):
            unit_pos = unit_positions[unit_id]
            direction = None
            # remove target if not possible
            if self.relic_map.map_possibles[unit_pos[0], unit_pos[1]]==0 and self.unit_has_target[unit_id]==1:
                self.unit_has_target[unit_id] = -1
                self.unit_targets[unit_id] = np.array([-1,-1])
                
            
            # remove target if it is already targeted by other unit
            if self.unit_targets[unit_id].tolist() in [a.tolist() for a in list(self.unit_targets.values())[:ii]]:
                #print("collision")
                self.unit_has_target[unit_id] = -1
                self.unit_targets[unit_id] = np.array([-1,-1])
                
            

            # if one possible, leave with probability 1-confidence
            if self.relic_map.map_possibles[unit_pos[0], unit_pos[1]]==1:
                #print("on possible")
                self.unit_targets[unit_id] = unit_pos
                self.unit_has_target[unit_id] = 1
                draw = np.random.binomial(1,1-np.clip(self.relic_map.map_confidence[unit_pos[0], unit_pos[1]],0,1),1)
                #draw = np.random.binomial(1,1-np.clip(self.relic_map.map_confidence[*unit_pos],0,1),1)
                if draw:
                    #print("move away")
                    direction = self.relic_map.move_away(unit_pos)

            # if on target and it's neither known nor possible
            if list(unit_pos)==list(self.unit_targets[unit_id]) and self.relic_map.map_possibles[unit_pos[0], unit_pos[1]]==0 and self.relic_map.map_knowns[unit_pos[0], unit_pos[1]]==0:
                #print("on target and it's nothing")
                self.unit_has_target[unit_id] = -1
            
            if self.unit_has_target[unit_id]==-1:
                # set target of unit to relic tile
                if poss:
                    rand = np.random.randint(0,len(poss)) ### closest relic target not random
                    dist = np.sum(np.abs(np.array(poss)-unit_pos),axis=1)
                    target = poss.pop(np.argmin(dist))
                    self.unit_has_target[unit_id] = 1
                
                # every 20 steps or if a unit doesn't have an assigned location to explore
                else:
                    if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                        if self.explore_targets:
                            target = self.explore_targets.pop(0)
                        else:
                            # pick a random location on the map for the unit to explore
                            rand_loc = np.array([np.random.randint(2, self.env_cfg["map_width"])-2, np.random.randint(2, self.env_cfg["map_height"])-2])
                            target = rand_loc
                            #print(target)
                        self.unit_has_target[unit_id] = 0
                
                self.unit_targets[unit_id] = target
            #print(self.unit_targets[unit_id])
            if not direction:
                direction = self.move_obstacle_avoid(obs, unit_id, unit_pos, direction_to(unit_pos, self.unit_targets[unit_id]))
            
            #print(unit_id, self.unit_has_target[unit_id], self.unit_targets[unit_id], unit_positions[unit_id])
            actions[unit_id] = [direction, 0, 0]
        # only let one unit at a time check tile
        discover_flag = 0
        for ii, unit_id in enumerate(available_unit_ids):
            unit_pos = unit_positions[unit_id]
            if self.relic_map.map_possibles[unit_pos[0]+direction_to_change(actions[unit_id,0])[0],unit_pos[1]+direction_to_change(actions[unit_id,0])[1]]==1:
                if discover_flag:
                    actions[unit_id]=[0,0,0]
                else:
                    discover_flag=1
        #print(self.unit_has_target, "\n", self.unit_targets, "\n",)
        self.relic_map.map_occupied = np.zeros((24,24))
        self.prev_points = team_points[self.team_id]
        self.prev_points_increase = increase
        self.prev_actions = actions
        return actions