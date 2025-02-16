from my_agent.lux.utils import direction_to, direction_to_change
import numpy as np
import random

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)


        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            # if we found at least one relic node
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                
                # if close to the relic node we want to move randomly around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            # every 20 steps or if a unit doesn't have an assigned location to explore
            else:
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    # pick a random location on the map for the unit to explore
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                # using the direction_to tool we can generate a direction that makes the unit move to the saved location
                # note that the first index of each unit's action represents the type of action. See specs for more details
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        return actions

class Agent2():
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
        self.range = self.env_cfg["unit_sensor_range"]
        self.width = self.env_cfg["map_width"]
        self.height = self.env_cfg["map_height"]
        self.explore_targets = [(self.range+1, self.height-self.range-1), 
                                (self.width-self.range-1, self.range+1), 
                                (self.width-self.start_pos[0]+self.range*(-1)**self.pnum, self.height-self.start_pos[0]+self.range*(-1)**self.pnum)
                               ]
        self.relic_targets = []
        self.explore_targets = []
        self.unit_has_target = -np.ones((self.n_units)) # -1=no target; 0=explore target; 1=relic target; 2=on relic
        self.unit_targets = dict(zip(range(0,self.n_units), np.zeros(self.n_units)))
        self.prev_points = 0
        self.prev_points_increase = 0
        self.prev_actions = None

    def direction_to_change(self, direction):
        if direction==0:
            change = [0,0]
        if direction==1:
            change = [0,-1]
        if direction==2:
            change = [1,0]
        if direction==3:
            change = [0,1]
        if direction==4:
            change = [-1,0]
        return np.array(change)
    
    def get_moves(self, obs, unit_id, unit_pos):
        prev_pos = [unit_pos[0] - self.direction_to_change(self.prev_actions[unit_id][0])[0], unit_pos[1] - self.direction_to_change(self.prev_actions[unit_id][0])[1]]
        new_pos = [[unit_pos[0], unit_pos[1]-1],
                  [unit_pos[0]+1, unit_pos[1]],
                  [unit_pos[0], unit_pos[1]+1],
                  [unit_pos[0]-1, unit_pos[1]]]
        moves = [0]
        for ii, pos in enumerate(new_pos):
            if pos[0]<0 or pos[1]<0 or pos[0]>=self.width or pos[1]>=self.height or (pos[0]==prev_pos[0] and pos[1]==prev_pos[1]) or obs["map_features"]["tile_type"][pos[0], pos[1]]==2 :
                pass
            else:
                moves.append(direction_to(unit_pos, pos))
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
                targets.append(np.array([pos[0]+i, pos[1]+j]))
        return targets
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
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
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
                self.relic_targets.extend(self.relic_to_targets(observed_relic_node_positions[id]))
        #print(self.relic_targets)
        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            if self.unit_has_target[unit_id]!=-1 and self.unit_has_target[unit_id]!=2 and unit_pos[0]==self.unit_targets[unit_id][0] and unit_pos[1]==self.unit_targets[unit_id][1]:
                if increase>self.prev_points_increase:
                    self.unit_has_target[unit_id]=2
                else:
                    self.unit_has_target[unit_id]=-1
            if self.unit_has_target[unit_id]==-1:
                # set target of unit to relic tile
                if self.relic_targets:
                    rand = np.random.randint(0,len(self.relic_targets)) ### closest relic target not random
                    dist = np.sum(np.abs(np.array(self.relic_targets)-unit_pos),axis=1)
                    target = self.relic_targets.pop(np.argmin(dist))
                    self.unit_has_target[unit_id] = 1
                
                # every 20 steps or if a unit doesn't have an assigned location to explore
                else:
                    if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                        if self.explore_targets:
                            target = self.explore_targets.pop(0)
                        else:
                            # pick a random location on the map for the unit to explore
                            rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                            target = rand_loc
                        self.unit_has_target[unit_id] = 0
                
                self.unit_targets[unit_id] = target
            #print(self.unit_has_target)
            direction = self.move_obstacle_avoid(obs, unit_id, unit_pos, direction_to(unit_pos, self.unit_targets[unit_id]))
            actions[unit_id] = [direction, 0, 0]
        #print(self.unit_has_target, "\n", self.unit_targets, "\n",)
        self.prev_points = team_points[self.team_id]
        self.prev_points_increase = increase
        self.prev_actions = actions

        return actions

class Agent3():
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
        a = np.stack((np.repeat(np.arange(24),24,axis=0).reshape((24,24)), np.repeat(np.arange(24),24,axis=0).reshape((24,24)).T),axis=2)
        self.explore_choices = a[np.sum(np.abs(a-np.array(self.start_pos)),axis=-1)<24-self.range]
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

    def sort_closest(self, targets, pos):
        distances = []
        for target in targets:
            _, d = a_star(pos, target, self.tile_map.map, np.zeros((24,24)), self.move_cost, 0)
            distances.append(d)
        return np.array(targets)[np.argsort(distances)].tolist()
        
    
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
        self.fragment_locations = self.relic_map.get_fragments(self.start_pos, own=True)
        self.fragment_locations = self.sort_closest(self.fragment_locations, self.start_pos)
        self.possible_locations = self.relic_map.get_possibles(self.start_pos, own=True)
        self.possible_locations = self.sort_closest(self.possible_locations, self.start_pos)
        self.occupied_fragments = []
        #print(self.fragment_locations, self.possible_locations)
        # TODO target closet first
        targets = self.fragment_locations+self.possible_locations
        attack_targets = self.get_attack_targets()
        self.n_explore_units = 0
        if self.match_num==2:
            self.n_explore_units = 3
        elif self.match_num==3:
            self.n_explore_units = 2
        for unit in range(self.n_units):
            if unit<len(targets):
                target = targets[unit]
                self.unit_has_target[unit] = 1
                self.unit_targets[unit] = target
                self.unit_path[unit],_ = a_star(self.start_pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                self.unit_path[unit].pop(0)
            else:
                if not attack_targets or self.n_explore_units>0:
                    target = self.get_explore(-np.ones((24,24)))
                    self.unit_has_target[unit] = 0
                    self.unit_targets[unit] = target
                    self.unit_path[unit],_ = a_star(self.start_pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                    self.unit_path[unit].pop(0)
                    self.n_explore_units -= 1
                else:#self.fragment_locations:
                    target = attack_targets[unit%len(attack_targets)]
                    self.unit_targets[unit] = target
                    self.unit_has_target[unit] = 3
                    self.unit_path[unit],_ = a_star(self.start_pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                    self.unit_path[unit].pop(0)
                    
        '''for unit in range(self.n_units):
            target = targets[unit%len(targets)]
            self.unit_has_target[unit] = 1
            self.unit_targets[unit] = target
            self.unit_path[unit],_ = a_star(self.start_pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
            self.unit_path[unit].pop(0)'''
        '''for unit, target in enumerate(self.fragment_locations + self.possible_locations):
            if unit<self.n_units:
                self.unit_has_target[unit] = 1
                self.unit_targets[unit] = target
                self.unit_path[unit],_ = a_star(self.start_pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                self.unit_path[unit].pop(0)'''
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
        fragments = self.relic_map.get_fragments(self.start_pos)
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

    def get_defend_targets(self, pos):
        #pos = self.occupied_fragments[unit%len(self.occupied_fragments)]
        surrounding = []
        distances = []
        for x in range(-2,3,1):
            for y in range(-2,3,1):
                if (x!=0 or y!=0) and pos[0]+x>=0 and pos[1]+y>=0 and pos[0]+x<=23 and pos[1]+y<=23:
                    new_pos = [pos[0]+x,pos[1]+y]
                    surrounding.append(new_pos)
                    distances.append(abs(new_pos[0]-abs(self.start_pos[0]-23))+abs(new_pos[1]-abs(self.start_pos[1]-23)))
        surrounding = np.array(surrounding)[np.argsort(np.array(distances))]
        surrounding = surrounding[:8].tolist()
        best_e = -np.inf
        best_pos = [-1,-1]
        for s in surrounding:
            if self.energy_map.map[s[0],s[1]]>best_e:
                best_e = self.energy_map.map[s[0],s[1]]
                best_pos = s
        return best_pos
        

    def get_enemy_targets(self, pos, enemy_positions, relative=True):
        targets = []
        for dx in range(-self.sap_range,self.sap_range):
            for dy in range(-self.sap_range,self.sap_range):
                if [pos[0]+dx,pos[1]+dy] in enemy_positions:
                    if relative:
                        targets.append([dx,dy])
                    else:
                        targets.append([pos[0]+dx,pos[1]+dy])
        #print(pos, enemy_positions, targets)
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
            if self.compare_positions(pos, self.unit_targets[unit]):
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
        path, _ = a_star(pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
        #print(pos, frags, dist, target, path)
        return path[1]
        '''if abs(target[0]-pos[0])>abs(target[1]-pos[1]):
            return [pos[0]+np.sign(target[0]-pos[0]), pos[1]]
        else:
            return [pos[0], pos[1]+np.sign(target[1]-pos[1])]'''

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        #print("Step: ", step)
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
                #self.relic_targets.extend(self.relic_to_targets(observed_relic_node_positions[ii]))
                self.relic_targets.extend(self.relic_map.get_possibles(self.start_pos, own=True))
                # remove duplicates from relic targets
                self.relic_targets = np.array(list({array.tobytes(): array for array in np.array(self.relic_targets)}.values())).tolist()
                #print(step, self.discovered_relic_nodes_ids, "\n", self.relic_targets, "\n",  self.unit_has_target, "\n",  self.unit_targets)
        # update maps
        available_unit_ids = np.where(unit_mask)[0].tolist()
        self.relic_map.step(unit_positions, increase)
        tile_shift = self.tile_map.update(current_tile_map)
        self.energy_map.update(current_energy_map)        

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
                    self.repath(unit_positions)
                    break
            
        if tile_shift:
            self.fragment_targets.extend(self.locked_fragment_targets)
            self.relic_targets.extend(self.locked_relic_targets)
            self.locked_relic_targets = []
            self.locked_fragment_targets = []
            self.repath(unit_positions)
            
        for unit in available_unit_ids:
            if self.prev_actions[unit][0] in [1,2,3,4] and self.compare_positions(self.previous_positions[unit], unit_positions[unit]) and self.previous_energys[unit]>self.move_cost:
                    self.unit_path[unit],_ = a_star(unit_positions[unit], self.unit_targets[unit], self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                    self.unit_path[unit].pop(0)


        #release_count = len(all_targets)
        for unit in available_unit_ids.copy():
            pos = [int(unit_positions[unit][0]), int(unit_positions[unit][1])]
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
                self.occupied_fragments.append([int(pos[0]), int(pos[1])])
                remain_units = available_unit_ids.copy()
                remain_units.remove(unit)
                for unit2 in remain_units:
                    if self.unit_targets[unit2][0]==pos[0] and self.unit_targets[unit2][1]==pos[1] and self.unit_has_target[unit]!=3:
                        self.unit_has_target[unit2] = -1
            # remove target if possible fragment has been cleared (by other unit)
            if self.unit_has_target[unit]==1 and self.relic_map.map_possibles[self.unit_targets[unit][0], self.unit_targets[unit][1]]==0 and self.relic_map.map_knowns[self.unit_targets[unit][0], self.unit_targets[unit][1]]!=1:
                self.unit_has_target[unit] = -1
            # retarget def units
            if self.unit_has_target[unit]==3 and self.compare_positions(self.unit_targets[unit],pos):
                if self.fragment_locations:
                    target = self.get_defend_targets(self.fragment_locations[unit%len(self.fragment_locations)])
                    self.unit_targets[unit] = target
                    self.unit_path[unit],_ = a_star(pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                    self.unit_path[unit].pop(0)
                else:
                    self.unit_has_target[unit] = -1
            # free unit that has been killed and reissue targets if necessary
            if self.unit_moved[unit] and self.compare_positions(pos, self.start_pos):
                self.free_target(unit, pos)
                self.unit_has_target[unit] = -1
                self.unit_moved[unit] = 0
            # free unit that's on target and not occupying fragment (NOT DONE ANYMORE TO KEEP UNIT TYPE)
            #if self.unit_has_target[unit]<2 and self.unit_targets[unit][0]==pos[0] and self.unit_targets[unit][1]==pos[1]:
                #print("unit at target", unit)
            #    self.unit_has_target[unit]=-1
            # untarget if target is asteroid, keep track of blocked targets to reuse when shift happens
            if self.tile_map.map[int(self.unit_targets[unit][0]),int(self.unit_targets[unit][1])]==2 and not self.compare_positions(pos,self.unit_targets[unit]):
            #print("blocked by asteroid")
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
                #print("remove unit", unit)
                available_unit_ids.remove(unit)
            if unit_energys[unit]<self.move_cost and not self.compare_positions(pos, self.unit_targets[unit]):
                self.free_target(unit, pos)
            #print("after ", unit, pos, self.unit_has_target[unit], self.unit_targets[unit], self.unit_path[unit])
            # if unit is available set path to nothing
            #if self.unit_has_target[unit]==-1:
            #    self.unit_path[unit] = []
            
        self.fragment_targets = self.relic_map.get_fragments(self.start_pos, own=True)
        self.relic_targets = self.relic_map.get_possibles(self.start_pos, own=True)
        #print(self.occupied_fragments, self.fragment_targets)
        for f in self.fragment_targets.copy():
            if self.tile_map.map[f[0],f[1]]==2 or f in self.occupied_fragments:
                self.fragment_targets.remove(f)
                #print(f, " removed")
        for r in self.relic_targets.copy():
            if self.tile_map.map[r[0],r[1]]==2:
                self.relic_targets.remove(r)
        #print(self.occupied_fragments, self.fragment_targets)
        #print(available_unit_ids)
        # send available units to fragments, possibles, explore targets
        positions = []
        for u in available_unit_ids:
            positions.append(unit_positions[u])
        #print(self.sort_targets(self.fragment_targets, positions))
        #print(step)
        #print(self.fragment_targets)
        #print(self.sort_targets(self.relic_targets, positions))
        #print(self.relic_targets)
        all_targets = self.sort_targets(self.fragment_targets, positions) + self.sort_targets(self.relic_targets, positions) + self.explore_targets
        target_type = [2 for i in range(len(self.fragment_targets))] + [1 for i in range(len(self.relic_targets))] + [0 for i in range(len(self.explore_targets))]
        #print(all_targets)
        for ii, goal in enumerate(all_targets):
            if available_unit_ids:
                unit, path = self.find_best_unit(goal, available_unit_ids, unit_positions)
                available_unit_ids.remove(unit)
                self.free_target(unit, unit_positions[unit])
                self.unit_path[unit] = path[1:]
                self.unit_has_target[unit] = target_type[ii]
                self.unit_targets[unit] = goal
                #try:
                if target_type[ii]==2:
                    self.fragment_targets.remove(goal)
                if target_type[ii]==1:
                    self.relic_targets.remove(goal)
                if target_type[ii]==0:
                    self.explore_targets.remove(goal)
                #except:
                #print(target_type[ii], goal, self.fragment_targets, self.relic_targets, self.explore_targets)
        
        # only keep targets that aren't exploring
        for unit in available_unit_ids.copy():
            if self.unit_has_target[unit]==0 or self.unit_has_target[unit]==3:
                if not self.compare_positions(unit_positions[unit], self.unit_targets[unit]):
                    available_unit_ids.remove(unit)

        attack_targets = self.get_attack_targets()
        # send remaining units to explore (first match) or target 
        for unit in available_unit_ids.copy():
            if not self.n_explore_units>0 and attack_targets:
                target_id, path = self.find_best_unit(unit_positions[unit], np.arange(len(attack_targets)), attack_targets)
                if path:
                    self.unit_path[unit] = path[::-1]
                self.unit_path[unit].append(attack_targets[target_id])
                self.unit_has_target[unit] = 3
                self.unit_targets[unit] = attack_targets[target_id]
            else:
                target = self.get_explore(current_tile_map)
                path, _ = a_star(unit_positions[unit], target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                available_unit_ids.remove(unit)
                self.unit_path[unit] = path[1:]
                self.unit_has_target[unit] = 0
                self.unit_targets[unit] = target
                self.n_explore_units -= 1
            
            
            '''for unit in available_unit_ids:
                if self.fragment_locations:
                    target = self.get_defend_targets(self.fragment_locations[unit%len(self.fragment_locations)])
                    self.unit_has_target[unit] = 3
                    self.unit_targets[unit] = target
                    self.unit_path[unit],_ = a_star(pos, target, self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                    self.unit_path[unit].pop(0)'''

        discover_flag = 0
        # Decide on action. Follow path, if multiple units want to move to possible fragment only let one through, if attacking fire on enemy instead of moving
        for unit in range(self.n_units):
            unit_pos = unit_positions[unit]
            if unit_mask[unit]:
                for node in self.unit_path[unit]:
                    if self.tile_map.map[node[0],node[1]]==2 or (self.nebula_drain==25 and self.tile_map.map[node[0],node[1]]==1):
                        path, _ = a_star(unit_pos, self.unit_targets[unit], self.tile_map.map, self.energy_map.map, self.move_cost, self.nebula_drain)
                        self.unit_path[unit] = path[1:]
                        break
                if unit_energys[unit]>self.sap_cost and (self.unit_has_target[unit]==3 or (self.unit_has_target[unit]==2 and self.energy_map.map[unit_pos[0],unit_pos[1]]>0)) and self.get_enemy_targets(unit_pos.tolist(), enemy_positions):
                    targets = self.get_enemy_targets(unit_pos.tolist(), enemy_positions, relative=False)
                    aim = self.preaim(targets[0])
                    actions[unit]=[5,aim[0]-unit_pos[0],aim[1]-unit_pos[1]]
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
        '''print("Step: ", step, "\n", self.unit_has_target, "\n", self.unit_targets, "\n", self.unit_path, "\n", actions)
        plt.imshow(self.relic_map.map_possibles.T)
        plt.show()
        plt.imshow(self.relic_map.map_confidence.T)
        plt.show()
        plt.imshow(self.relic_map.map_knowns.T)
        plt.show()'''
        self.previous_energys = unit_energys
        self.relic_map.map_occupied = np.zeros((24,24))
        self.prev_points = team_points[self.team_id]
        self.prev_points_increase = increase
        self.prev_actions = actions
        self.previous_positions = unit_positions
        return actions