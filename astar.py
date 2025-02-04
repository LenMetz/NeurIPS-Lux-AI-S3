import numpy as np

def reconstruct_path(came_from, came_from_flag, current):
    total_path = [current]
    while came_from_flag[current[0], current[1]]==1:
        current = [int(came_from[current[0],current[1],0]),int(came_from[current[0],current[1],1])]
        total_path.insert(0,current)
    return total_path

def h(start, goal, move_cost):
    steps = abs(goal[0]-start[0]) + abs(goal[1]-start[0])
    return steps*move_cost

def get_neighbors(node):
    neighbors = [[node[0],node[1]-1],
                [node[0]+1,node[1]],
                [node[0],node[1]+1],
                [node[0]-1,node[1]],]
    neighbors = np.unique(np.clip(np.array(neighbors),0,23),axis=0)
    neighbors = neighbors[(neighbors[:,:]!=np.array(node)).any(1)]
    return neighbors.tolist()
    
def find_lowest(f_score, open_set):
    index = None
    lowest = 1e20
    for ii, node in enumerate(open_set):
        if f_score[node[0],node[1]]<lowest:
            index = ii
            lowest = f_score[node[0],node[1]]
    return index
    
def a_star(start, goal, tile_map, energy_map, frag_map, move_cost, nebula_drain, use_energy=True, budget=100):
    asteroids = np.zeros((24,24))
    asteroids[tile_map==2] = asteroids[tile_map==2] + 1e6
    nebulas = np.zeros((24,24))
    nebulas[tile_map==1] = nebulas[tile_map==1] + nebula_drain
    #energy = -energy_map + np.full((24,24),np.max(energy_map))
    energy_max = np.max(energy_map)
    moves = np.full((24,24), move_cost)
    moves[frag_map==1] = 0
    budget = budget
    def d_move(pos):
        return asteroids[pos[0],pos[1]] + moves[pos[0],pos[1]]
    def d_energy(pos, adjust):
        if adjust:
            return nebulas[pos[0],pos[1]] - energy_map[pos[0],pos[1]] + energy_max + nebulas[pos[0],pos[1]]
        else:
            return nebulas[pos[0],pos[1]] - energy_map[pos[0],pos[1]] + nebulas[pos[0],pos[1]]

    def d(pos, use_energy):
        if use_energy:
            return d_move(pos) + d_energy(pos, use_energy)
        else:
            return d_move(pos)
            
    open_set = [start]
    came_from = -np.ones((24,24,2))
    came_from_flag = np.zeros((24,24))
    #came_from_flag[start[0], start[1]] = 1
    
    g_score = np.full((24,24),np.inf)
    g_score[start[0],start[1]] = 0    
    g_score_budget = np.full((24,24),np.inf)
    g_score_budget[start[0],start[1]] = 0
    
    f_score = np.full((24,24),np.inf)
    f_score[start[0],start[1]] = h(start, goal, move_cost)
    #if not use_energy:
    #    print("start")
    while open_set:
        lowest = find_lowest(f_score, open_set)
        current = open_set.pop(lowest)
        #print(current)
        if current[0]==goal[0] and current[1]==goal[1]:
            #if not use_energy:
            #    print("end succ", g_score_budget[current[0],current[1]], g_score[current[0],current[1]])
            return reconstruct_path(came_from, came_from_flag, current), g_score[current[0], current[1]]
        for neighbor in get_neighbors(current):
            temp_g_score = g_score[current[0],current[1]] + d([neighbor[0],neighbor[1]], use_energy=use_energy)
            if temp_g_score<g_score[neighbor[0],neighbor[1]]:
                came_from[neighbor[0],neighbor[1],0] = current[0]
                came_from[neighbor[0],neighbor[1],1] = current[1]
                came_from_flag[neighbor[0],neighbor[1]] = 1
                g_score[neighbor[0],neighbor[1]] = temp_g_score
                if not use_energy:
                    g_score_budget[neighbor[0],neighbor[1]] = g_score_budget[current[0],current[1]] + d_move([neighbor[0],neighbor[1]]) + d_energy([neighbor[0],neighbor[1]], False)
                    #print(g_score_budget[neighbor[0],neighbor[1]], g_score[neighbor[0],neighbor[1]])
                f_score[neighbor[0],neighbor[1]] = temp_g_score + h(neighbor,goal,move_cost)
                if not use_energy and g_score_budget[neighbor[0],neighbor[1]]>budget:
                    #print("out of energy")
                    pass
                else:
                    if neighbor not in open_set:
                        open_set.append(neighbor)
    return [[0,0]], 1e10 