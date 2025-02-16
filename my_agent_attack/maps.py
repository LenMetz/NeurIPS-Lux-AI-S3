import numpy as np
import random


class RelicMap():
    '''
    Relic map keeps track of locations of relic positions, known fragments, disproven fragments and possible fragments.
    It also stores the current status for each unit in relation to it's position and fragment locations
    map: 24 x 24 game map
        -1 = unknown
        0 = disproven fragment
        1 = possible fragment
        2 = known fragment
        3 = known and occupied
    '''
    def __init__(self, n_units):
        self.map_knowns = np.zeros((24,24))
        self.map_possibles = np.zeros((24,24))
        self.map_confidence = np.zeros((24,24))
        self.unit_status = np.zeros((n_units))

    def reset(self):
        pass
        #self.map[self.map==3] = 2
        #poss = knowns = np.transpose((self.map_confidence<=0.75).nonzero())
        #poss2 = knowns = np.transpose((self.map_confidence>=0.25).nonzero())
        #for p in poss:
        #    if p.tolist() in poss2.tolist():
        #        self.map_visited[p] = 0
                
    def new_relic(self, pos):
        #patch = self.map[pos[0]-2:pos[0]+3,pos[1]-2:pos[1]+3]
        #patch[patch==-1] = 1
        for x in range(-2,3,1):
            for y in range(-2,3,1):
                if pos[0]+x>=0 and pos[0]+x<=23 and pos[1]+y>=0 and pos[1]+y<=23 and self.map_possibles[pos[0],pos[1]]!=-1:
                    self.map_possibles[pos[0]+x,pos[1]+y] = 1
                    self.map_possibles[abs(pos[1]+y-23),abs(pos[0]+x-23)] = 1
        
        #self.map_confidence[pos[0]-2:pos[0]+3,pos[1]-2:pos[1]+3] = 9/25
        #self.map_confidence[abs(pos[1]-23)-2:abs(pos[1]-23)+3,abs(pos[0]-23)-2:abs(pos[0]-23)+3] = 9/25

    def get_fragments(self):
        knowns = np.transpose((self.map_knowns==1).nonzero())
        return knowns.tolist()

    def get_possibles(self):
        poss = np.transpose((self.map_possibles==1).nonzero())
        return poss.tolist()

    def move_away(self, pos):
        moves = [1,2,3,4]
        options = np.array([[pos[0],pos[1]-1],[pos[0]+1,pos[1]],[pos[0],pos[1]+1],[pos[0]-1,pos[1]]])
        for ii, option in enumerate(options):
            if np.max(option)>23 or np.min(option)<0:
                continue
            if self.map_knowns[option[0],option[1]]==1:
                return moves[ii]
            if self.map_possibles[option[0],option[1]]==0 and self.map_knowns[option[0],option[1]]==0:
                return moves[ii]
        return np.random.randint(1,5)
        
    def step(self, unit_positions, increase):
        S = []
        F = []
        ones = 0
        rest = []
        check_knowns = self.map_knowns.copy()
        check_possibles = self.map_possibles.copy()
        for unit in unit_positions:
            if check_knowns[unit[0],unit[1]]==1:
                ones += 1
                check_knowns[unit[0],unit[1]]=0
            if check_possibles[unit[0],unit[1]]==1:
                check_possibles[unit[0],unit[1]]=0
                S.append(unit)
        r1 = increase-ones
        r2 = 0
        c_sum = 0
        if r1<=0:
            for unit in S:
                self.map_possibles[unit[0],unit[1]]=0
        else:
            for unit in S:
                self.map_confidence[unit[0],unit[1]]=max(self.map_confidence[unit[0],unit[1]],r1/len(S))
            for unit in S:
                if self.map_confidence[unit[0],unit[1]]==0:
                    self.map_possibles[unit[0],unit[1]]=-1
                    self.map_possibles[abs(unit[1]-23),abs(unit[0]-23)]=-1
                if self.map_confidence[unit[0],unit[1]]==1:
                    self.map_possibles[unit[0],unit[1]]=0
                    self.map_knowns[unit[0],unit[1]]=1
                    self.map_possibles[abs(unit[1]-23),abs(unit[0]-23)]=0
                    self.map_knowns[abs(unit[1]-23),abs(unit[0]-23)]=1
                    
        
        
class TileMap():
    def __init__(self):
        self.map = -np.ones((24,24))
        self.known = np.zeros((24,24))
    def update(self, current):
        shift = self.check_shift(current)
        self.map[current!=-1] = current[current!=-1]
        self.known[self.map!=-1] = 1
        self.mirror()
        return shift
        
    def mirror(self):
        self.map[::-1,::-1].T[self.map!=-1] = self.map[self.map!=-1]
        self.known[::-1,::-1].T[self.known==1] = self.known[self.known==1]
        
    def check_shift(self, current):
        if np.sum(1*((self.known*self.map)[current!=-1]!=(self.known*current)[current!=-1]))>0:
            map1 = (self.known*self.map)[0:23,1:24]
            map2 = (self.known*self.map)[1:24,0:23]
            mapcp = self.map.copy()
            self.map = -np.ones((24,24))
            current1 = (self.known*current)[1:24,0:23]
            current2 = (self.known*current)[0:23,1:24]
            new_known = np.zeros((24,24))
            if np.sum(1*(map1[current1!=-1]!=current1[current1!=-1]))>np.sum(1*(map2[current2!=-1]!=current2[current2!=-1])):
                self.map[0:23,1:24] = mapcp[1:24,0:23]
                new_known[0:23,1:24] = self.known[1:24,0:23]
                self.known = new_known
            else:
                self.map[1:24,0:23] = mapcp[0:23,1:24]
                new_known[1:24,0:23] = self.known[0:23,1:24]
                self.known = new_known
            return 1
        else:
            return 0

class EnergyMap():
    def __init__(self):
        self.map = np.full((24,24),1.5)
        self.known = np.zeros((24,24))
    def update(self, current):
        self.check_shift(current)
        self.map[current!=-1] = current[current!=-1]
        self.known[self.map!=1.5] = 1
        self.mirror()
        
    def mirror(self):
        self.map[::-1,::-1].T[self.map!=1.5] = self.map[self.map!=1.5]
        self.known[::-1,::-1].T[self.known==1] = self.known[self.known==1]
        
    def check_shift(self, current):
        if np.sum(1*((self.known*self.map)[current!=-1]!=(self.known*current)[current!=-1]))>0:
            self.map = np.full((24,24),1.5)
            self.known = np.zeros((24,24))
        
        