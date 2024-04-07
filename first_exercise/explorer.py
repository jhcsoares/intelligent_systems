# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from time import sleep

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc, direction=None):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.direction = direction
        
        self.walk_stack = Stack()  # a stack to store the movements
        self.backtracking_stack = Stack()
        self.explored_coordinates = [(0, 0)]

        self.walking_time = 0
        self.first_difficulty = 0

        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
    
    def get_explored_coordinates(self):
        return self.explored_coordinates

    def add_explored_coordinate(self, coordinate):
        self.explored_coordinates.append(coordinate)

    def get_direction_delta(self):
        delta = 0

        if self.direction == "u":
            delta = 0
        elif self.direction == "r":
            delta = 2
        elif self.direction == "d":
            delta = 4
        elif self.direction == "l":
            delta = 6
        
        return delta

    def set_walk_stack(self):
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()

        explored_coordinates_list = self.get_explored_coordinates()

        delta = self.get_direction_delta()

        has_pushed = False

        for i in range(0, 8):
            index = (i + delta) % 8
            
            position = Explorer.AC_INCR[index]

            coordinates = (position[0] + self.x, position[1] + self.y)

            if obstacles[index] == VS.CLEAR and coordinates not in explored_coordinates_list:
                self.walk_stack.push(coordinates)
                has_pushed = True
        
        return has_pushed
        
    def explore(self):
        has_pushed = self.set_walk_stack()

        if has_pushed:
            xf, yf = self.walk_stack.pop()
            dx = xf - self.x
            dy = yf - self.y
            
            self.backtracking_stack.push((dx, dy))
            self.add_explored_coordinate((xf, yf))

            # Moves the body to another position
            rtime_bef = self.get_rtime()
            result = self.walk(dx, dy)

            # Test the result of the walk action
            # Should never bump, but for safe functionning let's test
            if result == VS.BUMPED:
                # update the map with the wall
                self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
                #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

            if result == VS.EXECUTED:
                # check for victim returns -1 if there is no victim or the sequential
                # the sequential number of a found victim

                #self.walk_stack.push((dx, dy))

                # update the agent's position relative to the origin
                self.x += dx
                self.y += dy          

                # Check for victims
                seq = self.check_for_victim()
                if seq != VS.NO_VICTIM:
                    vs = self.read_vital_signals()
                    self.victims[vs[0]] = ((self.x, self.y), vs)
                    print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                    #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
                
                rtime_aft = self.get_rtime()

                # Calculates the difficulty of the visited cell
                difficulty = (rtime_bef - rtime_aft)
                
                if not self.first_difficulty:
                    self.first_difficulty = difficulty

                self.walking_time += difficulty

                if dx == 0 or dy == 0:
                    difficulty = difficulty / self.COST_LINE
                else:
                    difficulty = difficulty / self.COST_DIAG
                
                # Update the map with the new cell
                self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
                #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        else:
            self.come_back()

        return
    
    def come_back(self):
        if not self.backtracking_stack.is_empty():
            dx, dy = self.backtracking_stack.pop()
            dx = dx * -1 
            dy = dy * -1

            rtime_bef = self.get_rtime()

            result = self.walk(dx, dy)

            if result == VS.BUMPED:
                print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
                return
            
            if result == VS.EXECUTED:
                # update the agent's position relative to the origin
                
                #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                self.x += dx
                self.y += dy

                rtime_aft = self.get_rtime()

                # Calculates the difficulty of the visited cell
                difficulty = (rtime_bef - rtime_aft)

                if not self.first_difficulty:
                    self.first_difficulty = difficulty

                self.walking_time += difficulty

                if dx == 0 or dy == 0:
                    difficulty = difficulty / self.COST_LINE
                else:
                    difficulty = difficulty / self.COST_DIAG

    def deliberate(self) -> bool:
        #print(f"walking_time: {self.walking_time} / rtime: {self.get_rtime()}")
        #print(self.walking_time+self.get_rtime())
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        #consumed_time = self.TLIM - self.get_rtime()
        # if consumed_time < self.get_rtime():
        
        if self.walking_time + 2*self.first_difficulty < self.TLIM/2:
            self.explore()
            return True

        # time to come back to the base
        if self.backtracking_stack.is_empty() and self.x == 0 and self.y == 0:
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
            #input(f"{self.NAME}: type [ENTER] to proceed")
            self.resc.go_save_victims(self.map, self.victims)
            
            return False

        self.come_back()
        return True
