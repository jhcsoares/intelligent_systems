##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position

import os
import random
import math
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from time import sleep
import heapq


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance from start node
        self.h = 0  # Heuristic - estimated distance from current node to end node
        self.f = 0  # Total cost (g + h)
        self.dx = 0
        self.dy = 0
        self.vic_seq = -1
        self.difficulty = 0
        self.step_cost = 0

    def __lt__(self, other):
        return self.f < other.f


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = None  # explorer will pass the map
        self.victims = None  # list of found victims
        self.plan = []  # a list of planned actions
        self.plan_x = 0  # the x position of the rescuer during the planning phase
        self.plan_y = 0  # the y position of the rescuer during the planning phase
        self.plan_visited = set()  # positions already planned to be visited
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.comeback_plan_walk_time = 0.0
        self.comeback_plan = []
        self.plan_walk_time = 0.0  # previewed time to walk during rescue
        self.x = 0  # the current x position of the rescuer when executing the plan
        self.y = 0  # the current y position of the rescuer when executing the plan

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def go_save_victims(self, map, victims_sequence, unified_victims_map):
        """The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        # print(f"\n\n*** R E S C U E R ***")
        self.map = map  # coordinate as key
        self.victims = victims_sequence
        self.unified_victims_map = (
            unified_victims_map  # victim_id: unified_victims_map[1]
        )

        # print the found victims - you may comment out
        # for seq, data in self.victims.items():
        #    coord, vital_signals = data
        #    x, y = coord
        #    print(f"{self.NAME} Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")

        # print(f"{self.NAME} time limit to rescue {self.plan_rtime}")

        self.__planner()

        self.set_state(VS.ACTIVE)

    def a_star(self):
        open_list = []
        closed_dict = {}

        start_position = (0, 0)
        start_node = Node(start_position)
        heapq.heappush(open_list, start_node)

        victim_location = self.unified_victims_map[self.victims.pop(0)][0]
        end_position = victim_location

        while open_list:
            current_node = heapq.heappop(open_list)
            current_position = current_node.position

            if current_position in closed_dict:
                continue

            closed_dict[current_position] = current_node.g

            if current_position == end_position:
                reverse_node = current_node
                reverse_path = []
                while reverse_node:
                    if reverse_node.vic_seq == VS.NO_VICTIM:
                        reverse_path.append((reverse_node.dx, reverse_node.dy, False))
                        self.plan_walk_time += reverse_node.step_cost
                        self.plan_rtime -= reverse_node.step_cost
                    else:
                        reverse_path.append((reverse_node.dx, reverse_node.dy, True))
                        self.plan_walk_time += (
                            reverse_node.step_cost + self.COST_FIRST_AID
                        )
                        self.plan_rtime -= reverse_node.step_cost

                    reverse_node = reverse_node.parent

                self.plan_walk_time += current_node.g

                return_path = self.calculate_return_path(current_node.position)

                if return_path:
                    if self.plan_walk_time + self.comeback_plan_walk_time >= self.TLIM:
                        self.plan.extend(self.comeback_plan)
                        return
                    else:
                        self.plan.extend(reverse_path[::-1])
                        self.comeback_plan = return_path

                if not self.victims:
                    self.plan.extend(self.comeback_plan)
                    return

                victim_location = self.unified_victims_map[self.victims.pop(0)][0]
                end_position = victim_location

                open_list = []
                closed_dict = {}
                heapq.heappush(open_list, Node(current_node.position))
                continue

            _, _, actions_res = self.map.get(current_position)

            for i, ar in enumerate(actions_res):
                if ar != VS.CLEAR:
                    continue

                dx, dy = Rescuer.AC_INCR[i]
                neighbor_position = (
                    current_position[0] + dx,
                    current_position[1] + dy,
                )

                if not self.map.get(neighbor_position):
                    continue

                difficulty, vic_seq, _ = self.map.get(neighbor_position)

                step_cost = (
                    self.COST_LINE * difficulty
                    if dx == 0 or dy == 0
                    else self.COST_DIAG * difficulty
                )

                g_cost = current_node.g + step_cost
                if (
                    neighbor_position in closed_dict
                    and g_cost >= closed_dict[neighbor_position]
                ):
                    continue

                h_cost = (neighbor_position[0] - end_position[0]) ** 2 + (
                    neighbor_position[1] - end_position[1]
                ) ** 2
                f_cost = g_cost + h_cost

                new_node = Node(neighbor_position, current_node)
                new_node.g = g_cost
                new_node.h = h_cost
                new_node.f = f_cost
                new_node.dx = dx
                new_node.dy = dy
                new_node.difficulty = difficulty
                new_node.vic_seq = vic_seq
                new_node.step_cost = step_cost

                heapq.heappush(open_list, new_node)

        return None  # No path found

    def calculate_return_path(self, start_position):
        """A* search from start_position to (0, 0)"""
        self.comeback_plan_walk_time = 0.0

        open_list = []
        closed_dict = {}

        start_node = Node(start_position)
        heapq.heappush(open_list, start_node)

        end_position = (0, 0)

        while open_list:
            current_node = heapq.heappop(open_list)
            current_position = current_node.position

            if current_position in closed_dict:
                continue

            closed_dict[current_position] = current_node.g

            if current_position == end_position:
                reverse_node = current_node
                reverse_path = []
                while reverse_node:
                    self.comeback_plan_walk_time += reverse_node.step_cost
                    reverse_path.append((reverse_node.dx, reverse_node.dy, False))
                    reverse_node = reverse_node.parent
                return reverse_path[::-1]

            _, _, actions_res = self.map.get(current_position)

            for i, ar in enumerate(actions_res):
                if ar != VS.CLEAR:
                    continue

                dx, dy = Rescuer.AC_INCR[i]
                neighbor_position = (
                    current_position[0] + dx,
                    current_position[1] + dy,
                )

                if not self.map.get(neighbor_position):
                    continue

                difficulty, _, _ = self.map.get(neighbor_position)

                step_cost = (
                    self.COST_LINE * difficulty
                    if dx == 0 or dy == 0
                    else self.COST_DIAG * difficulty
                )

                g_cost = current_node.g + step_cost
                if (
                    neighbor_position in closed_dict
                    and g_cost >= closed_dict[neighbor_position]
                ):
                    continue

                h_cost = (neighbor_position[0] - end_position[0]) ** 2 + (
                    neighbor_position[1] - end_position[1]
                ) ** 2
                f_cost = g_cost + h_cost

                new_node = Node(neighbor_position, current_node)
                new_node.g = g_cost
                new_node.h = h_cost
                new_node.f = f_cost
                new_node.dx = dx
                new_node.dy = dy
                new_node.step_cost = step_cost

                heapq.heappush(open_list, new_node)

        return None  # No path found

    def __depth_search(self, actions_res):
        enough_time = True
        ##print(f"\n{self.NAME} actions results: {actions_res}")

        for i, ar in enumerate(actions_res):
            if ar != VS.CLEAR:
                ##print(f"{self.NAME} {i} not clear")
                continue

            # planning the walk
            dx, dy = Rescuer.AC_INCR[i]  # get the increments for the possible action
            target_xy = (self.plan_x + dx, self.plan_y + dy)

            # checks if the explorer has not visited the target position
            if not self.map.in_map(target_xy):
                ##print(f"{self.NAME} target position not explored: {target_xy}")
                continue

            # checks if the target position is already planned to be visited
            if target_xy in self.plan_visited:
                ##print(f"{self.NAME} target position already visited: {target_xy}")
                continue

            # Now, the rescuer can plan to walk to the target position
            self.plan_x += dx
            self.plan_y += dy
            difficulty, vic_seq, next_actions_res = self.map.get(
                (self.plan_x, self.plan_y)
            )
            # print(f"{self.NAME}: planning to go to ({self.plan_x}, {self.plan_y})")

            if dx == 0 or dy == 0:
                step_cost = self.COST_LINE * difficulty
            else:
                step_cost = self.COST_DIAG * difficulty

            # print(f"{self.NAME}: difficulty {difficulty}, step cost {step_cost}")
            # print(f"{self.NAME}: accumulated walk time {self.plan_walk_time}, rtime {self.plan_rtime}")

            # check if there is enough remaining time to walk back to the base
            if self.plan_walk_time + step_cost > self.plan_rtime:
                enough_time = False
                # print(f"{self.NAME}: no enough time to go to ({self.plan_x}, {self.plan_y})")

            if enough_time:
                # the rescuer has time to go to the next position: update walk time and remaining time
                self.plan_walk_time += step_cost
                self.plan_rtime -= step_cost
                self.plan_visited.add((self.plan_x, self.plan_y))

                if vic_seq == VS.NO_VICTIM:
                    self.plan.append((dx, dy, False))  # walk only
                    # print(f"{self.NAME}: added to the plan, walk to ({self.plan_x}, {self.plan_y}, False)")

                if vic_seq != VS.NO_VICTIM:
                    # checks if there is enough remaining time to rescue the victim and come back to the base
                    if self.plan_rtime - self.COST_FIRST_AID < self.plan_walk_time:
                        print(f"{self.NAME}: no enough time to rescue the victim")
                        enough_time = False
                    else:
                        self.plan.append((dx, dy, True))
                        # print(f"{self.NAME}:added to the plan, walk to and rescue victim({self.plan_x}, {self.plan_y}, True)")
                        self.plan_rtime -= self.COST_FIRST_AID

            # let's see what the agent can do in the next position
            if enough_time:
                self.__depth_search(
                    self.map.get((self.plan_x, self.plan_y))[2]
                )  # actions results
            else:
                return

        return

    def __planner(self):
        """A private method that calculates the walk actions in a OFF-LINE MANNER to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        """ This plan starts at origin (0,0) and chooses the first of the possible actions in a clockwise manner starting at 12h.
        Then, if the next position was visited by the explorer, the rescuer goes to there. Otherwise, it picks the following possible action.
        For each planned action, the agent calculates the time will be consumed. When time to come back to the base arrives,
        it reverses the plan."""

        # This is a off-line trajectory plan, each element of the list is a pair dx, dy that do the agent walk in the x-axis and/or y-axis.
        # Besides, it has a flag indicating that a first-aid kit must be delivered when the move is completed.
        # For instance (0,1,True) means the agent walk to (x+0,y+1) and after walking, it leaves the kit.

        self.a_star()
        # self.plan_visited.add(
        #     (0, 0)
        # )  # always start from the base, so it is already visited
        # difficulty, vic_seq, actions_res = self.map.get((0, 0))
        # self.__depth_search(actions_res)

        # push actions into the plan to come back to the base
        if self.plan == []:
            return

        # come_back_plan = []

        # for a in reversed(self.plan):
        #     # triple: dx, dy, no victim - when coming back do not rescue any victim
        #     come_back_plan.append((a[0] * -1, a[1] * -1, False))

        # self.plan.extend(come_back_plan)

    def deliberate(self) -> bool:
        """This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do"""
        # sleep(0.25)
        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
            return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy, there_is_vict = self.plan.pop(0)

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            if there_is_vict:
                rescued = self.first_aid()  # True when rescued
                if rescued:
                    pass
                else:
                    pass

        else:
            pass

        return True
