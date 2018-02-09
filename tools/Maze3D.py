import numpy as np


class Maze_3D:
    """
    A wrapper class for a 3D maze, containing all the information about the maze.
    Maze has the third dimension t
    to other maze
    """
    def __init__(self,
                 height,
                 width,
                 time_length,
                 start_state,
                 goal_states,
                 reward_goal=0.0,
                 reward_move=-1.0,
                 maxSteps=1e5,
                 cost_matrix=None,
                 cf={},
                 start_min=0,
                 ):

        self.WORLD_WIDTH = width
        self.WORLD_HEIGHT = height
        self.TIME_LENGTH = time_length
        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTION_STAY = 4
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_STAY]
        # start state
        self.START_STATE = start_state
        # goal states
        self.GOAL_STATES = goal_states
        # all reward
        self.reward_goal = reward_goal
        self.reward_move = reward_move

        self.maxSteps = maxSteps
        self.hourly_travel_distance = cf.hourly_travel_distance
        self.start_min = start_min
        self.short_steps = start_min/2
        self.cost_matrix = cost_matrix

    def takeAction(self, state, action):
        """
        :param state:
        :param action:
        :param stochastic_wind:
        :return:
        """
        x, y, t = state
        terminal_flag = False
        # the time always goes forward
        t += 1
        if t >= self.TIME_LENGTH:
            assert "OMG this should never happened, check bug in the code!"

        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        elif action == self.ACTION_STAY:
             x, y = x, y

        # added lower cone (depending on the goal)
        dist_manhantan = self.heuristic_fn((x, y, t), self.GOAL_STATES)
        time_remain = self.TIME_LENGTH - t
        if time_remain < dist_manhantan:
            print("OMG this should never happened, check bug in the code!")
            # # we can no longer reach the goal from this point
            reward = -1 * self.reward_goal
            terminal_flag = True
            return [x, y, t], reward, terminal_flag

        # This is a bug...Di Wu should fix it, but he does not have time now...
        reward_move = self.cost_matrix[self.wind_model, x, y,
                                       min(int(t + self.short_steps) // self.hourly_travel_distance, int(self.cost_matrix.shape[-1]-1))]
        if tuple([x, y, t]) in self.GOAL_STATES:
            # We add reward move because the goal state could have wind speed larger than 13...
            reward = self.reward_goal
            terminal_flag = True
        else:
            reward = reward_move

        return [x, y, t], reward, terminal_flag

    def heuristic_fn(self, a, b):
        """
        https://en.wikipedia.org/wiki/A*_search_algorithm
         For the algorithm to find the actual shortest path, the heuristic function must be admissible,
         meaning that it never overestimates the actual cost to get to the nearest goal node.
         That's easy!
        :param a:
        :param b:
        :return:
        """

        (x1, y1, t1) = a
        (x2, y2, t2) = b[0]
        return abs(x1 - x2) + abs(y1 - y2)

    def in_bound(self, id):
        (x, y, t) = id
        return 0 <= x < self.WORLD_HEIGHT and 0 <= y < self.WORLD_WIDTH and t < self.TIME_LENGTH

    def lower_cone(self, id):
        dist_manhantan = self.heuristic_fn(id, self.GOAL_STATES)
        time_remain = self.TIME_LENGTH - 1 - id[2]
        return time_remain >= dist_manhantan

    def neighbors(self, id):
        (x, y, t) = id
        # Voila, time only goes forward, but we can stay in the same position
        # The following should be strictly follow action sequence:
        # up, down, left, right, stay
        results = [(x - 1, y, t + 1), (x + 1, y, t + 1), (x, y - 1, t + 1), (x, y + 1, t + 1), (x, y, t + 1)]
        # upper cone means the space allowed to traverse from starting point
        results = list(filter(self.in_bound, results))
        # lower cone means the space allowed to explore in order to reach the goal
        results = list(filter(self.lower_cone, results))
        # we also need within the wall wind limit
        # results = filter(self.in_wind, results)  # However, with this condition, we might never find a route
        return results

    def viable_actions(self, current_state, viable_neighbours):
        x1, y1, t1 = current_state
        viable_actions = []
        for s in viable_neighbours:
            x2, y2, t2 = s
            if x2 == x1 - 1 and y2 == y1:
                viable_actions.append(self.ACTION_UP)
            elif x2 == x1 + 1 and y2 == y1:
                viable_actions.append(self.ACTION_DOWN)
            elif x2 == x1 and y2 == y1 - 1:
                viable_actions.append(self.ACTION_LEFT)
            elif x2 == x1 and y2 == y1 + 1:
                viable_actions.append(self.ACTION_RIGHT)
            elif x2 == x1 and y2 == y1:
                viable_actions.append(self.ACTION_STAY)
            else:
                assert "Invalid action!"

        return np.array(viable_actions)

