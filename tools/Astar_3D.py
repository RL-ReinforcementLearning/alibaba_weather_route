import heapq
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class GridWithWeights_3D():
    def __init__(self, width, height, time_length, wall_wind, hourly_travel_distance, wind_costs, rainfall_costs, short_steps):
        self.width = width
        self.height = height
        self.time_length = time_length
        self.wall_wind = wall_wind
        self.weights = []
        self.hourly_travel_distance = hourly_travel_distance
        self.wind_costs = wind_costs
        self.rainfall_costs = rainfall_costs
        self.short_steps = short_steps

    def upper_cone(self, id):
        (x, y, t) = id
        return 0 <= x < self.width and 0 <= y < self.height and 0 < t < self.time_length

    def lower_cone(self, id):
        dist_manhantan = heuristic(id, self.goal)
        time_remain = self.time_length - id[2]
        return time_remain >= dist_manhantan

    def in_wind(self, id):
        (x, y, t) = id
        return self.weights[x, y, t] < self.wall_wind

    def neighbors(self, id):
        (x, y, t) = id
        # Voila, time only goes forward, but we can stay in the same position
        results = [(x + 1, y, t + 1), (x, y - 1, t + 1), (x - 1, y, t + 1), (x, y + 1, t + 1), (x, y, t + 1)]
        # upper cone means the space allowed to traverse from starting point
        results = filter(self.upper_cone, results)
        # lower cone means the space allowed to explore in order to reach the goal
        results = filter(self.lower_cone, results)
        # we also need within the wall wind limit
        # results = filter(self.in_wind, results)  # However, with this condition, we might never find a route
        return results

    def cost(self, to_node):
        weight_idx = to_node[0], to_node[1], (to_node[2]+self.short_steps)//self.hourly_travel_distance
        cost = self.weights[weight_idx]
        wind_cost = self.wind_costs[weight_idx]
        rainfall_cost = self.rainfall_costs[weight_idx]

        # remainder = np.mod(to_node[2], self.hourly_travel_distance)
        # if 25 <= remainder:
        #     cost += 0.1*(remainder-25)
        # elif remainder <= 5:
        #     cost += 0.1*(5-remainder)

        # remainder = np.mod(to_node[2], self.hourly_travel_distance)
        # if 25 <= remainder or remainder <= 5:
        #     if cost < 2:
        #         cost = np.exp(cost-1)

        return cost, wind_cost, rainfall_cost


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def heuristic(a, b):
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
    (x2, y2, t2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def a_star_search_3D(graph, start, goals):
    """
    :param graph:
    :param start: 3D, (x,y,0)
    :param goal: 2D, spam 3D space
    :return:
    """
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    graph.goal = goals[0]

    wind_costs = {}
    rainfall_costs = {}
    wind_costs[start] = None
    rainfall_costs[start] = None

    while not frontier.empty():
        current = frontier.get()

        # we change to in because the goals are in time span now
        if current in goals:
            break

        for next in graph.neighbors(current):
            # print(str(current) + '   ->   ' + str(next))
            costs = graph.cost(next)
            new_cost = cost_so_far[current] + costs[0]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # we use the x,y for goals because if admissible criterion is met, we are A OK!
                priority = new_cost + heuristic(goals[0], next)
                frontier.put(next, priority)
                came_from[next] = current
                wind_costs[next] = costs[1]
                rainfall_costs[next] = costs[2]

    return came_from, cost_so_far, [current], wind_costs, rainfall_costs


def dijkstra(graph, start, goals):
    """
    :param graph:
    :param start: 3D, (x,y,0)
    :param goal: 2D, spam 3D space
    :return:
    """
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    graph.goal = goals[0]

    while not frontier.empty():
        current = frontier.get()

        # we change to in because the goals are in time span now
        if current in goals:
            break

        for next in graph.neighbors(current):
            # print(str(current) + '   ->   ' + str(next))
            new_cost = cost_so_far[current] + graph.cost(next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current
    return came_from, cost_so_far, [current]


def walk_final_grid_go_to(START_STATE, came_from, final_goal_time, include_all=False):
    """
    A helper function to walk the whole grid world
    :return:
    """
    go_to_all = {}
    currentState = final_goal_time
    steps = 0
    while not currentState == START_STATE:
        # track the steps
        steps += 1
        prev = came_from[currentState]
        go_to_all[prev] = currentState
        currentState = prev

    if include_all:
        others_got_to = {}
        for k in came_from.keys():
            if k not in go_to_all.keys():
                if k not in others_got_to.keys():
                    others_got_to[k] = [came_from[k]]
                else:
                    others_got_to[k].append(came_from[k])

        go_to_all = dict(list(go_to_all.items())+list(others_got_to.items()))

    return go_to_all, steps


def convert_3D_maze_to_grid(maze, cf):
    """
    This function is mainly for
    :param maze:
    :return:
    """
    # data from main article
    diagram = GridWithWeights_3D(maze.WORLD_HEIGHT, maze.WORLD_WIDTH, maze.TIME_LENGTH, cf.wall_wind)
    diagram.weights = maze.wind_real_day_hour_total
    return diagram


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_grid_3d(graph, **style):
    """
    3D plot of a maze
    :param graph:
    :param style:
    :return:
    """

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=11, azim=-19)

    # plot the walls
    walls_idx = np.where(graph.weights >= graph.wall_wind)
    ax.scatter(walls_idx[2], walls_idx[1], graph.width-walls_idx[0], c='b', marker='^')
    ax.set_xlim(0, graph.time_length); ax.set_xlabel('Time'); ax.set_xticks(range(0, graph.time_length))
    ax.set_ylim(0, graph.height); ax.set_ylabel('Width'); ax.set_yticks(range(0, graph.height))
    ax.set_zlim(0, graph.width); ax.set_zlabel('Height'); ax.set_zticks(range(0, graph.width))
    # plot the start
    ax.scatter(style['start'][2], style['start'][1], graph.width-style['start'][0], c='r', marker='o')
    for g in style['goal']:
        ax.scatter(g[2], g[1], graph.width-g[0], c='g', marker='o')

    # plot the path
    goal_reached = tuple(set(style['goal']).intersection(set(style['came_from'].keys())))[0]
    current_state = goal_reached
    while style['came_from'][current_state]:
        x1, y1, t1 = current_state
        x2, y2, t2 = style['came_from'][current_state]
        current_state = x2, y2, t2
        x2, x1 = graph.width-x2, graph.width-x1
        a = Arrow3D([t2, t1], [y2, y1],
                    [x2, x1], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    if 'title' in style.keys():
        plt.title(style['title'])

    return

