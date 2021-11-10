import random
from InsectGym.Voronoi.voronoi_graph import VoronoiGraph

"""
class for generating the Voronoi diagram maze with randomized depth first search,
and solving the maze.

    -depth-first search
    -breadth-first search
    -depth-first search backtracking
    -dijkstra

"""
def max_neighbour_num(a_graph):
    max_num = 0
    for key, item in a_graph.items():
        if len(item) > max_num:
            max_num = len(item)
    return max_num

class VoronoiMaze:
    def __init__(self, width=100, height=100, multi_route_prob=0):
        self.voronoi = VoronoiGraph(width, height)
        self.graph = self.voronoi.cells
        self.path_graph = {}
        self.edges_to_remove, self.legal_maze_path_edges = \
            self.generate_maze(multi_route_prob=multi_route_prob)
        self.max_viable_neighbours = max_neighbour_num(self.path_graph)
        self.voronoi.draw_right_edges()
        self.voronoi.draw_left_edges()
        self.voronoi.draw_bottom_edges()
        self.voronoi.draw_top_edges()
        self.start, self.exit = self.get_enter_exit_locations()

    def add_path_to_graph(self, vertex1, vertex2):
        if vertex1 not in self.path_graph:
            self.path_graph[vertex1] = []
        if vertex2 not in self.path_graph[vertex1]:
            self.path_graph[vertex1].append(vertex2)
        if vertex2 not in self.path_graph:
            self.path_graph[vertex2] = []
        if vertex1 not in self.path_graph[vertex2]:
            self.path_graph[vertex2].append(vertex1)

    def generate_maze(self, multi_route_prob=0):
        """
        randomized depth first search, and returns edges to remove from the
        voronoi diagram, along with legal edges to traverse when solving
        """
        def randomized_dfs(current, visited, edges_to_remove, legal_edges):
            visited.append(current)
            neighbors = self.graph[current]
            random.shuffle(neighbors)
            for n in neighbors:
                if n not in visited:
                    legal_edges[(current, n)] = True
                    legal_edges[(n, current)] = True
                    edges_to_remove.append(self.voronoi.point_pairs_separating_edges[(current, n)])
                    self.add_path_to_graph(current, n)
                    randomized_dfs(n, visited, edges_to_remove, legal_edges)

        start = self.voronoi.points[random.randint(0, len(self.voronoi.points) - 1)]
        edges_to_remove = []
        # edges that are legal to traverse when solving the maze
        legal_traversal_edges = {}
        randomized_dfs(start, [], edges_to_remove, legal_traversal_edges)
        if multi_route_prob > 0:
            for key, item in self.graph.items():
                for n2 in item:
                    if (key, n2) not in legal_traversal_edges:
                        if random.uniform(0, 1) < multi_route_prob:
                            legal_traversal_edges[(key, n2)] = True
                            legal_traversal_edges[(n2, key)] = True
                            edges_to_remove.append(self.voronoi.point_pairs_separating_edges[(key, n2)])
                            self.add_path_to_graph(key, n2)
        return edges_to_remove, legal_traversal_edges

    def get_enter_exit_locations(self):
        """get enter and exit locations from edge points"""
        # just using first and last point for now because random points were often too close
        # nums = random.sample(range(0,len(self.voronoi.edge_side_points)-1),2)
        start = self.voronoi.edge_side_points[0]
        exit = self.voronoi.edge_side_points[-1]
        return start, exit
