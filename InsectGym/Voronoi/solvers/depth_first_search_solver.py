class MazeSolverDFS:
    def __init__(self, maze):
        self.name = 'voronoi-dfs'
        self.maze = maze
        self.path = self.solve_maze_dfs()

    def solve_maze_dfs(self):
        """returns depth-first search path for matplotlib animation"""
        stack = [self.maze.start]
        visited = []
        while stack:
            current = stack.pop()
            visited.append(current)
            if current == self.maze.exit_index:
                # found the exit
                break
            neighbors = self.maze.graph[current]
            for neighbor in neighbors:
                edge = (current, neighbor)
                reverse_edge = (neighbor, current)
                if edge in self.maze.legal_maze_path_edges and neighbor not in visited:
                    stack.append(neighbor)
                elif reverse_edge in self.maze.legal_maze_path_edges and neighbor not in visited:
                    stack.append(neighbor)
        return visited
