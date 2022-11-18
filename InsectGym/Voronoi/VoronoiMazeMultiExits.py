import random
from InsectGym.Voronoi.voronoi_maze import VoronoiMaze
import numpy as np

class VoronoiMazeMultiExits(VoronoiMaze):
    def __init__(self, width=100, height=100, multi_route_prob=0, num_exits=2):
        super(VoronoiMazeMultiExits, self).__init__(width, height, multi_route_prob)
        self.start, self.exit = self.get_enter_exit_locations(num_exits)

    def get_enter_exit_locations(self, num_exits=1):
        """get enter and exit locations from edge points"""
        # just using first and last point for now because random points were often too close
        # nums = random.sample(range(0,len(self.voronoi.edge_side_points)-1),2)
        start = self.voronoi.edge_side_points[0]
        exit = self.voronoi.edge_side_points[-num_exits:]
        return start, exit


if __name__ == "__main__":
    env = VoronoiMazeMultiExits()
    obs = env.reset()
    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # Render the game
        env.render()

        if done:
            break

    env.close()
