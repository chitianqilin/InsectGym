import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvas
import random
from gym import Env, spaces, GoalEnv
from InsectGym.Voronoi.voronoi_maze import VoronoiMaze
from InsectGym.Voronoi.voronoi_maze_plots import VoronoiMazePlot
from InsectGym.Voronoi.VoronoiWorld import VoronoiWorld, Robot
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

class VoronoiWorldGoal(VoronoiWorld, GoalEnv):
    def __init__(self, colors_dict=None, multi_route_prob=0.1, plot_path=None, task_path=None,
                 random_start=False, num_goals=1):
        # super(VoronoiWorldTarget, self).__init__(colors_dict=colors_dict, multi_route_prob=multi_route_prob)
        self.random_start = random_start
        super(VoronoiWorldGoal, self).__init__(colors_dict=colors_dict, multi_route_prob=multi_route_prob,
                                               plot_path=plot_path, task_path=task_path, num_exits=num_goals)
        # self.width = 100
        # self.height = 100
        # self.maze = VoronoiMaze(width=self.width, height=self.height, multi_route_prob=multi_route_prob)
        # self.locations = np.array(self.maze.voronoi.points)
        # print("max_viable_neighbours = %d" % self.maze.max_viable_neighbours)
        # if not colors_dict:
        #     self.colors_dict = {
        #         "background_color": "#e0e0e0",
        #         "maze_line_color": "navy",
        #         "neighbor_line_color": "orange",
        #         "point_color": "orange",
        #         "start_color": "blue",
        #         "exit_color": "green",
        #         "polygon_face_color": "#1fc600",
        #         "polygon_edge_color": "none",
        #         "polygon_backtracking_color": "purple",
        #         "polygon_backtracking_edge_color": "none"
        #     }
        # else:
        #     self.colors_dict = colors_dict

        # # Define a 2-D observation space
        # self.observation_shape = (1,)
        # self.observation_space = spaces.Discrete(len(self.maze.voronoi.points))
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Discrete(len(self.maze.voronoi.points)),
            achieved_goal=spaces.Discrete(len(self.maze.voronoi.points)),
            observation=spaces.Discrete(len(self.maze.voronoi.points)),
        ))
        # # Define an action space according to self.maze.max_viable_neighbours
        # self.action_space = spaces.Discrete(self.maze.max_viable_neighbours, )
        #
        # self.number_of_locations = len(self.locations)
        # self.init_plot_on_canvas()
        # self.reset()


    def step(self, action):
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"


        # apply the action to the robot
        if len(self.maze.path_graph[self.robot.location]) > action:
            new_location = self.maze.path_graph[self.robot.location][action]
            location_index = self.coordinate_to_index(new_location)
            self.robot.move_to(new_location, location_index)
        else:
            self.reward -= 0  # no punishment to hit the wall for now.
        self.robot.cost()
        # If out of fuel, end the episode.
        if self.robot.fuel_left == 0:
            done = True
        if np.any(self.robot.location_index == self.goal_location_index):
            done = True
        obs = {
            'observation': self.robot.location_index,
            'achieved_goal': self.robot.location_index,
            'desired_goal': self.goal_location_index
        }
        state = {'robot_location': self.robot.location, 'goal_location': self.goal_location, 'actual_action':action}
        self.reward = self.compute_reward(self.robot.location_index, self.goal_location_index, state)
        return obs, self.reward, done, state

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Reward for executing a step.
        reward = -1
        if len(self.maze.path_graph[info['robot_location']]) <= info['actual_action']:
            self.reward -= 0  # no punishment to hit the wall for now.
        if np.any(achieved_goal == desired_goal):
            reward += 20
        return reward

    def reset(self):
        if self.random_start:
            self.start_location_index = random.choice(range(self.number_of_locations))
        else:
            if hasattr(self, 'goal_location_index'):
                self.start_location_index = self.robot.location_index
            else:
                self.start_location_index = random.choice(range(self.number_of_locations))
        self.robot = Robot(location=self.index_to_coordinate(self.start_location_index),
                           location_index=self.start_location_index)
        self.goal_location_index = np.random.choice(len(self.locations), self.num_exits).tolist()
        self.goal_location = self.index_to_coordinate(self.goal_location_index)
        self.reward = 0
        self.init_enter_exit_on_canvas()
        self.draw_location_on_canvas()
        # return the observation
        obs = {
            'observation': self.robot.location_index,
            'achieved_goal': self.robot.location_index,
            'desired_goal': self.goal_location_index,
        }
        print(obs)
        return obs


if __name__ == "__main__":
    env = VoronoiWorldGoal(num_goals=2)
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
