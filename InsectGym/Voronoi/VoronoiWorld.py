import os.path

import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvas
import random
from gym import Env, spaces
from InsectGym.Voronoi.voronoi_maze import VoronoiMaze
from InsectGym.Voronoi.voronoi_maze_plots import VoronoiMazePlot
from InsectGym.Voronoi.VoronoiMazeMultiExits import VoronoiMazeMultiExits
from InsectGym.Voronoi.VoronoiMazeMultiExitsPlots import VoronoiMazeMultiExitsPlots
import json
from InsectGym.Utils.io import sterilize
import pickle

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def location_compare(loc1, loc2, allow_error=1e-10):
    return np.all(np.less(np.array(loc1) - np.array(loc2), allow_error))


def fig_to_RGB_array(fig):
    mp_canvas = FigureCanvas(fig)
    # mp_canvas.setStyleSheet("background-color:transparent;")
    mp_canvas.draw()
    return np.array(mp_canvas.renderer.buffer_rgba())


class Robot:
    def __init__(self, location, location_index=0):
        self.location = location
        self.max_fuel = 10000
        self.fuel_left = self.max_fuel
        self.step = 1
        self.location_index = location_index

    def move_to(self, new_location, location_index=None):
        self.location = new_location
        self.location_index = location_index

    def cost(self):
        self.fuel_left -= 1
        self.step += 1


class VoronoiWorld(Env):
    def __init__(self, colors_dict=None, multi_route_prob=0.1, plot_path=None, task_path=None, num_exits=1):
        super(VoronoiWorld, self).__init__()
        self.width = 100
        self.height = 100
        self.num_exits = num_exits
        if self.num_exits == 1:
            self.maze = VoronoiMaze(width=self.width, height=self.height, multi_route_prob=multi_route_prob)
        else:
            self.maze = VoronoiMazeMultiExits(width=self.width, height=self.height, multi_route_prob=multi_route_prob,
                                              num_exits=self.num_exits)
        self.locations = np.array(self.maze.voronoi.points)  # not self.maze.voronoi.vor.point because of 4 boundary points?
        self.number_of_locations = len(self.locations)
        print("max_viable_neighbours = %d" % self.maze.max_viable_neighbours)
        if not colors_dict:
            self.colors_dict = {
                "background_color": "#e0e0e0",
                "maze_line_color": "navy",
                "neighbor_line_color": "orange",
                "point_color": "orange",
                "start_color": "blue",
                "exit_color": "green",
                "polygon_face_color": "#1fc600",
                "polygon_edge_color": "none",
                "polygon_backtracking_color": "purple",
                "polygon_backtracking_edge_color": "none"
            }
        else:
            self.colors_dict = colors_dict

        # num_actions = self.maze.max_viable_neighbours
        # num_states = len(self.maze.voronoi.points)
        # P = {state: {action: [] for action in range(num_actions)}
        # 	 for state in range(num_states)}

        # Define a 2-D observation space
        self.observation_shape = (1,)
        self.observation_space = spaces.Discrete(len(self.maze.voronoi.points))

        # Define an action space according to self.maze.max_viable_neighbours
        self.action_space = spaces.Discrete(self.maze.max_viable_neighbours, )
        # if plot_path is not None:
        #     self.maze_plot = VoronoiMazePlot(self.maze, colors_dict=self.colors_dict, )
        #     self.maze_plot.draw_voronoi(plot_path=plot_path, save=True)
        #     self.maze_plot.draw_maze(plot_path=plot_path, save=True, label_index=True)
        self.init_plot_on_canvas(plot_path)
        self.reset()

        if task_path is not None:
            # self.to_JSON(task_path)
            self.pickle(task_path=task_path)
            pass

    def init_plot_on_canvas(self, plot_path=None):
        if self.num_exits == 1:
            self.maze_plot = VoronoiMazePlot(self.maze, colors_dict=self.colors_dict)
        else:
            self.maze_plot = VoronoiMazeMultiExitsPlots(self.maze, colors_dict=self.colors_dict)
        if plot_path is not None:
            self.maze_plot.draw_voronoi(plot_path=plot_path, save=True)
            self.maze_plot.draw_maze(plot_path=plot_path, save=True, label_index=True)
        self.canvas_backgroud = fig_to_RGB_array(self.maze_plot.fig)
        self.canvas = self.canvas_backgroud
        self.maze_plot.clear_existing_elements()

    def init_enter_exit_on_canvas(self):
        self.maze_plot.draw_enter_exit(enter_index=self.index_to_coordinate(self.start_location_index),
                                       exit_index=self.index_to_coordinate(self.goal_location_index))
        # self.maze_plot.draw_enter_exit(enter_index=self.start_location_index,
        #                                exit_index=self.goal_location_index)
        self.canvas_enter_exit = fig_to_RGB_array(self.maze_plot.fig)
        # self.canvas_enter_exit [self.canvas_enter_exit  > 250] = 0
        # self.canvas_backgroud_enter_exit = cv2.addWeighted(self.canvas_backgroud, 0.9, self.canvas_enter_exit, 0.3, 0)
        # self.canvas_backgroud_enter_exit = np.array(self.canvas_backgroud.astype(dtype=np.uint16) * self.canvas_enter_exit / 255).astype(dtype=np.uint8)
        self.canvas_backgroud_enter_exit = cv2.multiply(self.canvas_backgroud, self.canvas_enter_exit, scale=1.0 / 255)
        self.maze_plot.clear_enter_exit()

    def draw_location_on_canvas(self):
        # self.maze_plot.ax.clear()
        self.maze_plot.animate_fill_polygon(self.robot.location)
        forgorund = fig_to_RGB_array(self.maze_plot.fig)
        # forgorund[forgorund > 250] = 0
        # self.canvas = cv2.addWeighted(self.canvas_backgroud_enter_exit, 0.9, forgorund, 0.5, 0)
        # self.canvas = np.array(self.canvas_backgroud_enter_exit.astype(dtype=np.uint16) * forgorund/ 255).astype(dtype=np.uint8)
        self.canvas = cv2.multiply(self.canvas_backgroud_enter_exit, forgorund, scale=1.0 / 255)
        # self.canvas = self.canvas_backgroud * forgorund
        # self.canvas = self.canvas/self.canvas.max()*255

        text = 'step: {} | Fuel Left: {} | Rewards: {}'.format(self.robot.step, self.robot.fuel_left, self.reward)
        # Put the info on canvas
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font,
                                  0.8, (0, 0, 0), 1, cv2.LINE_AA)

    def reset(self):
        self.start_location_index = self.coordinate_to_index(self.maze.start)
        self.robot = Robot(location=self.maze.start, location_index=self.start_location_index)
        if self.num_exits == 1:
            self.goal_location_index = self.coordinate_to_index(self.maze.exit)
        else:
            self.goal_location_index = self.coordinates_to_indexs(self.maze.exit)

        self.goal_location = self.maze.exit
        self.reward = 0
        self.location_index = self.coordinate_to_index(self.robot.location)
        # Draw elements on the canvas
        self.init_enter_exit_on_canvas()
        self.draw_location_on_canvas()
        # return the observation
        return self.location_index

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        # Draw elements on the canvas
        self.draw_location_on_canvas()
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)
        elif mode == "rgb_array":
            return self.canvas
        else:
            super(VoronoiWorld, self).render(mode=mode)  # just raise an exception

    def close(self):
        cv2.destroyAllWindows()

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Reward for executing a step.
        self.reward = -1

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
            self.reward += 20
            done = True
        state = {'robot_location': self.robot.location, 'goal_location': self.goal_location}
        return self.robot.location_index, self.reward, done, state

    def coordinate_to_index(self, coordinate):
        # self.maze.voronoi.points
        # self.maze.voronoi.vor.point
        first_indexes = np.where(self.locations[:, 0] == coordinate[0])
        for first_index in first_indexes:
            if self.locations[first_index, 1] == coordinate[1]:
                return first_index[0]

    def coordinates_to_indexs(self, locations):
        # self.maze.voronoi.points
        # self.maze.voronoi.vor.point
        indexs = []
        for location in locations:
            indexs.append(self.coordinate_to_index(location))
        return indexs

    def index_to_coordinate(self, location_index):
        if np.isscalar(location_index):
            return self.maze.voronoi.points[location_index]
        else:
            point_list = []
            for an_index in location_index:
                point_list.append(self.maze.voronoi.points[an_index])
            return point_list

    def to_JSON(self, task_path=None):
        if task_path is not None:
            if not os.path.exists(task_path):
                os.makedirs(task_path)
            with open(os.path.join(task_path, 'VoronoiWorld.json'), 'w') as f:
                # json.dump(self, f)
                json.dump(self, f, default=sterilize, sort_keys=True, indent=4)
        else:
            return json.dumps(self, default=sterilize,
                              sort_keys=True, indent=4)

    # def save_as_JSON(self, task_path='VornoidWorld'):
    #     if not os.path.exists(task_path):
    #         os.makedirs(task_path)
    #     with open(os.path.join(task_path, 'VoronoiWorld.json'), 'w') as f:
    #         json.dump(self.to_JSON(), f)
    def pickle(self, task_path=None):
        if task_path is not None:
            if not os.path.exists(task_path):
                os.makedirs(task_path)
            with open(os.path.join(task_path, 'VoronoiWorld.pkl'), 'wb') as f:
                pickle.dump(self, f)


if __name__ == "__main__":
    env = VoronoiWorld(num_exits=2)
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
