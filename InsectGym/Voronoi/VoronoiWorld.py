import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvas
import random
from gym import Env, spaces
from InsectGym.Voronoi.voronoi_maze import VoronoiMaze
from InsectGym.Voronoi.voronoi_maze_plots import VoronoiMazePlot

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def location_compare(loc1, loc2, allow_error=1e-10):
	return np.all(np.less(np.array(loc1) - np.array(loc2), allow_error))


def fig_to_RGB_array(fig):
	mp_canvas = FigureCanvas(fig)
	# mp_canvas.setStyleSheet("background-color:transparent;")
	mp_canvas.draw()
	return np.array(mp_canvas.renderer.buffer_rgba())


class Robot:
	def __init__(self, location):
		self.location = location
		self.max_fuel = 10000
		self.fuel_left = self.max_fuel
		self.step = 1
		self.location_index = 0

	def move_to(self, new_location, location_index=None):
		self.location = new_location
		self.location_index = location_index

	def cost(self):
		self.fuel_left -= 1
		self.step += 1


class VoronoiWorld(Env):
	def __init__(self, colors_dict=None, multi_route_prob=0.1):
		super(VoronoiWorld, self).__init__()
		self.width = 100
		self.height = 100
		self.maze = VoronoiMaze(width=self.width, height=self.height, multi_route_prob=multi_route_prob)
		self.locations = np.array(self.maze.voronoi.points)
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
		# Draw elements on the canvas
		self.init_plot_on_canvas()
		self.reset()

		self.target_location = self.coordinate_to_index(self.maze.exit)



	def init_plot_on_canvas(self):
		self.maze_plot = VoronoiMazePlot(self.maze, colors_dict=self.colors_dict)
		self.canvas_backgroud = fig_to_RGB_array(self.maze_plot.fig)
		self.canvas = self.canvas_backgroud
		self.maze_plot.clear_existing_elements()
		# self.maze_plot.ax.patch.set_alpha(0)
		# self.maze_plot.fig.patch.set_alpha(0)

	def draw_location_on_canvas(self):
		# self.maze_plot.ax.clear()
		self.maze_plot.animate_fill_polygon(self.robot.location)
		forgorund=fig_to_RGB_array(self.maze_plot.fig)
		forgorund[forgorund>250] = 0
		self.canvas = cv2.addWeighted(self.canvas_backgroud,0.9,forgorund,0.5,0)
		# self.canvas = self.canvas_backgroud * forgorund
		# self.canvas = self.canvas/self.canvas.max()*255

		text = 'step: {} | Fuel Left: {} | Rewards: {}'.format(self.robot.step, self.robot.fuel_left, self.reward)
		# Put the info on canvas
		self.canvas = cv2.putText(self.canvas, text, (10, 20), font,
								  0.8, (0, 0, 0), 1, cv2.LINE_AA)

	def reset(self):
		self.robot = Robot(location=self.maze.start)
		self.goal = random.choice(self.maze.voronoi.vor.points)  # voronoi.edge_side_points
		self.reward = 0
		self.location_index = self.coordinate_to_index(self.robot.location)
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
		if self.robot.location_index==self.target_location:
			self.reward += 20
			done = True
		self.location_index = self.coordinate_to_index(self.robot.location)
		state = {'location_index':self.location_index,'robot_location':self.robot.location, 'goal': self.goal}
		return self.location_index, self.reward, done, state

	def coordinate_to_index(self, location):
		# self.maze.voronoi.points
		# self.maze.voronoi.vor.point
		first_indexes = np.where(self.locations[:,0]==location[0])
		for first_index in first_indexes:
			if self.locations[first_index, 1]==location[1]:
				return first_index[0]


if __name__ == "__main__":
	env = VoronoiWorld()
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
