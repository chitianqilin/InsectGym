from InsectGym.Voronoi.voronoi_maze_plots import VoronoiMazePlot
import numpy as np
from matplotlib.patches import Polygon

class VoronoiMazeMultiExitsPlots(VoronoiMazePlot):
    def __init__(self, maze, colors_dict=None, location=None, label_index=True, enter_index=None, exit_index=None):
        super(VoronoiMazeMultiExitsPlots, self).__init__(maze, colors_dict=colors_dict, location=location,
                                                         label_index=label_index,
                                                         enter_index=enter_index, exit_index=exit_index)
    def draw_enter_exit(self, enter_index=None, exit_index=None):
        [p.remove() for p in reversed(self.ax.patches)]
        """plot enter and exit cells on the graph"""
        if enter_index is None:
            self.start_polygon = Polygon(self.get_polygon_points(self.maze.start), True)
        else:
            self.start_polygon = Polygon(self.get_polygon_points(enter_index), True)
        self.start_polygon.set_facecolor(self.start_color)
        self.ax.add_patch(self.start_polygon)

        if exit_index is None:
            exit_index = self.maze.exits
            # self.exit_polygon = Polygon(self.get_polygon_points(self.maze.exit), True)
        self.exit_polygons = []
        for an_exit in exit_index:
            an_exit_polygon = Polygon(self.get_polygon_points(an_exit), True)
            an_exit_polygon.set_facecolor(self.exit_color)
            an_exit_polygon.set_alpha(0.4)
            self.ax.add_patch(an_exit_polygon)
            self.exit_polygons.append(an_exit_polygon)
        return

    def clear_enter_exit(self):
        self.start_polygon.remove()
        for an_exit_polygon in self.exit_polygons:
            an_exit_polygon.remove()

