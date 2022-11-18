import os
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from InsectGym.Voronoi.voronoi_maze import VoronoiMaze
import matplotlib

class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)


Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=550)


class VoronoiMazePlot:
    """optional pass in colors_dict of colors for plot"""

    def __init__(self, maze, colors_dict=None, location=None, label_index=True, enter_index=None, exit_index=None):
        self.maze = maze
        self.maze_line_thickness = 1
        if not colors_dict:
            self.background_color = "teal"
            self.maze_line_color = "white"
            self.neighbor_line_color = "orange"
            self.point_color = "orange"
            self.start_color = "orange"
            self.exit_color = "red"
            self.polygon_face_color = "yellow"
            self.polygon_edge_color = "none"
            self.polygon_backtracking_color = "purple"
            self.polygon_backtracking_edge_color = "none"
        else:
            self.set_colors(colors_dict)
        self.name = 'voronoi-maze'

        self.fig, self.ax = self.initialize_plot()
        self.draw_voronoi()

        self.draw_maze(location, label_index=label_index, enter_index=enter_index, exit_index=exit_index)


    def set_colors(self, colors_dict):
        print("setting colors")
        self.background_color = colors_dict.get("background_color")
        self.maze_line_color = colors_dict.get("maze_line_color")
        self.neighbor_line_color = colors_dict.get("neighbor_line_color")
        self.point_color = colors_dict.get("point_color")
        self.start_color = colors_dict.get("start_color")
        self.exit_color = colors_dict.get("exit_color")
        self.polygon_face_color = colors_dict.get("polygon_face_color")
        self.polygon_edge_color = colors_dict.get("polygon_edge_color")
        self.polygon_backtracking_color = colors_dict.get("polygon_backtracking_color")
        self.polygon_backtracking_edge_color = colors_dict.get("polygon_backtracking_edge_color")
        return

    def initialize_plot(self):
        """set up matplotlib plot"""
        fig = plt.figure(figsize=(5, 5))
        ax = self.reset_axis()
        return fig, ax

    def reset_axis(self):
        """setup matplotlib axis or reset to redraw maze"""
        ax = plt.axes()
        ax.set_aspect("equal")
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        plt.gca().set_axis_off()
        plt.margins(0.009, 0.009)
        return ax

    def clear_maze(self, keep_enter_exit=True, enter=None, exit=None):
        """clears previous plot polygons for each new path solver"""
        [p.remove() for p in reversed(self.ax.patches)]
        if keep_enter_exit:
            self.draw_enter_exit(enter_index=enter, exit_index=exit)
        return

    def draw_seed_points(self, seed_thickness=0.5, label_index=False):
        px = [p[0] for p in self.maze.voronoi.points]
        py = [p[1] for p in self.maze.voronoi.points]
        self.points = self.ax.scatter(px, py, s=seed_thickness, color=self.point_color)
        self.location_labels = []
        if label_index:
            for i, location in enumerate(self.maze.voronoi.points):
                self.location_labels.append(self.ax.annotate(str(i), location, fontsize='x-small', alpha=0.4))
        return

    def clear_existing_elements(self):
        self.clear_maze(keep_enter_exit=False)
        self.points.remove()
        for a_location_label in self.location_labels:
            a_location_label.remove()
        self.location_labels=[]
        if hasattr(self, 'filled_polygon'):
            self.filled_polygon.remove()
        for a_wall_line in self.wall_lines:
            a_wall_line[0].remove()
        # self.start_polygon.remove()
        # self.exit_polygon.remove()
        # for child in self.ax.get_children():
        #     if isinstance(child, matplotlib.text.Annotation) or isinstance(child, matplotlib.collections.PathCollection) \
        #             or isinstance(child, matplotlib.patches.Polygon) or isinstance(child, matplotlib.pyplot.plot):
        #         child.remove()

    def fill_polygon(self, location):
        polygon = Polygon(self.get_polygon_points(location), True)
        polygon.set_facecolor("blue")
        polygon.set_alpha(0.4)
        self.filled_polygon = self.ax.add_patch(polygon)

    def animate_fill_polygon(self, location):
        [p.remove() for p in reversed(self.ax.patches)]
        # if hasattr(self, 'filled_polygon') and isinstance(self.filled_polygon, matplotlib.patches.Polygon):
        #     self.filled_polygon.remove()
        self.fill_polygon(location)
        return self.filled_polygon

    def clear_enter_exit(self):
        self.start_polygon.remove()
        self.exit_polygon.remove()

    def draw_enter_exit(self, enter_index=None, exit_index=None):
        [p.remove() for p in reversed(self.ax.patches)]
        """plot enter and exit cells on the graph"""
        if enter_index is None:
            self.start_polygon = Polygon(self.get_polygon_points(self.maze.start), True)
        else:
            self.start_polygon = Polygon(self.get_polygon_points(enter_index), True)
        if exit_index is None:
            exit_index = self.maze.exit_index
            # self.exit_polygon = Polygon(self.get_polygon_points(self.maze.exit), True)
        #if len(exit_index)==1:
        self.exit_polygon = Polygon(self.get_polygon_points(exit_index), True)


        self.start_polygon.set_facecolor(self.start_color)
        self.exit_polygon.set_facecolor(self.exit_color)
        self.exit_polygon.set_alpha(0.4)
        self.ax.add_patch(self.start_polygon)
        self.ax.add_patch(self.exit_polygon)
        return

    def draw_voronoi(self, plot_path='visualizations', save=False):
        """draws the full voronoi diagram"""
        self.draw_seed_points(seed_thickness=5)
        self.boundary_lines = []
        for edge in self.maze.voronoi.updated_voronoi_edges:
            first = edge[0]
            second = edge[1]
            self.boundary_lines.append(self.ax.plot([first[0], second[0]], [first[1], second[1]], color=self.maze_line_color,
                         linewidth=self.maze_line_thickness))
        self.path_lines = []
        for edge in self.maze.voronoi.graph_edges:
            first = edge[0]
            second = edge[1]
            self.path_lines.append(self.ax.plot([first[0], second[0]], [first[1], second[1]], color=self.neighbor_line_color,
                         linewidth=self.maze_line_thickness))
        if save:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            plt.savefig(os.path.join(plot_path, "initial_voronoi_diagram.png"), bbox_inches='tight', pad_inches=0.0)
        # clear axis
        plt.cla()
        return

    def draw_maze(self, location=None, plot_path="visualizations", label_index=False, save=False,
                  enter_index=None, exit_index=None):
        self.ax = self.reset_axis()
        self.draw_seed_points(label_index=label_index)
        self.wall_lines = []
        for edge in self.maze.voronoi.updated_voronoi_edges:
            first = edge[0]
            second = edge[1]
            if edge in self.maze.edges_to_remove:
                pass
            elif (second, first) in self.maze.edges_to_remove:
                pass
            else:
                self.wall_lines.append(self.ax.plot([first[0], second[0]], [first[1], second[1]], color=self.maze_line_color,
                             linewidth=self.maze_line_thickness))
        patches = []
        # self.draw_enter_exit(enter_index=enter_index, exit_index=exit_index)
        if location is not None:
            self.fill_polygon(location)
        if save:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            plt.savefig(os.path.join(plot_path, "voronoi_maze_initial.png"), bbox_inches='tight', pad_inches=0)

        return
    #
    # def draw_location_indexes(self):
    #     self.texts=[]
    #     for index in range(len(self.maze.voronoi.vor.points)-4):
    #         self.texts.append(self.ax.text(self.maze.voronoi.vor.points[index][0], self.maze.voronoi.vor.points[index][1],
    #                                   str(index),
    #                 horizontalalignment='center',
    #                 verticalalignment='center',
    #                 rotation=0,
    #                 fontsize=100,
    #                 transform=self.ax.transAxes) )

    def get_polygon_points(self, point):
        """gets points to create matplotlib Polygon object"""
        polygon_points = []
        for wall in self.maze.voronoi.cell_walls[point]:
            if wall[0] not in polygon_points:
                polygon_points.append(wall[0])
            if wall[1] not in polygon_points:
                polygon_points.append(wall[1])
        return polygon_points

    def animate(self, plot_path="visualizations"):
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        anim = animation.FuncAnimation(self.fig, self.animate_polygons, frames=len(self.path), interval=5, blit=True,
                                       repeat=False)
        anim.save(os.path.join(plot_path, 'visualizations/voronoi_animation-{}.mp4'.format(self.name)),
                  writer=writer)
        return

    def animate_polygons(self, frame):
        point = self.path[frame]
        polygon = Polygon(self.get_polygon_points(point), True)
        if point not in [self.maze.start, self.maze.exit_index]:
            if point in self.path[:frame]:
                # backtracking
                polygon.set_facecolor(self.polygon_backtracking_color)
                polygon.set_edgecolor(self.polygon_backtracking_edge_color)
                self.ax.add_patch(polygon)
            else:
                polygon.set_facecolor(self.polygon_face_color)
                polygon.set_edgecolor(self.polygon_edge_color)
                self.ax.add_patch(polygon)
        if point == self.maze.exit_index:
            polygon.set_facecolor(self.exit_color)
            polygon.set_alpha(1)
            self.ax.add_patch(polygon)
        return []
