from InsectGym.Voronoi.VoronoiMazeMultiExits import VoronoiMazeMultiExits
from InsectGym.Voronoi.VoronoiMazeMultiExitsPlots import VoronoiMazeMultiExitsPlots
from InsectGym.Voronoi.VoronoiWorld import VoronoiWorld

class VoronoiWorldMultiExits(VoronoiWorld):
    def __init__(self, colors_dict=None, multi_route_prob=0.1, plot_path=None, task_path=None):
        super(VoronoiWorldMultiExits, self).__init__()



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