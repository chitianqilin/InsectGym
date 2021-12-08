import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+-------------+",
    "|S:X:Y: :X: :D|",
    "| :X: : :X: : |",
    "| : : : : : : |",
    "|X:X: : : :X:X|",
    "| : : : : : :M|",
    "|B: : : : : :X|",
    "+-------------+",
]


# isopen =[[1, 0, 1, 1, 0, 1, 1],
#           [1, 0, 1, 1, 0, 1, 1],
#           [1, 1, 1, 1, 1, 1, 1],
#           [0, 0, 1, 1, 1, 0, 0],
#           [1, 1, 1, 1, 1, 1, 1],
#           [1, 1, 1, 1, 1, 1, 0]]

class MultiPassengerTaxiEnv(discrete.DiscreteEnv):
    """
    The MultiPassengerTaxi Problem
    from "Dearden, R., Friedman, N. and Russell, S. (1998) ‘Bayesian Q-learning’, Proceedings of the National Conference on Artificial Intelligence, pp. 761–768."

    Description:
    There are four designated locations in the grid world indicated by (M)agenta, Y(ellow), and B(lue). When the episode starts, the taxi starts off at S. The taxi drives to the passengers locations. If the taxi is on the same square with the passenger, the passegner is picked up. Then the taxi drives to the passengers' destination D. When the taxi arrives to the destination, the goal of the task is achevied and the episode ends. The taxi can carry multiple passenges at the same time, and the reward only provides when the goal is achevied. The reward is provided according to the number of passengers the taxi carries when it arrives to the destination.

    Observations:
    There are (6*7)*2**3=336 discrete states since there are (6*7)=42 taxi positions, 2**3 locations of the passenger (including the case when the passenger is in the taxi).
    Note that there are ((6*7)-9)*2**3-(3-2*3-3)=252 states that can actually be reached during an episode. The missing states correspond to situations in which the taxi is not accessable or is on where a passenger locates but does not pick the passenger up, which is different from our assumptions.


    Passenger locations:
    - 0: Y(ellow) 0,2
    - 1: M(agenta) 4,6
    - 2: B(lue) 5,0

    Actions:
    There are 4 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west


    Rewards:
    The only reward is provided when the taxi arrives the destination with passenger(s). One passenager for 1, two passengers for 3, and three pasengers for 15.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (Y, M,and B): locations for passengers and destinations

    state space is represented by:
        (taxi_row, taxi_col, passenger_1_on, passenger_2_on, passenger_3_on)
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype="c")

        # self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs = locs = [(0, 2), (4, 6), (5, 0)]
        self.dest_loc = (0, 6)
        self.reward_options = (0, 3, 15)
        num_passenger = 3
        self.num_rows = 6
        self.num_columns = 7
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        num_states = (self.num_rows * self.num_columns) * 2 **num_passenger
        # - 0: move south
        # - 1: move north
        # - 2: move east
        # - 3: move west
        self.movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        initial_state_distrib = np.zeros(num_states)
        initial_state_distrib[self.encode(0, 0, [0, 0, 0])] = 1
        initial_state_distrib /= initial_state_distrib.sum()
        num_actions = 4
        P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for pass_on_0 in range(2):
                    for pass_on_1 in range(2):
                        for pass_on_2 in range(2):
                            state = self.encode(row, col, [pass_on_0, pass_on_1, pass_on_2])
                            temp_row, temp_col, temp_pass_on = self.decode(state)
                            assert (temp_row==row and temp_col==col and temp_pass_on==[pass_on_0, pass_on_1, pass_on_2])
                            for action in range(num_actions):
                                # defaults
                                new_row, new_col, new_pass_on_0, new_pass_on_1, new_pass_on_2 \
                                    = row, col, pass_on_0, pass_on_1, pass_on_2
                                reward = (0)  # default reward when there is no arrive
                                done = False
                                new_row = row + self.movements[action][0]
                                new_row = min(new_row, self.max_row)
                                new_row = max(new_row, 0)
                                new_col = min(col + self.movements[action][1], self.max_col)
                                new_col = max(new_col, 0)
                                if self.desc[1 + new_row, 2 * new_col + 1] == b"X":
                                    new_row = row
                                    new_col = col
                                taxi_loc = (new_row, new_col)
                                if pass_on_0 == 0:
                                    if taxi_loc == self.locs[0]:
                                        new_pass_on_0 = 1
                                if pass_on_1 == 0:
                                    if taxi_loc == self.locs[1]:
                                        new_pass_on_1 = 1
                                if pass_on_2 == 0:
                                    if taxi_loc == self.locs[2]:
                                        new_pass_on_2 = 1
                                if taxi_loc == self.dest_loc:
                                    new_num_pass_on = new_pass_on_0 + new_pass_on_1 + new_pass_on_2
                                    if new_num_pass_on > 0:
                                        done = True
                                        reward = (self.reward_options[new_num_pass_on-1])

                                new_state = self.encode(
                                    new_row, new_col, [new_pass_on_0, new_pass_on_1, new_pass_on_2]
                                )
                                P[state][action].append((1.0, new_state, reward, done))

        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib
        )

    def encode(self, taxi_row, taxi_col, pass_on):
        # 6 7, [2,2,2]
        # pass_on=[1,0,1]
        i = taxi_row
        i *= self.num_columns
        i += taxi_col
        i *= 2
        i += pass_on[0]
        i *= 2
        i += pass_on[1]
        i *= 2
        i += pass_on[2]
        assert 0 <= i
        return i

    def decode(self, i):
        pass_on = []
        out = []
        pass_on.append(i % 2)
        i = i // 2
        pass_on.append(i % 2)
        i = i // 2
        pass_on.append(i % 2)
        i = i // 2
        out.append(list(reversed(pass_on)))
        out.append(i % self.num_columns)
        i = i // self.num_columns
        out.append(i)
        assert 0 <= i < self.num_rows
        return reversed(out)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        taxi_row, taxi_col, pass_on = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x


        if sum(pass_on):  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )
        else:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )

        for i in range(len(pass_on)):
            if not pass_on[i]:
                pi, pj = self.locs[i]
                out[1 + pi][2 * pj + 1] = utils.colorize(
                    out[1 + pi][2 * pj + 1], "blue", bold=True)

        di, dj = self.dest_loc
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(
                    ["South", "North", "East", "West"][
                        self.lastaction
                    ]
                )
            )
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()

if __name__ == "__main__":
    env = MultiPassengerTaxiEnv()
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