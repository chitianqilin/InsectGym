import random
import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "          +---------+",
    "odor      | : :M: : |",
    "maggot    | : : : : |",
    "reinforcer| : : : : |",
    "          +---------+",
]


class MaggotInPetriDishEnv(discrete.DiscreteEnv):
    """
    Maggot In Petri Dish associative learning experiment
    Gerber, B., & Hendel, T. (2006). Outcome expectations drive learned behaviour in larval Drosophila. Proceedings.
    Biological Sciences / The Royal Society, 273(1604), 2965–2968. https://doi.org/10.1098/rspb.2006.3673

    Description:
    An maggot learns in learning petri dishes with different odors and reinforcers.

    "          +---------+",
    "odor      |A:A:M:O:O|",
    "maggot    | : : : : |",
    "reinforcer|F:F:F:F:F|",
    "          +---------+",

    Observations:
    training regimes:   amylacetate (AM),   1-octanol (OCT)
    odor states:        none,   amylacetate (AM),   1-octanol (OCT)
    reinforcer states:  none,   fructose,   quinine hemisulphate

    pertri dishes setting up is by combination of the odor states.
    Hence, the petri dish's state can be: AM and None, AM and OCT, None and OCT.

    For a maggot, there are 3 states of odors and 3 states of reinforcers. Treating odor states and reinforcer states as two
    independent dimensions, there are 3X3=9 combinations, hence 12 states in total.

    Odor locations:
    - 0: amylacetate (AM) or none
    - 1: an middle odors state
    - 2: 1-octanol (OCT) or none

    Actions:
    There are 5 discrete deterministic actions:
    - 0: None
    - 1: move to amylacetate (AM)
    - 2: escape from amylacetate (AM)
    - 3: move to 1-octanol (OCT)
    - 4: escape from 1-octanol (OCT)

    conditions:
    c0: s1 is None
    c1: s1 is AM
    c2: s2 is None
    c3: s2 is OCT


    Three cases

    | new  old  |  AM               |   AM              |  mid          |  None             |  None             |
    | --------- | -----             |  -----            | -----         | -----             | -----             |
    |   AM      |   a0, a1, a3, a4  |   a1              |               |                   |                   |
    |   AM      |   a2              |   a0, a3, a4      |   a1          |                   |                   |
    |   mid     |                   |   a2              |   a0, a3, a4  |   a1              |                   |
    |   None    |                   |                   |   a2          |   a0, a3, a4      |   a1              |
    |   None    |                   |                   |               |   a2              |   a0, a2, a3, a4  |

    | new  old  |  AM               |   AM              |  mid          |  OCT              |  OCT              |
    | --------- | -----             |  -----            | -----         | -----             | -----             |
    |   AM      |   a0, a1, a4      |   a1, a4          |               |                   |                   |
    |   AM      |   a2, a3          |   a0              |   a1, a4      |                   |                   |
    |   mid     |                   |   a2, a3          |   a0          |   a1, a4          |                   |
    |   OCT     |                   |                   |   a2, a3      |   a0              |   a1, a4          |
    |   OCT     |                   |                   |               |   a2, a3          |   a0, a2, a3      |

    | new  old  |  None             |   None            |  mid          |  OCT              |  OCT              |
    | --------- | -----             |  -----            | -----         | -----             | -----             |
    |   None    |   a0, a1, a2, a4  |   a4              |               |                   |                   |
    |   None    |   a3              |   a0, a1, a2,     |   a4          |                   |                   |
    |   mid     |                   |   a3              |   a0, a1, a2  |   a4              |                   |
    |   OCT     |                   |                   |   a3          |   a0, a1, a2      |   a4              |
    |   OCT     |                   |                   |               |   a3              |   a0, a1, a2, a3  |

    the whole state transition matrix:
    | new  old |   S0                           |                   S1                      |                   S2                       |                      S3                      |            S4                 |
    | -------- | -----                          | -----                                     |    -----                                   |
    |  S0      | a0; a1; a2 if c0; a3 if c2; a4 | a1 if c1; a4 if c3                        |                                            |                                              |                               |
    |  S1      | a2 if c1; a3 if c3             | a0; a1 if c0; a2 if c0; a3 if c2; a4 if c2| a1 if c1; a4 if c3                         |                                              |                               |
    |  S2      |                                | a2 if c1; a3 if c3                        | a0, a1 if c0; a2 if c0; a3 if c2; a4 if c2 | a1 if c1; a4 if c3                           |                               |
    |  S3      |                                |                                           | a2 if c1; a3 if c3                         | a0; a1 if c0; a2 if c0; a3 if c2; a4 if c2   | a1 if c1; a4 if c3            |
    |  S4      |                                |                                           |                                            | a2 if c1; a3 if c3                           | a0; a1 if c0; a2; a3; a4 if c2 |

    Summary of the rules:
    Given AM only be possible on the left side and OCT only be possible on the right side.
    if a0:
        do not move.
    if AM exists:
        if a1 and not on very left side:
            move to left.
        if a2 and not on very right side:
            move to right
    if OCT exists:
        if a3 and not on very right side:
            move to right.
        if a4 and not on very left side:
            move to left
    if Other cases:
        do not move.


    Rewards:
    Reward depends on the reinforcer the maggot is on. When the maggot is on fructose, it gets 1 as reward. When the
    maggot is on quinine hemisulphate, it gets -1 as punishment.

    state space is represented by:
        (odor state, reinforcer state, location)
    """

    metadata = {"render.modes": ["human", "ansi"]}

    # fructose, quinine
    def __init__(self, odor=['AM', 'OCT'], reinforcer=['fructose'], num_locations=5):
        optional_odor = (None, 'AM', 'OCT')
        optional_reinforcer = (None, 'fructose', 'quinine')
        for an_odor in odor:
            assert an_odor in optional_odor, an_odor + " is not an available odor in the experiment"
        for a_reinforcer in reinforcer:
            assert a_reinforcer in optional_reinforcer, a_reinforcer + " is not an available reinforcer in the experiment"

        self.num_locations = num_locations
        self.odor = odor
        self.reinforcer = reinforcer
        self.reward_options = (1, -1)
        num_states = (self.num_locations)

        initial_state_distrib = np.zeros(num_states)
        initial_state_distrib[2] = 1
        initial_state_distrib /= initial_state_distrib.sum()

        num_actions = 5
        P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }


        done = False

        if self.odor[0] != self.odor[1]:
            for a_location in range(self.num_locations):
                for an_action in range(num_actions):
                    new_location = a_location
                    reward = 0
                    if an_action == 0:
                        pass
                    elif 'AM' in self.odor:
                        if an_action == 1 and a_location != 0:
                            new_location = a_location - 1
                        elif an_action == 2 and a_location != 4:
                            new_location = a_location + 1
                        if an_action == 1:
                            if 'fructose' in self.reinforcer:
                                reward = 1
                            elif 'quinine' in self.reinforcer:
                                reward = -1
                            else:
                                reward = 0
                    elif 'OCT' in self.odor:
                        if an_action == 3 and a_location != 4:
                            new_location = a_location + 1
                        elif an_action == 4 and a_location != 0:
                            new_location = a_location - 1
                        if an_action == 3:
                            if 'fructose' in self.reinforcer:
                                reward = 1
                            elif 'quinine' in self.reinforcer:
                                reward = -1
                            else:
                                reward = 0
                    P[a_location][an_action].append((1.0, new_location, reward, done))
        if self.odor[0] == self.odor[1]:
            for a_location in range(self.num_locations):
                for an_action in range(num_actions):
                    # if an_action == 1 or an_action == 3:
                    #     if 'fructose' in self.reinforcer:
                    #         reward = 1
                    #     elif 'quinine' in self.reinforcer:
                    #         reward = -1
                    #     else:
                    #         reward = 0
                    # else:
                    #     reward = 0
                    if an_action == 1 and 'AM' in self.odor:
                        if 'fructose' in self.reinforcer:
                            reward = 1
                        elif 'quinine' in self.reinforcer:
                            reward = -1
                        else:
                            reward = 0
                    elif an_action == 3 and 'OCT' in self.odor:
                        if 'fructose' in self.reinforcer:
                            reward = 1
                        elif 'quinine' in self.reinforcer:
                            reward = -1
                        else:
                            reward = 0
                    else:
                        reward = 0
                    if an_action == 0:
                        new_location = a_location
                        P[a_location][an_action].append((1.0, new_location, reward, done))
                    else:
                        if a_location == 0:
                            P[a_location][an_action].append((0.5, a_location, reward, done))
                            new_location = a_location + 1
                            P[a_location][an_action].append((0.5, new_location, reward, done))
                        if a_location == 4:
                            new_location = a_location - 1
                            P[a_location][an_action].append((0.5, new_location, reward, done))
                            P[a_location][an_action].append((0.5, a_location, reward, done))
                        else:
                            new_location = a_location - 1
                            P[a_location][an_action].append((0.5, new_location, reward, done))
                            new_location = a_location + 1
                            P[a_location][an_action].append((0.5, new_location, reward, done))


        self.mapInit()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib
        )

    def encode(self, odor, reinforcer, location):
        return location

    def decode(self, i):
        out = [None, None, i]
        odor_letter = self.desc[1, self.layer_name_length + (i + 1) * 2]
        reinforcer_letter = self.desc[3, self.layer_name_length + (i + 1) * 2]
        if odor_letter == b'A':
            out[0] = 'AM'
        elif odor_letter == b'O':
            out[0] = 'OCT'
        elif odor_letter == b'M':
            out[0] = random.choice(self.odor)
        if reinforcer_letter == b'F':
            out[1] = 'fructose'
        elif reinforcer_letter == b'Q':
            out[1] = 'quinine'
        return out

    def mapInit(self):
        self.desc = np.asarray(MAP, dtype="c")
        self.layer_name_length = 9
        odor_0_letter = 'N'
        odor_1_letter = 'N'
        odor_mid_letter = 'M'
        if self.odor[0] == self.odor[1]:
            if self.odor[0] == 'AM':
                odor_0_letter = 'A'
                odor_1_letter = 'A'
                odor_mid_letter = 'A'
            elif self.odor[0] == 'OCT':
                odor_0_letter = 'O'
                odor_1_letter = 'O'
                odor_mid_letter = 'O'
            elif self.odor[0] is None:
                odor_0_letter = 'N'
                odor_1_letter = 'N'
                odor_mid_letter = 'N'
        else:
            if 'AM' in self.odor:
                odor_0_letter = 'A'
            if 'OCT' in self.odor:
                odor_1_letter = 'O'
        self.desc[1, self.layer_name_length + 2] = odor_0_letter
        self.desc[1, self.layer_name_length + 4] = odor_0_letter
        self.desc[1, self.layer_name_length + 6] = odor_mid_letter
        self.desc[1, self.layer_name_length + 8] = odor_1_letter
        self.desc[1, self.layer_name_length + 10] = odor_1_letter

        if 'fructose' in self.reinforcer:
            reinforcer_letter = 'F'
        elif 'quinine' in self.reinforcer:
            reinforcer_letter = 'Q'
        else:
            reinforcer_letter = 'N'
        for a_loc in range(self.num_locations):
            self.desc[3, self.layer_name_length + 2 * (a_loc + 1)] = reinforcer_letter

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        out[2][self.layer_name_length + (self.s + 1) * 2] = utils.colorize('m', "yellow", highlight=True)
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()


if __name__ == "__main__":
    odor = ['AM', 'OCT']
    reinforcer = ['fructose']
    # odor = ['OCT', 'OCT']
    # reinforcer = [None]
    env = MaggotInPetriDishEnv(odor, reinforcer)
    obs = env.reset()
    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # Render the game
        env.render()
        print(obs, reward, done, info, env.decode(obs))

        if done:
            break

    env.close()
