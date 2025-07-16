import numpy as np
import random

"""
The RANDOM baseline policy. Every action is totally random.
"""

action_vectors_1d = [
            -1,
            1,
            0
        ]

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]

def random_policy(state, agent_pos):
    """
    Random policy. State and agent_pos are not considered, but included for easy replacement.

    :param state:
    :param agent_pos:
    :return: action chosen
    """
    return random.randint(0, len(action_vectors)-1)


def random_policy_1d(state, agent_pos):
    """
    Random policy in 1d. Used because the regular random policy would not move in a 1d space 60% of the time.
    """
    return random.randint(0, len(action_vectors_1d)-1)