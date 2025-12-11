"""
Battle of Two Armies – Custom MAgent2 Environment
Fully compatible with MAgent2 + PettingZoo
"""

from magent2.environments.battle_v4 import parallel_env as base_parallel
from magent2.environments.battle_v4 import env as base_env
import numpy as np


def gen_custom_config():
    """
    Creates the agent configuration for 'small' and 'big'.
    It modifies the attributes used internally by battle_v4.
    """

    from magent2.environments.battle_v4 import default_config

    cfg = default_config()

    # ---- MODIFY ATTRIBUTES OF GROUP 0 (SMALL) ----
    small = cfg["agents"][0]
    small["hp"] = 12
    small["speed"] = 1
    small["attack_range"] = 1
    small["view_range"] = 5
    small["damage"] = 2
    small["step_recover"] = 0.4

    # reward shaping equivalent to your code
    small["attack_reward"] = 5
    small["kill_reward"] = 0
    small["dead_penalty"] = -2
    small["attack_penalty"] = -0.01

    # ---- MODIFY ATTRIBUTES OF GROUP 1 (BIG) ----
    big = cfg["agents"][1]
    big["hp"] = 8
    big["speed"] = 2
    big["attack_range"] = 1.5
    big["view_range"] = 6
    big["damage"] = 2

    big["attack_reward"] = 0
    big["kill_reward"] = 5
    big["dead_penalty"] = 0
    big["attack_penalty"] = -0.02

    return cfg


# ------------- ENV AEC VERSION ----------------
def env(map_size=45, max_cycles=500, render_mode=None):
    cfg = gen_custom_config()
    return base_env(map_size=map_size,
                    max_cycles=max_cycles,
                    render_mode=render_mode,
                    custom_config=cfg)


# ------------- ENV PARALLEL VERSION ----------------
def parallel_env(map_size=45, max_cycles=500, render_mode=None):
    cfg = gen_custom_config()
    return base_parallel(map_size=map_size,
                         max_cycles=max_cycles,
                         render_mode=render_mode,
                         custom_config=cfg)


# ------------- API COMPAT ----------------
def raw_env(map_size=45, max_cycles=500, render_mode=None):
    return env(map_size, max_cycles, render_mode)
