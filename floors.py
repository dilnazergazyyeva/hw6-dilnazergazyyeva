"""Floor rules, stored separately from the solver.

To add a new floor type, add ONE entry here (and use its letter in a map).
The solver in gridworld.py never needs to change.

kind:
  "floor"    - ordinary traversable tile, deterministic movement
  "ice"      - slippery: 'intended' prob. for chosen direction, 'perp_each' for each perpendicular
  "goal"     - terminal
  "hole"     - terminal
  "wall"     - not a state, cannot be entered
"""

BASE_FLOORS = {
    "S": {"name": "start",         "kind": "floor", "reward": -1},
    ".": {"name": "ordinary floor","kind": "floor", "reward": -1},
    "M": {"name": "mud",           "kind": "floor", "reward": -3},
    "I": {"name": "ice",           "kind": "ice",   "reward": -1, "intended": 0.8, "perp_each": 0.1},
    "G": {"name": "goal",          "kind": "goal",  "reward": 10},
    "H": {"name": "hole",          "kind": "hole",  "reward": -10},
    "#": {"name": "wall",          "kind": "wall"},
}

# Part C extension: two new floors, nothing else changes.
EXTENDED_FLOORS = dict(BASE_FLOORS)
EXTENDED_FLOORS["D"] = {"name": "sand",          "kind": "floor", "reward": -2}
EXTENDED_FLOORS["W"] = {"name": "shallow water", "kind": "floor", "reward": -5}

PAPER_MAP = [".M.G",
             ".#IH",
             "S..."]
EXTENDED_MAP = [".M.G",
                ".#IH",
                "SD.W"]
