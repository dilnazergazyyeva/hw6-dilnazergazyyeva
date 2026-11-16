"""Grid World model + value iteration + simulator (pure Python, no dependencies).

Coordinates are (row, col), both starting at 1, as in the assignment.
"""
import csv
import random

ACTIONS = ["up", "right", "down", "left"]            # also the tie-break order
DELTA = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERP = {"up": ("left", "right"), "down": ("left", "right"),
        "left": ("up", "down"), "right": ("up", "down")}
ARROW = {"up": "↑", "right": "→", "down": "↓", "left": "←"}


class GridWorld:
    def __init__(self, rows, floors, gamma=0.9):
        self.rows, self.floors, self.gamma = rows, floors, gamma
        self.R, self.C = len(rows), len(rows[0])
        self.start = None
        for r in range(1, self.R + 1):
            for c in range(1, self.C + 1):
                if self.tile((r, c)) == "S":
                    self.start = (r, c)
        self.states = [(r, c) for r in range(1, self.R + 1)
                       for c in range(1, self.C + 1)
                       if self.kind((r, c)) != "wall"]
        self.nonterminal = [s for s in self.states if not self.is_terminal(s)]

    # ---- map queries -------------------------------------------------
    def inside(self, s):
        return 1 <= s[0] <= self.R and 1 <= s[1] <= self.C

    def tile(self, s):
        return self.rows[s[0] - 1][s[1] - 1]

    def kind(self, s):
        return self.floors[self.tile(s)]["kind"]

    def is_terminal(self, s):
        return self.kind(s) in ("goal", "hole")

    def reward_of(self, s_end):
        """Reward = reward of the tile where the move ENDS."""
        return self.floors[self.tile(s_end)]["reward"]

    # ---- dynamics ----------------------------------------------------
    def move(self, s, direction):
        """Deterministic result of one step; walls/boundary -> stay."""
        dr, dc = DELTA[direction]
        n = (s[0] + dr, s[1] + dc)
        if not self.inside(n) or self.kind(n) == "wall":
            return s
        return n

    def transitions(self, s, a):
        """List of (prob, next_state). Slip depends on the tile we LEAVE."""
        if self.is_terminal(s):
            return []                                   # no moves after termination
        f = self.floors[self.tile(s)]
        if f["kind"] == "ice":
            dirs = [(f["intended"], a)] + [(f["perp_each"], p) for p in PERP[a]]
        else:
            dirs = [(1.0, a)]
        return [(p, self.move(s, d)) for p, d in dirs]  # each outcome resolved separately

    def q_value(self, V, s, a):
        return sum(p * (self.reward_of(s2) + self.gamma * V[s2])
                   for p, s2 in self.transitions(s, a))

    # ---- value iteration --------------------------------------------
    def value_iteration(self, tol=1e-6, max_iter=10000):
        V = {s: 0.0 for s in self.states}               # V0 = 0 (terminals stay 0)
        history = [dict(V)]
        delta, it, converged = float("inf"), 0, False
        while it < max_iter:
            newV = dict(V)
            for s in self.nonterminal:
                newV[s] = max(self.q_value(V, s, a) for a in ACTIONS)   # old V used throughout
            delta = max(abs(newV[s] - V[s]) for s in self.states)
            V = newV
            it += 1
            history.append(dict(V))
            if delta < tol:
                converged = True
                break
        return V, it, delta, converged, history

    def extract_policy(self, V, eps=1e-9):
        """Greedy policy. Ties (within eps) -> first action in order up,right,down,left."""
        policy, ties = {}, {}
        for s in self.nonterminal:
            q = {a: self.q_value(V, s, a) for a in ACTIONS}
            best = max(q.values())
            tied = [a for a in ACTIONS if best - q[a] < eps]
            policy[s], ties[s] = tied[0], tied
        return policy, ties

    # ---- pretty printing --------------------------------------------
    def value_map(self, V, digits=3):
        lines = []
        for r in range(1, self.R + 1):
            cells = []
            for c in range(1, self.C + 1):
                s = (r, c)
                if self.kind(s) == "wall":
                    cells.append("#".center(9))
                else:
                    cells.append((f"{V[s]:.{digits}f}" + ("*" if self.is_terminal(s) else " ")).rjust(9))
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def policy_map(self, policy, ties=None):
        lines = []
        for r in range(1, self.R + 1):
            cells = []
            for c in range(1, self.C + 1):
                s = (r, c)
                t = self.tile(s)
                if self.kind(s) == "wall":
                    cells.append("  #  ")
                elif self.is_terminal(s):
                    cells.append(f"  {t}  ")
                else:
                    cells.append(f"{t}:{ARROW[policy[s]]}  ")
            lines.append(" ".join(cells))
        out = "\n".join(lines)
        if ties:
            tl = [f"  {s}: {', '.join(ties[s])}" for s in self.nonterminal if len(ties[s]) > 1]
            out += "\nTied best actions (chosen = first in up,right,down,left order):\n" + ("\n".join(tl) if tl else "  none")
        return out

    # ---- simulation --------------------------------------------------
    def run_episode(self, policy, seed, max_steps=100):
        rng = random.Random(seed)
        s, ret, disc, steps, outcome = self.start, 0.0, 1.0, 0, "time_limit"
        for _ in range(max_steps):
            a = policy[s]
            trans = self.transitions(s, a)
            u, acc, s2 = rng.random(), 0.0, trans[-1][1]
            for p, cand in trans:
                acc += p
                if u < acc:
                    s2 = cand
                    break
            ret += disc * self.reward_of(s2)
            disc *= self.gamma
            steps += 1
            s = s2
            if self.is_terminal(s):
                outcome = "goal" if self.kind(s) == "goal" else "hole"
                break
        return steps, ret, outcome

    def simulate(self, policy, map_name, batch_seeds=(11, 22, 33), n=50, max_steps=100, csv_path=None):
        rows, summary = [], []
        for bs in batch_seeds:
            res = []
            for i in range(n):
                seed = 1000 * bs + i
                steps, ret, outcome = self.run_episode(policy, seed, max_steps)
                rows.append([map_name, bs, seed, steps, round(ret, 6), outcome])
                res.append((steps, ret, outcome))
            summary.append({
                "map": map_name, "batch": bs,
                "goals": sum(o == "goal" for _, _, o in res),
                "holes": sum(o == "hole" for _, _, o in res),
                "limits": sum(o == "time_limit" for _, _, o in res),
                "mean_steps": sum(x[0] for x in res) / n,
                "mean_ret": sum(x[1] for x in res) / n})
        if csv_path:
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["map", "batch_seed", "episode_seed", "steps", "discounted_return", "outcome"])
                w.writerows(rows)
        return summary
