# HW6 submission

## AI-use declaration

I used Claude (Anthropic) for: (1) writing the Python code (`floors.py`, `gridworld.py`, `part_b.py`, `part_c.py`); (2) checking my paper calculations in Part A (the paper work itself is handwritten by me); (3) computing the exact expected success rates for Part B as a sanity check; (4) drafting the explanations in this file. I ran all code myself, checked that the outputs match my paper calculations, and I can explain every function.

## Setup and reproduction

Windows PowerShell:

```
python -m venv .venv
.venv\Scripts\activate        (if blocked: Set-ExecutionPolicy -Scope Process Bypass)
pip install -r requirements.txt
pip freeze
python part_b.py
python part_c.py
```

Versions used: python 3.13.15, gymnasium 1.0.0, numpy 2.5.5. `pip freeze` inside `.venv` shows: cloudpickle 3.1.2, Farama-Notifications 0.0.6, gymnasium 1.0.0, numpy 2.5.5, typing_extensions 4.16.0. Part C uses only the Python standard library. Both scripts write their results into `outputs/`.

## Part A — Paper calculations

Handwritten scan (committed before any code): `outputs/photo_5465144699761925739_y.jpg`, `outputs/photo_5465144699761925740_y.jpg`, `outputs/photo_5465144699761925741_y.jpg`. Typed copy of the same calculations: [part_a_solution.md](part_a_solution.md).

Summary: V1 = -1 for every nonterminal tile except (1,3) = +10. V2(1,2) = 8 (right), V2(1,3) = 10 (right), V2(2,3) = 5.21 (up); no ties. Two rounds are not enough because V2 only looks 2 steps ahead; values keep changing until round 12.

## Part B — FrozenLake

### Fixed policy, written before the experiments

State number = row*4 + col, counting rows and columns from 0. Actions: LEFT, DOWN, RIGHT, UP.

| State | Action | Why |
| --- | --- | --- |
| 0 | DOWN | start; the route goes down the left column |
| 4 | DOWN | (1,0) -> (2,0); not RIGHT (that is hole 5) |
| 8 | RIGHT | (2,0) -> (2,1); DOWN would be hole 12 |
| 9 | RIGHT | (2,1) -> (2,2) |
| 10 | DOWN | (2,2) -> (3,2); RIGHT would be hole 11 |
| 14 | RIGHT | (3,2) -> goal 15 |
| 1, 3 | UP | off-route states, reached only by slipping; none of the three slip outcomes of UP is a hole |
| 2 | DOWN | none of its three slip outcomes is a hole; heads back toward the route |
| 6 | DOWN | between holes 5 and 7; DOWN leads to 10 |
| 13 | RIGHT | (3,1) -> 14; LEFT would be hole 12 |

It is the shortest hole-free path (0 -> 4 -> 8 -> 9 -> 10 -> 14 -> 15, 6 steps), and it is not changed between experiments.

### Results

Versions: python 3.13.15, gymnasium 1.0.0, numpy 2.5.5.

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 0 | 50 | 0 | 0 | 7.98 |
| no | random | 22 | 0 | 50 | 0 | 0 | 6.42 |
| no | random | 33 | 1 | 49 | 0 | 2 | 6.66 |
| no | fixed | 11 | 50 | 0 | 0 | 100 | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100 | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100 | 6.00 |
| yes | random | 11 | 0 | 50 | 0 | 0 | 8.30 |
| yes | random | 22 | 2 | 48 | 0 | 4 | 7.76 |
| yes | random | 33 | 0 | 50 | 0 | 0 | 6.74 |
| yes | fixed | 11 | 6 | 44 | 0 | 12 | 7.28 |
| yes | fixed | 22 | 2 | 48 | 0 | 4 | 9.50 |
| yes | fixed | 33 | 2 | 48 | 0 | 4 | 7.20 |

Pooled over 150 episodes per condition: non-slippery random 1/150 (0.7%), non-slippery fixed 150/150 (100%), slippery random 2/150 (1.3%), slippery fixed 10/150 (6.7%).

Sanity check: the exact probabilities from the FrozenLake rules (Markov-chain calculation) are about 1.4% (random, both modes), 100% (fixed, not slippery) and 6.3% (fixed, slippery). My measured rates are close to these.

Files: [episode rows](outputs/part_b_episodes.csv), [summary](outputs/part_b_summary.md), [traces](outputs/part_b_traces.txt).

Example traces: for the fixed policy with slipping, both a successful and a failed episode (from batch 11) are in `part_b_traces.txt`; the non-slippery fixed policy has a successful trace. The non-slippery fixed policy never failed, and the random policy had no goal in batch 11 (in either mode), so there are no such traces. No episode reached the 100-step limit (time limits are 0 everywhere).

### Explanation

* **What slipping changed:** without slipping, the fixed policy walks the same 6-step path every time and reaches the goal in all 150 episodes. With slipping, each move goes in the intended direction only 1/3 of the time and sideways 2/3 of the time, so the same policy ends in a hole most of the time (success fell from 100% to 6.7%). The random policy is not hurt (0.7% vs 1.3%, both near 1.4%), because a random action followed by a random slip is still a random action.
* **Consistency of batches:** non-slippery batches are identical (deterministic). Slippery fixed-policy batches gave 6, 2 and 2 goals out of 50 (12%, 4%, 4%), so single batches of 50 vary a lot; the pooled 6.7% is a better estimate than any one batch.
* **Why one successful episode is not enough:** with a success probability of about 6%, one lucky goal can happen by chance. Many episodes over several batches are needed to estimate the probability and its spread.
* **Why this is not training:** the policies never change after seeing rewards and nothing is learned or updated; we only evaluate fixed behaviour.

## Part C — Custom solver

### Compare code with paper

Code printed V1 = -1 everywhere except (1,3) = 10; V2(1,2) = 8, V2(1,3) = 10, V2(2,3) = 5.21 (Q-values: up 5.21, right -7.39, down -2.71, left -0.91). **They match my paper calculation exactly** (an `assert` in `part_c.py` checks it); no discrepancies to correct.

### Values, policies and convergence

* Paper map: [outputs/part_c_paper_map.txt](outputs/part_c_paper_map.txt) — converged in **12 iterations**, final change 2.17e-07 (< 1e-6; safeguard 10,000 not reached).
* Extended map: [outputs/part_c_extended_map.txt](outputs/part_c_extended_map.txt) — **11 iterations**, final change 2.41e-07.
* **Tie-breaking:** if several actions are equal within 1e-9, the first in the order up, right, down, left is chosen. No ties occurred in the final policies.

Final values (paper map):
```
 4.200   8.000  10.000   G
 2.780     #     5.824   H
 1.536   2.818   4.242  2.818
```
Policy: row1 → → → G; row2 ↑ # ↑ H; row3 → → ↑ ←.

Final values (extended map):
```
 4.200   8.000  10.000   G
 2.780     #     5.824   H
 1.502   2.818   4.242  2.818
```
Policy: row1 → → → G; row2 ↑ # ↑ H; row3 ↑ → ↑ ←.

### Simulation results (3 batches × 50 episodes, seeds 1000·bs+i, 100-step limit)

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 42 | 8 | 0 | 4.98 | 0.8913 |
| paper | 22 | 46 | 4 | 0 | 5.04 | 1.9148 |
| paper | 33 | 48 | 2 | 0 | 5.04 | 2.4921 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

Episodes: [paper](outputs/part_c_episodes_paper.csv), [extended](outputs/part_c_episodes_extended.csv).

### Explain your results

* **What changed with sand and water:** only V(start) changed (1.536 → 1.502) and the action at S (right → up). The cheap bottom route now begins with sand (-2) instead of ordinary floor (-1) and passes the ice tile; the left/top route (up, up, right through mud, right, right) became better. Values of tiles that do not lead through D or W stayed the same; W (-5) is never visited because nobody wants to enter it.
* **Cost vs risk:** on the paper map the ice route is slightly better (1.536 vs 1.502 for the mud route) even though it carries about an 11% chance of falling into the hole from the ice tile (0.1 per attempt, retry after a wall bump). With sand the ice route loses the cost comparison, so the policy switches to the fully deterministic route that pays 3 for mud but has zero risk. Result: 0 holes in 150 extended episodes versus 14 holes in 150 paper episodes.
* **Ice:** from the ice tile (2,3) the policy goes up (towards the goal). Up is the best action because a left-slip hits the wall (harmless) and only a right-slip is lethal (0.1); "right" would be fatal 80% of the time.
* **Simulation vs values:** V is the expected discounted return of an infinite horizon; a simulation is a finite sample average (150 episodes, 100-step cap; 0.9^100 ≈ 3e-5, so the cap hardly matters here). The extended map is deterministic along the chosen route, so every episode returns exactly 1.502 = V(start). On the paper map the average of the three batches is 1.766 versus V = 1.536: batch means jump between 0.89 and 2.49 because a single hole changes the return by about 13 points; the difference (0.23) is smaller than one standard error (≈0.33 for 150 episodes), so it is just sampling noise.

### Rule checks

[outputs/part_c_rule_checks.txt](outputs/part_c_rule_checks.txt): 28 checks, all PASS — boundary and wall bumps stay in place with the reward of the tile where the move ends (wall bump on mud costs -3), ice probabilities 0.8/0.1/0.1 sum to 1 for every action and each outcome is resolved separately against walls, rewards M=-3, D=-2, W=-5, ., S, I = -1, G=+10, H=-10, no transitions out of terminal states, terminal values 0, walls are not states.

### Adding a floor
Floor rules live in `floors.py` (a dict). `D` and `W` were added with two lines there; `gridworld.py` (value iteration) was not changed.



