# Part A — paper solution (copy this BY HAND onto paper, photograph it, commit the photo BEFORE the code)

Map (row, col), rewards for ending a move on a tile: S,. = -1, M = -3, I = -1, G = +10, H = -10, # wall.

```
(1,1) .   (1,2) M   (1,3) .   (1,4) G
(2,1) .   (2,2) #   (2,3) I   (2,4) H
(3,1) S   (3,2) .   (3,3) .   (3,4) .
```

## 1. MDP in words
* **States**: the 11 non-wall tiles; G and H are terminal (value 0, episode ends). The wall is not a state.
* **Actions**: up, right, down, left (in every non-terminal tile).
* **Transitions**: on S, ., M the chosen direction happens with prob. 1. When the move STARTS on ice (2,3): 0.8 chosen direction, 0.1 each perpendicular. Each outcome is resolved separately: a wall/outside -> stay in place.
* **Rewards**: the reward of the tile where the move ends (also when blocked and staying). Entering G = +10 (no extra floor cost), H = -10.
* **Goal of the agent**: maximize the expected sum of discounted rewards, r0 + 0.9 r1 + 0.81 r2 + ... (gamma = 0.9). Not "reach G at any cost": mud and long routes cost, holes are very bad.

## 2. V1 (V0 = 0 everywhere, so V1(s) = max_a of expected immediate reward)
* (1,1): up stay -1, right M -3, down (2,1) -1, left stay -1 -> **-1**
* (1,2) M: up stay on M -3, right (1,3) -1, down wall -> stay on M -3, left (1,1) -1 -> **-1**
* (1,3): right = G -> **+10** (up stay -1, down ice -1, left M -3)
* (2,1): all four moves end on '.' -> **-1**
* (2,3) ice, action by action (expected immediate reward):
  * up: 0.8(-1) + 0.1(-1: left slip hits wall, stay on ice) + 0.1(-10: right slip into H) = -1.9
  * right: 0.8(-10) + 0.1(-1) + 0.1(-1) = -8.2
  * down: 0.8(-1) + 0.1(-1: wall) + 0.1(-10) = -1.9
  * left: 0.8(-1: wall, stay) + 0.1(-1 up) + 0.1(-1 down) = **-1.0** -> V1 = **-1**
* (3,1): **-1**; (3,2): **-1**; (3,3): **-1**; (3,4): up would be H (-10) but right/down/left give -1 -> **-1**
* G, H: 0

V1 map:
```
 -1    -1    10    G(0)
 -1     #    -1    H(0)
 -1    -1    -1    -1
```

## 3. V2 at three tiles (use V1 only): Q = sum p * (R + 0.9 * V1(s'))
**(1,2) mud**
* up: stay on M: -3 + 0.9(-1) = -3.9
* right: (1,3): -1 + 0.9(10) = **8**
* down: wall -> stay on M: -3 + 0.9(-1) = -3.9
* left: (1,1): -1 + 0.9(-1) = -1.9
-> **V2(1,2) = 8**, best: right.

**(1,3)**
* up: stay: -1 + 9 = 8
* right: G: 10 + 0.9(0) = **10**
* down: (2,3): -1 + 0.9(-1) = -1.9
* left: M: -3 + 0.9(-1) = -3.9
-> **V2(1,3) = 10**, best: right.

**(2,3) ice** (V1: (1,3)=10, (2,3)=-1, (3,3)=-1, H=0)
* up: 0.8[-1+0.9(10)] + 0.1[-1+0.9(-1)] + 0.1[-10+0] = 6.4 - 0.19 - 1 = **5.21**
* right: 0.8[-10] + 0.1[-1+9] + 0.1[-1-0.9] = -8 + 0.8 - 0.19 = -7.39
* down: 0.8[-1-0.9] + 0.1[-1-0.9] + 0.1[-10] = -1.52 - 0.19 - 1 = -2.71
* left: 0.8[-1-0.9] + 0.1[-1+9] + 0.1[-1-0.9] = -1.52 + 0.8 - 0.19 = -0.91
-> **V2(2,3) = 5.21**, best: up.

## 4. Best-action arrows after two rounds
(1,2) -> right; (1,3) -> right; (2,3) -> up. No ties at these three tiles.

**Why two rounds do not prove the final optimal policy:** V2 is the best expected return for only a 2-step horizon. Values keep changing in later rounds (my solver needed 12 rounds to change by less than 0.000001; the start tile is still -1 in V1 and becomes 1.54 at the end), so far-away tiles have not "heard" about the goal yet, and the best action can still change as information spreads. Many tiles are even tied in V1 (all 4 actions equal). Only after the values stop changing (convergence) is the greedy policy final.
