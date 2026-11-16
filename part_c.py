"""Part C: value iteration on the paper map and the extended map + simulations + rule checks."""
import io, contextlib
import sys; sys.stdout.reconfigure(encoding="utf-8")
from floors import BASE_FLOORS, EXTENDED_FLOORS, PAPER_MAP, EXTENDED_MAP
from gridworld import GridWorld, ACTIONS, PERP, DELTA

OUT = "outputs/"
TOL, MAXIT = 1e-6, 10000


def solve(name, rows, floors, fname):
    w = GridWorld(rows, floors)
    V, it, delta, ok, hist = w.value_iteration(TOL, MAXIT)
    policy, ties = w.extract_policy(V)
    text = []
    text.append(f"=== {name} ===  map: {rows}")
    text.append(f"Converged: {ok}; iterations: {it}; final max change: {delta:.3e}  (tol {TOL}, safeguard {MAXIT})")
    text.append("\nV1:\n" + w.value_map(hist[1]))
    text.append("\nV2:\n" + w.value_map(hist[2]))
    text.append("\nFinal V (* = terminal, value 0):\n" + w.value_map(V))
    text.append("\nGreedy policy:\n" + w.policy_map(policy, ties))
    text.append("\nQ-values at every nonterminal tile (final V):")
    for s in w.nonterminal:
        q = ", ".join(f"{a}={w.q_value(V, s, a):.4f}" for a in ACTIONS)
        text.append(f"  {s} [{w.tile(s)}]: {q}")
    open(OUT + fname, "w", encoding="utf-8").write("\n".join(text) + "\n")
    return w, V, it, delta, ok, hist, policy, ties


# ---------------- 1. paper map ----------------
wp, Vp, itp, dp, okp, hp, pp, tp = solve("PAPER MAP", PAPER_MAP, BASE_FLOORS, "part_c_paper_map.txt")
print("V1 (paper map):\n" + wp.value_map(hp[1]))
print("\nRequested V2 values:")
for s in [(1, 2), (1, 3), (2, 3)]:
    q = {a: round(wp.q_value(hp[1], s, a), 4) for a in ACTIONS}
    print(f"  V2{s} = {hp[2][s]:.4f}   Q = {q}")

# compare with hand calculation
hand_v1 = {(1,1):-1,(1,2):-1,(1,3):10,(2,1):-1,(2,3):-1,(3,1):-1,(3,2):-1,(3,3):-1,(3,4):-1}
assert all(abs(hp[1][s]-v) < 1e-12 for s, v in hand_v1.items()), "V1 differs from hand"
assert abs(hp[2][(1,2)]-8) < 1e-12 and abs(hp[2][(1,3)]-10) < 1e-12 and abs(hp[2][(2,3)]-5.21) < 1e-12
print("Code V1/V2 match hand calculation: OK")

# ---------------- 2. extended map ----------------
we, Ve, ite, de, oke, he, pe, te = solve("EXTENDED MAP", EXTENDED_MAP, EXTENDED_FLOORS, "part_c_extended_map.txt")

for w_, V_, it_, d_, ok_, p_, t_, nm in [(wp, Vp, itp, dp, okp, pp, tp, "paper"), (we, Ve, ite, de, oke, pe, te, "extended")]:
    print(f"\n[{nm}] converged={ok_} iterations={it_} final change={d_:.3e}")
    print(w_.value_map(V_))
    print(w_.policy_map(p_, t_))

# ---------------- 3. simulations ----------------
sp = wp.simulate(pp, "paper", csv_path=OUT + "part_c_episodes_paper.csv")
se = we.simulate(pe, "extended", csv_path=OUT + "part_c_episodes_extended.csv")
lines = ["| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
for r in sp + se:
    lines.append(f"| {r['map']} | {r['batch']} | {r['goals']} | {r['holes']} | {r['limits']} | {r['mean_steps']:.2f} | {r['mean_ret']:.4f} |")
for nm, S, V_, w_ in [("paper", sp, Vp, wp), ("extended", se, Ve, we)]:
    avg = sum(r["mean_ret"] for r in S) / 3
    lines.append(f"\n{nm}: average of 3 batch means = {avg:.4f};  V*(start {w_.start}) = {V_[w_.start]:.4f}")
open(OUT + "part_c_simulation_summary.md", "w").write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines))

# ---------------- 4. rule checks ----------------
chk = []
def check(msg, cond, detail=""):
    chk.append(f"[{'PASS' if cond else 'FAIL'}] {msg} {detail}")
    assert cond, msg

w = we  # extended world contains every tile type
# boundary: up from (1,1) stays, gets reward of (1,1) = -1
t = w.transitions((1, 1), "up");   check("Boundary: up from (1,1) stays", t == [(1.0, (1, 1))], str(t))
t = w.transitions((1, 1), "left"); check("Boundary: left from (1,1) stays", t == [(1.0, (1, 1))], str(t))
check("Boundary reward = reward of tile where move ends (-1 on '.')", w.reward_of((1, 1)) == -1)
# wall: down from (1,2) [mud] hits wall (2,2): stay on mud, reward -3
t = w.transitions((1, 2), "down"); check("Wall: down from mud (1,2) stays", t == [(1.0, (1, 2))], str(t))
check("Wall block on mud gives mud reward -3", w.reward_of(t[0][1]) == -3)
# ice probabilities
for a in ACTIONS:
    t = w.transitions((2, 3), a)
    check(f"Ice: probabilities of action '{a}' sum to 1", abs(sum(p for p, _ in t) - 1) < 1e-12, str(t))
check("Ice: intended 0.8, perpendicular 0.1 each",
      [p for p, _ in w.transitions((2, 3), "up")] == [0.8, 0.1, 0.1])
check("Ice slip uses the tile we LEAVE: moving INTO ice from (1,3) down is deterministic",
      w.transitions((1, 3), "down") == [(1.0, (2, 3))])
# ice with walls: left slip from ice resolves against wall (2,2) separately
t = w.transitions((2, 3), "up")
check("Ice up: perpendicular-left hits wall (2,2) -> stay (2,3); perpendicular-right -> hole (2,4)",
      t == [(0.8, (1, 3)), (0.1, (2, 3)), (0.1, (2, 4))], str(t))
# floor rewards
for tl, rw in [("M", -3), ("D", -2), ("W", -5), (".", -1), ("S", -1), ("I", -1), ("G", 10), ("H", -10)]:
    pos = [(r, c) for r in range(1, 4) for c in range(1, 5) if w.tile((r, c)) == tl][0]
    check(f"Reward of '{tl}' = {rw}", w.reward_of(pos) == rw)
# deterministic movement from D and W
check("Sand D moves deterministically", w.transitions((3, 2), "right") == [(1.0, (3, 3))])
check("Water W moves deterministically", w.transitions((3, 4), "left") == [(1.0, (3, 3))])
# terminal handling
check("No transitions out of goal", w.transitions((1, 4), "up") == [])
check("No transitions out of hole", w.transitions((2, 4), "up") == [])
check("V(goal) = V(hole) = 0", Ve[(1, 4)] == 0 and Ve[(2, 4)] == 0)
# entering goal gives exactly +10 (V(1,3) after V1 step = 10, no floor cost added)
check("Entering G gives +10 without extra floor cost (V1(1,3)=10)", hp[1][(1, 3)] == 10)
# episode stops at terminal: run an episode and verify steps stop at first terminal
steps, ret, outcome = w.run_episode(pe, 1000 * 11)
check("Episode ends at first terminal state and rewards stop", outcome in ("goal", "hole") and steps < 100, f"(steps={steps}, outcome={outcome})")
# adding a floor needs no solver change
check("Walls are not states", all(w.kind(s) != "wall" for s in w.states) and (2, 2) not in w.states)
open(OUT + "part_c_rule_checks.txt", "w").write("\n".join(chk) + "\n")
print("\n" + "\n".join(chk))
