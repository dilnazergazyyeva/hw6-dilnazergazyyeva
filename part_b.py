"""Part B: FrozenLake-v1 4x4 simulations with Gymnasium (random vs fixed policy, slippery vs not)."""
import csv, sys, platform
import numpy as np
import gymnasium as gym

OUT = "outputs/"
MAP = "4x4"
MAX_STEPS = 100
BATCH_SEEDS = [11, 22, 33]
N = 50
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3          # Gymnasium FrozenLake action codes
ACT_NAME = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

# Map (state = row*4 + col):      Fixed policy written BEFORE running experiments:
#   S F F F    0  1  2  3         one action for each nonterminal state
#   F H F H    4  5  6  7         (holes 5,7,11,12 and goal 15 are terminal -> no action)
#   F F F H    8  9 10 11
#   H F F G   12 13 14 15
FIXED_POLICY = {0: DOWN, 1: UP, 2: DOWN, 3: UP,
                4: DOWN, 6: DOWN,
                8: RIGHT, 9: RIGHT, 10: DOWN,
                13: RIGHT, 14: RIGHT}


def fixed_policy(state, rng):
    return FIXED_POLICY[state]


def random_policy(state, rng):
    return int(rng.integers(4))


POLICIES = {"random": random_policy, "fixed": fixed_policy}


def run_episode(env, policy, episode_seed, rng, record_trace=False):
    """reset()/step() loop; stops on terminated or truncated, 100-step limit."""
    desc = env.unwrapped.desc.astype(str)       # 4x4 array of 'S','F','H','G'
    state, _ = env.reset(seed=episode_seed)
    total, steps, trace = 0.0, 0, []
    if record_trace:
        trace.append(f"episode_seed={episode_seed}\nstart state {state}\n{env.render()}")
    terminated = truncated = False
    while steps < MAX_STEPS:
        a = policy(state, rng)
        state, r, terminated, truncated, _ = env.step(a)
        total += r
        steps += 1
        if record_trace:
            trace.append(f"step {steps}: action {ACT_NAME[a]} -> state {state}, reward {r}\n{env.render()}")
        if terminated or truncated:
            break
    row, col = divmod(state, 4)
    tile = desc[row, col]
    if terminated and tile == "G":
        outcome = "goal"
    elif terminated and tile == "H":
        outcome = "hole"
    else:
        outcome = "time_limit"                  # not terminated within 100 steps
    return steps, total, outcome, trace


def main():
    versions = f"python {platform.python_version()}, gymnasium {gym.__version__}, numpy {np.__version__}"
    print(versions)
    rows, summary, traces = [], [], {}
    for slippery in (False, True):
        env = gym.make("FrozenLake-v1", map_name=MAP, is_slippery=slippery, render_mode="ansi")
        cond = "slippery" if slippery else "non-slippery"
        for pname, pol in POLICIES.items():
            for bs in BATCH_SEEDS:
                res = []
                for i in range(N):
                    ep_seed = 1000 * bs + i
                    rng = np.random.default_rng(ep_seed)       # action generator seeded too
                    want_trace = (bs == 11 and i < 50)
                    steps, tot, outc, tr = run_episode(env, pol, ep_seed, rng, record_trace=want_trace)
                    rows.append([cond, bs, ep_seed, pname, steps, tot, outc])
                    res.append((steps, tot, outc))
                    # keep first success / first failure / first episode as examples
                    key = (cond, pname)
                    if want_trace:
                        traces.setdefault((key, "first"), "\n".join(tr))
                        if outc == "goal":
                            traces.setdefault((key, "success"), "\n".join(tr))
                        if outc == "hole":
                            traces.setdefault((key, "failure"), "\n".join(tr))
                        if outc == "time_limit":
                            traces.setdefault((key, "time_limit"), "\n".join(tr))
                g = sum(o == "goal" for _, _, o in res); h = sum(o == "hole" for _, _, o in res)
                t = sum(o == "time_limit" for _, _, o in res)
                summary.append((cond, pname, bs, g, h, t, 100 * g / N, sum(s for s, _, _ in res) / N))
        env.close()

    with open(OUT + "part_b_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "batch_seed", "episode_seed", "policy", "steps", "total_reward", "outcome"])
        w.writerows(rows)

    lines = [f"Versions: {versions}\n",
             "| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c, p, bs, g, h, t, sp, ms in summary:
        lines.append(f"| {'yes' if c == 'slippery' else 'no'} | {p} | {bs} | {g} | {h} | {t} | {sp:.0f} | {ms:.2f} |")
    lines.append("\nPooled over the 3 batches (150 episodes):")
    for c in ("non-slippery", "slippery"):
        for p in POLICIES:
            sel = [s for s in summary if s[0] == c and s[1] == p]
            lines.append(f"- {c}, {p}: success {sum(s[3] for s in sel)}/150 = {100*sum(s[3] for s in sel)/150:.1f}%")
    open(OUT + "part_b_summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    with open(OUT + "part_b_traces.txt", "w") as f:
        f.write("Fixed policy: " + ", ".join(f"{s}:{ACT_NAME[a]}" for s, a in sorted(FIXED_POLICY.items())) + "\n\n")
        for (key, kind), text in sorted(traces.items()):
            f.write(f"##### {key[0]} / {key[1]} / {kind} #####\n{text}\n\n")
        missing = [(c, p, k) for c in ("non-slippery", "slippery") for p in POLICIES
                   for k in ("success", "failure") if ((c, p), k) not in traces]
        f.write("Outcomes NOT observed among the first 50 episodes of batch 11 (no trace saved): " + str(missing) + "\n")
    print("Saved outputs/part_b_*.{csv,md,txt}")


if __name__ == "__main__":
    main()
