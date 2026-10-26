# RL-Theorey

Learning reinforcement learning from scratch, one small step at a time. The goal is to build up to training a robot arm in MuJoCo, starting with toy environments I write myself.

## Current file

### `line_env.py` — LineEnv (step 1)

A toy environment: a cart (`self.x`) on a number line, and a flag (`self.t`) as the target.

- `reset()` picks a random cart position and a random flag position in [-10, 10] and returns the observation `[x, t]`.
- `step(action)` pushes the cart by `action`, then returns three things:
  - the observation `[x, t]` (what the world looks like after the push)
  - the reward `-abs(x - t)` (best possible score is 0, on the flag)
  - `done` (True when the cart is within 0.1 of the flag)
- Running the file plays up to 100 random pushes as a **baseline agent**. A trained agent later has to beat this.

Plain Python, no installs needed.

## How to run

```bash
python3 line_env.py
```

## Note: `if __name__ == "__main__":`

The test loop at the bottom of `line_env.py` sits under this line. It works like a switch for the test code only:

- Running `python3 line_env.py` directly: `__name__` is `"__main__"`, so the test loop runs.
- Importing it from another file (`import line_env` or `from line_env import LineEnv`): the class loads, but the test loop is skipped.
- Lines above the `if` always run in both cases.

This matters for the next step, where a Gymnasium file will import `LineEnv` and shouldn't trigger the 100-push test every time.

## Plan

1. LineEnv in plain Python (done)
2. Neural networks from scratch in NumPy (Currently in progress)
3. Turn LineEnv into a Gymnasium environment
4. PyTorch
5. Reach environment on a simulated arm in MuJoCo
6. Probability for RL, then RL theory and PPO
7. Train the single arm, then two arms, then sim-to-real
