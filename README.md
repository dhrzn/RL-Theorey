# RL-Theorey

Learning reinforcement learning from scratch, one small step at a time. The goal is to build up to training a robot arm in MuJoCo, starting with toy examples I write myself.

## How the pieces fit together

An RL system has two halves, and I'm building each one on its own first.

- **The game (environment):** `line_env.py`. It defines the world and the rules: what the agent sees, what it can do, and how it gets scored.
- **The player (agent):** `neuron.py`. It's the controller that looks at the world and decides what to do. Here I train a single neuron to learn how.

Right now the two files are separate. Later, the trained neuron will play LineEnv: its output becomes the action that goes into `step()`.

## `line_env.py`: LineEnv (the game)

A cart (`self.x`) on a number line, and a flag (`self.t`) as the target.

- `reset()` picks a random cart position and flag position in [-10, 10] and returns the observation `[x, t]`.
- `step(action)` pushes the cart by `action`, then returns three things:
  - the observation `[x, t]` (what the world looks like after the push)
  - the reward `-abs(x - t)` (best possible score is 0, on the flag)
  - `done` (True when the cart is within 0.1 of the flag)
- Running the file plays up to 100 random pushes as a **baseline agent**. A trained agent later has to beat this.

Plain Python, no installs needed. Run it with `python3 line_env.py`.

### Note: `if __name__ == "__main__":`

The test loop at the bottom of `line_env.py` sits under this line, so it only runs when the file is run directly. Importing the file (`from line_env import LineEnv`) loads the class without running the 100-push test. This matters for the next step, where a Gymnasium file will import `LineEnv`.

## `neuron.py`: a single neuron (the player)

A neuron, `tanh(w1*x + w2*t + b)`, that learns to choose a push from a position `x` and a target `t`. It's trained against an answer key, `tanh(t - x)`.

- The loss is the mean squared error between the neuron's push and the answer key.
- Slopes come from the chain rule (weights, weighted sum, push, loss). They're averaged over a batch of 7 pairs, and each weight moves by `-step_size * average slope`.
- With step size 0.02 and 1000 laps, the MSE falls smoothly to about 0.001, and the weights approach the true rule (-1, 1, 0). Mine ended near (-0.86, 0.84, 0.08).

What I learned:
- With only large-gap pairs, the answer-key pushes sit at about ±1, so the MSE reaches about 0 without learning the real rule. The neuron only learns the sign.
- Adding small-gap pairs forced the weights toward the true rule, and the neuron handled an unseen pair, `[6, 5.5]`.
- A step size that's too big makes the MSE bounce, and one that's too small is slow.

Needs NumPy. Run it with `python3 neuron.py`.

## Plan

1. LineEnv in plain Python (done)
2. Neural networks from scratch in NumPy (in progress: single neuron done, next is a two-layer network on XOR)
3. Turn LineEnv into a Gymnasium environment
4. PyTorch
5. Reach environment on a simulated arm in MuJoCo
6. Probability for RL, then RL theory and PPO
7. Train the single arm, then two arms, then sim-to-real
