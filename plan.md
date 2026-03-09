# Plan: Making RL Learn Nuanced Euchre Strategy

## Context

The bug fixes already committed (double softmax, dead entropy, broken returns, reward
asymmetry, detached gradients, no advantage normalization) were necessary but may not be
sufficient. Pure REINFORCE against random opponents has fundamental limitations for a
team game with delayed rewards. This plan lays out a phased approach to get the model
learning real strategy.

---

## Phase 1: Validate Bug Fixes Work (policy_gradient.py, train_gradient.py)

**Goal:** Confirm the 6 bug fixes produce measurable improvement before adding complexity.

1. **Add training diagnostics to `train_gradient.py`** — track and print per-update:
   - Trump call rate (% of opportunities where model calls vs passes)
   - Entropy per decision type (card/trump/discard separately)
   - Action distribution stats (are outputs still collapsing to uniform or one-hot?)
   - Euchre rate (how often model gets euchred when it calls)

2. **Run a baseline training session** — 500 updates, batch size 100, and check:
   - Does trump call rate stay between 15-40% (not 0% or 100%)?
   - Does entropy stay above 0.3 for trump decisions?
   - Does win rate trend upward from ~50% baseline?

**If this works** (model calls sometimes, entropy stays healthy, win rate improves): proceed to Phase 2.
**If it still collapses**: increase `entropy_beta` to 0.05-0.1, or reduce learning rate to 0.00003.

**Files changed:** `train_gradient.py` (diagnostics only)

---

## Phase 2: Add a Critic Network for Variance Reduction (new file + basic_nn.py)

**Goal:** Replace the crude running-average baseline with a learned state-value function.
This is the single biggest architectural improvement for the team credit assignment problem.

The core issue: in REINFORCE, the return for "I called trump" includes noise from my
partner's card play and the opponents' random choices. A critic that learns V(s) — "how
good is this game state for my team?" — lets us compute advantage = R - V(s), which
strips out the noise from other players' actions. The policy gradient only reinforces
decisions that did *better than expected given the state*, not just decisions that
happened to coincide with wins.

1. **Create `networks/critic_nn.py`** — a simple state-value network:
   - Same input encodings as the policy network (161 for card, 49 for trump, 35 for discard)
   - Separate value heads matching the policy heads
   - Output: single scalar V(s) per head
   - Architecture: 2 hidden layers (64, 32) with ReLU, no softmax

2. **Modify `PolicyGradientTrainer`** to become Actor-Critic:
   - Add critic network alongside the existing policy (actor) network
   - During `play_game()`: record V(s) predictions alongside each decision
   - In `train_on_batch()`: compute advantage = return - V(s) instead of return - mean(return)
   - Add critic loss: MSE between V(s) and actual returns
   - Total loss = policy_loss + 0.5 * critic_loss - entropy_beta * entropy
   - Separate optimizers for actor and critic (critic can use higher LR)

3. **Key design decision**: critic shares no weights with the policy network. This is
   simpler and avoids gradient interference between the policy and value objectives.

**Files changed:** new `networks/critic_nn.py`, `training/policy_gradient.py`

---

## Phase 3: Self-Play Instead of Random Opponents (policy_gradient.py)

**Goal:** Train against copies of itself instead of random AI. This forces the model to
learn strategy that works against *good* play, not just strategy that exploits random
opponents' mistakes.

Against random opponents, "always pass" works because random players call bad hands.
Against a copy of itself, "always pass" means *neither* side calls, leading to stuck
games and no reward signal — which forces the model to learn *when* calling is good.

1. **Modify `play_game()`**: instead of `PlayerType.RANDOM_AI` for opponents, load a
   frozen copy of the current model and use it for positions 1 & 3.

2. **Opponent update schedule**: don't update the opponent every batch (causes instability).
   Instead, freeze the opponent and update it every N batches (e.g., every 20 updates).
   This is the standard "league training" / "fictitious self-play" approach.

3. **Keep a pool of past model snapshots**: periodically save checkpoints and randomly
   select opponents from the pool. This prevents cyclic strategies (A beats B, B beats C,
   C beats A) and ensures the model learns robust play.

**Files changed:** `training/policy_gradient.py`, `train_gradient.py`

---

## Phase 4: Reward Shaping Refinements (policy_gradient.py)

**Goal:** Give the model denser, more informative reward signals that capture Euchre-specific
strategy beyond just win/loss.

1. **Trump calling reward based on hand strength**: When the model calls trump, give a
   small immediate reward proportional to hand strength (number of trump, bowers, aces).
   This bootstraps the calling decision before the hand outcome is known. Scale it small
   (0.01-0.03) so it doesn't dominate the actual outcome reward.

2. **Leading trump reward**: When the model leads a trump card as the first play and wins
   the trick, give a small bonus. Leading trump to draw out opponents' trump is a core
   Euchre strategy.

3. **Partner coordination signal**: When the model's partner wins a trick, give a smaller
   positive reward (e.g., 0.5x the normal trick reward) to the model's decisions in that
   hand. This teaches the model that helping your partner win is also good.

4. **Differentiate the caller's reward from the partner's reward**: When the model calls
   trump, the model's trump decision gets the full hand-outcome reward. But card-play
   decisions by the *partner* position should get a slightly attenuated version (0.7x),
   since the partner didn't make the calling decision. This helps disentangle team credit.

**Files changed:** `training/policy_gradient.py`

---

## Phase 5 (Optional): PPO Instead of REINFORCE (new file)

**Goal:** If training is still unstable after Phases 1-4, switch from REINFORCE to PPO
(Proximal Policy Optimization). PPO clips the policy update to prevent catastrophic
changes that cause the model to suddenly forget good behavior.

This is a bigger refactor but PPO is strictly better than REINFORCE for this type of
problem:
- Clipped objective prevents policy collapse
- Multiple epochs per batch (reuse data, more sample-efficient)
- Natural entropy bonus integration
- Better stability with the critic

1. **Create `training/ppo_trainer.py`**:
   - Collect trajectories same as current `play_game()`
   - Compute GAE (Generalized Advantage Estimation) using the critic
   - Run K epochs (3-5) per batch with clipped surrogate objective
   - Clip ratio epsilon = 0.2 (standard)

2. **Create `train_ppo.py`** entry point with appropriate hyperparameters.

**Files changed:** new `training/ppo_trainer.py`, new `train_ppo.py`

---

## Implementation Order and Dependencies

```
Phase 1 (diagnostics)     ← Do first, validates bug fixes
    |
Phase 2 (critic network)  ← Biggest impact on team credit assignment
    |
Phase 3 (self-play)       ← Biggest impact on strategy quality
    |
Phase 4 (reward shaping)  ← Fine-tuning, can be done in parallel with 3
    |
Phase 5 (PPO)             ← Only if 1-4 aren't enough
```

Phases 1-3 are the critical path. Phase 4 can be mixed in at any point. Phase 5 is
insurance if the simpler approach doesn't converge.

## Estimated Scope

| Phase | New Files | Modified Files | Complexity |
|-------|-----------|---------------|------------|
| 1     | 0         | 1             | Small      |
| 2     | 1         | 1             | Medium     |
| 3     | 0         | 2             | Medium     |
| 4     | 0         | 1             | Small      |
| 5     | 2         | 0             | Large      |
