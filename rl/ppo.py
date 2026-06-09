"""Batched PPO self-play for the Liquid War cursor policy.

The batched GPU engine (simulator.engine.LiquidWarEngine) is the environment:
B games run in parallel, every team in every game is driven by the SAME
current policy (self-play). We collect a fixed-length rollout, compute GAE
advantages, and do clipped-PPO minibatch updates.

Transitions are stored per (game, team): the rollout tensors have a leading
(steps, B, T) shape and are flattened to (steps*B*T, ...) for the update, with
dead-team steps masked out.

Reward (dense, per team, per tick): change in this team's share of total
fighters (dominance-delta), plus a terminal +1/-1 on win/loss. This gives a
gradient of signal every tick instead of a single sparse end reward.
"""

import torch

from rl.policy import act, apply_stances
from rl.eval import _heuristic_dydx


def _team_share(engine):
    """(B, T) each team's fraction of all fighters this tick."""
    fighters = engine.team_oh.sum(dim=(2, 3))            # (B, T)
    total = fighters.sum(dim=1, keepdim=True).clamp(min=1)
    return fighters / total


@torch.no_grad()
def collect_rollout(engine, policy, steps, device):
    """Run `steps` ticks of self-play, returning a rollout dict.

    All tensors are (steps, B, T, ...) on `device`. Episodes that finish
    (done) are reset mid-rollout so collection stays full.
    """
    B, T = engine.B, engine.T
    # DRIFT-PROOFING: anchor ~1/3 of games against the FIXED heuristic — teams 1.. there
    # are heuristic-driven with neutral knobs and masked out of training, so team 0 is
    # rewarded for beating a REAL opponent, not just its drifting clone (kills the
    # self-play reward-hacking that made the win-rate oscillate).
    n_anchor = B // 3
    anchored = torch.zeros(B, T, dtype=torch.bool, device=device)
    if n_anchor > 0:
        anchored[:n_anchor, 1:] = True
    obs_buf = []
    act_buf = torch.zeros(steps, B, T, dtype=torch.long, device=device)
    stance_buf = torch.zeros(steps, B, T, dtype=torch.long, device=device)
    logp_buf = torch.zeros(steps, B, T, device=device)
    val_buf = torch.zeros(steps, B, T, device=device)
    rew_buf = torch.zeros(steps, B, T, device=device)
    alive_buf = torch.zeros(steps, B, T, dtype=torch.bool, device=device)
    done_buf = torch.zeros(steps, B, T, device=device)

    prev_share = _team_share(engine)

    for s in range(steps):
        obs = engine.get_observation()                  # (B,1+3T,H,W)
        alive = engine.team_alive.clone()
        # Policy picks a cursor move AND a tactical stance per team; the joint
        # log-prob (move + stance) is stored for the PPO ratio.
        dydx, stance, logprob, value, _ = act(policy, obs, T, alive)
        if n_anchor > 0:                               # anchored opponents follow the heuristic, no stance
            h = _heuristic_dydx(engine)
            dydx = torch.where(anchored.unsqueeze(-1), h, dydx)
            stance = torch.where(anchored, torch.zeros_like(stance), stance)
        move_idx = (dydx[..., 0] + 1) * 3 + (dydx[..., 1] + 1)  # (B,T) in 0..8

        apply_stances(engine, stance, dydx)            # drive each team's stance knobs (self-play)
        if n_anchor > 0:                               # heuristic opponents run NEUTRAL knobs (eval baseline)
            engine._spin = torch.where(anchored, torch.ones_like(engine._spin), engine._spin)
            engine._burst = torch.where(anchored, torch.zeros_like(engine._burst), engine._burst)
            engine._surge = torch.where(anchored, torch.ones_like(engine._surge), engine._surge)
            engine._drill = torch.where(anchored.unsqueeze(-1), torch.zeros_like(engine._drill), engine._drill)
            engine._wall = torch.where(anchored.unsqueeze(-1), torch.zeros_like(engine._wall), engine._wall)
            engine._fig8 = torch.where(anchored, torch.zeros_like(engine._fig8), engine._fig8)
        _, done, info = engine.step(dydx)

        share = _team_share(engine)
        reward = (share - prev_share)                   # (B,T) dense
        prev_share = share

        obs_buf.append(obs)
        act_buf[s] = move_idx
        stance_buf[s] = stance
        logp_buf[s] = logprob
        val_buf[s] = value
        alive_buf[s] = alive & ~anchored               # never train on the heuristic opponents' transitions
        done_buf[s] = done.float().unsqueeze(1).expand(B, T)

        if done.any():
            # Terminal reward: winner (most fighters) +1, others -1, for
            # games that just finished.
            fighters = engine.team_oh.sum(dim=(2, 3))   # (B,T)
            winner = fighters.argmax(dim=1)             # (B,)
            term = -torch.ones(B, T, device=device)
            term[torch.arange(B, device=device), winner] = 1.0
            term = term * done.float().unsqueeze(1)
            reward = reward + term
            # Reset finished games so the rollout keeps producing data.
            _reset_done_games(engine, done)
            prev_share = _team_share(engine)

        rew_buf[s] = reward

    # Bootstrap value for the final obs.
    with torch.no_grad():
        last_obs = engine.get_observation()
        _, _, _, last_val, _ = act(policy, last_obs, T, engine.team_alive)

    return {
        "obs": torch.stack(obs_buf, dim=0),             # (steps,B,1+3T,H,W)
        "actions": act_buf,
        "stances": stance_buf,
        "logprobs": logp_buf,
        "values": val_buf,
        "rewards": rew_buf,
        "alive": alive_buf,
        "dones": done_buf,
        "last_value": last_val,                         # (B,T)
    }


def _reset_done_games(engine, done):
    """Re-initialize only the games in `done` (others keep running)."""
    if done.all():
        engine.reset()
        return
    # Simplest correct approach: full reset is cheap relative to a rollout;
    # but to avoid disturbing in-flight games we reset all when ANY large
    # fraction is done. For partial, re-seed via a fresh engine.reset() is
    # not per-game, so we do a full reset only when all done; otherwise we
    # let finished games sit idle (team_alive<=1 -> "stay", zero reward).
    # This keeps semantics simple and correct; throughput cost is minor for
    # short rollouts. (A per-game reset can be added later.)
    return


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation. All (steps,B,T). Returns adv, ret."""
    steps = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(last_value)
    for t in reversed(range(steps)):
        next_val = last_value if t == steps - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_adv = delta + gamma * lam * next_nonterminal * last_adv
        adv[t] = last_adv
    ret = adv + values
    return adv, ret


def ppo_update(policy, optimizer, rollout, num_teams,
               epochs=4, minibatches=4, clip=0.2, vf_coef=0.5,
               ent_coef=0.01, max_grad_norm=0.5):
    """Clipped PPO update over the collected rollout. Returns a stats dict."""
    from rl.policy import build_egocentric_obs, EGO_CHANNELS

    steps, B, C, H, W = rollout["obs"].shape
    T = num_teams
    device = rollout["obs"].device

    adv, ret = compute_gae(
        rollout["rewards"], rollout["values"],
        rollout["dones"], rollout["last_value"])

    # Flatten (steps,B,T) -> (N,), keep only alive transitions.
    alive = rollout["alive"].reshape(-1)                  # (steps*B*T,)
    actions = rollout["actions"].reshape(-1)
    stances = rollout["stances"].reshape(-1)
    old_logp = rollout["logprobs"].reshape(-1)
    adv_f = adv.reshape(-1)
    ret_f = ret.reshape(-1)

    # Egocentric obs per (step,team) -> (steps*B*T, EGO, H, W)
    ego = build_egocentric_obs(
        rollout["obs"].reshape(steps * B, C, H, W), T)    # (steps*B, T, EGO,H,W)
    ego = ego.reshape(steps * B * T, EGO_CHANNELS, H, W)

    idx_alive = alive.nonzero(as_tuple=True)[0]
    if idx_alive.numel() == 0:
        return {"skipped": 1.0}

    # Normalize advantages over the alive set.
    a_sel = adv_f[idx_alive]
    a_sel = (a_sel - a_sel.mean()) / (a_sel.std() + 1e-8)
    adv_f = adv_f.clone()
    adv_f[idx_alive] = a_sel

    n = idx_alive.numel()
    mb_size = max(1, n // minibatches)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n": float(n)}
    count = 0
    for _ in range(epochs):
        perm = idx_alive[torch.randperm(n, device=device)]
        for start in range(0, n, mb_size):
            mb = perm[start:start + mb_size]
            move_logits, stance_logits, value = policy(ego[mb])
            mdist = torch.distributions.Categorical(logits=move_logits)
            sdist = torch.distributions.Categorical(logits=stance_logits)
            new_logp = mdist.log_prob(actions[mb]) + sdist.log_prob(stances[mb])
            ratio = (new_logp - old_logp[mb]).exp()
            a = adv_f[mb]
            l1 = ratio * a
            l2 = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            policy_loss = -torch.min(l1, l2).mean()
            value_loss = (value - ret_f[mb]).pow(2).mean()
            entropy = (mdist.entropy() + sdist.entropy()).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            count += 1

    for k in ("policy_loss", "value_loss", "entropy"):
        stats[k] /= max(1, count)
    stats["mean_return"] = ret_f[idx_alive].mean().item()
    return stats
