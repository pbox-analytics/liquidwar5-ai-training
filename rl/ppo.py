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

from rl.policy import (ACTIONS, EGO_CHANNELS, NUM_STANCES, act, apply_stances,
                       build_egocentric_obs)
from rl.eval import _heuristic_dydx

# Flat-action ids for the scripted anchored opponents (looked up, not hardcoded,
# so a reordered ACTIONS table can't silently retarget them).
_A_MAEL_UNDERTOW = ACTIONS.index(("Maelstrom", "undertow"))   # 19
_A_DOOM_2X = ACTIONS.index(("Doom", "2x"))                    # 17


def _team_share(engine):
    """(B, T) each team's fraction of all fighters this tick."""
    fighters = engine.team_oh.sum(dim=(2, 3))            # (B, T)
    total = fighters.sum(dim=1, keepdim=True).clamp(min=1)
    return fighters / total


@torch.no_grad()
def collect_rollout(engine, policy, steps, device, stance_eps=0.0):
    """Run `steps` ticks of self-play, returning a rollout dict.

    All tensors are (steps, B, T, ...) on `device`. Finished games are
    excluded from the buffer and the whole batch is re-seeded once >30% of
    games are done (the engine has no per-game reset).

    stance_eps: EPS-GREEDY STANCES — on decision ticks each team picks a
    uniform-random stance with prob eps; the stored logprob is the mixture
    log(eps/N + (1-eps)*p) so PPO stays unbiased (ppo_update must be passed
    the SAME eps so its ratio compares like with like).
    """
    B, T = engine.B, engine.T
    # DEAD-AIR FIX: engine `done` (teams_left <= 1) is LATCHED — a finished
    # game keeps reporting done every tick, so the old code re-fired the
    # terminal +1/-1 every tick and filled the buffer with post-victory idle
    # transitions (~99% of data once most games finished). Track the previous
    # tick's done state so the terminal reward fires ONCE (newly done) and
    # already-finished games never reach the buffer.
    prev_done = engine.team_alive.sum(dim=1) <= 1       # (B,) done state now
    if prev_done.float().mean().item() > 0.3:           # stale batch -> re-seed up front
        engine.reset()
        prev_done = torch.zeros(B, dtype=torch.bool, device=device)
    deadair = torch.zeros((), device=device)            # finished-game ticks collected
    # DRIFT-PROOFING: anchor ~1/3 of games against the FIXED heuristic — teams 1.. there
    # are heuristic-driven and masked out of training, so team 0 is rewarded for
    # beating a REAL opponent, not just its drifting clone (kills the self-play
    # reward-hacking that made the win-rate oscillate). The anchored games split
    # into thirds by game index — neutral knobs (as before), heuristic HOLDING
    # Maelstrom undertow, heuristic HOLDING Doom 2x — so Doom stops feasting on
    # well-less prey and the Maelstrom counter actually appears in the data.
    n_anchor = B // 3
    anchored = torch.zeros(B, T, dtype=torch.bool, device=device)
    anch_neutral = torch.zeros(B, T, dtype=torch.bool, device=device)
    anch_mael = torch.zeros(B, T, dtype=torch.bool, device=device)
    anch_doom = torch.zeros(B, T, dtype=torch.bool, device=device)
    if n_anchor > 0:
        anchored[:n_anchor, 1:] = True
        third = n_anchor // 3
        anch_neutral[:n_anchor - 2 * third, 1:] = True
        anch_mael[n_anchor - 2 * third:n_anchor - third, 1:] = True
        anch_doom[n_anchor - third:n_anchor, 1:] = True
    # STICKY STANCES: re-decide the stance every K ticks and HOLD it between —
    # formations need time to exist before they can earn credit (see act()).
    K_HOLD = 45
    held = getattr(engine, "_held_stance", None)
    if held is None or held.shape != (B, T):
        held = torch.zeros(B, T, dtype=torch.long, device=device)
    obs_buf = []
    act_buf = torch.zeros(steps, B, T, dtype=torch.long, device=device)
    stance_buf = torch.zeros(steps, B, T, dtype=torch.long, device=device)
    logp_buf = torch.zeros(steps, B, T, device=device)
    val_buf = torch.zeros(steps, B, T, device=device)
    rew_buf = torch.zeros(steps, B, T, device=device)
    alive_buf = torch.zeros(steps, B, T, dtype=torch.bool, device=device)
    done_buf = torch.zeros(steps, B, T, device=device)
    decide_buf = torch.zeros(steps, dtype=torch.bool, device=device)

    prev_share = _team_share(engine)

    for s in range(steps):
        obs = engine.get_observation()                  # (B,1+3T,H,W)
        alive = engine.team_alive.clone()
        # Policy picks a cursor move AND a tactical stance per team; the joint
        # log-prob (move + stance) is stored for the PPO ratio.
        decide = int(engine.tick) % K_HOLD == 0
        dydx, stance, logprob, value, _ = act(policy, obs, T, alive,
                                              held_stance=held, decide=decide)
        if stance_eps > 0 and decide:
            # EPS-GREEDY STANCES: second forward for the stance log-probs
            # (cheap — decision ticks are 1/K_HOLD of steps). Re-sample a
            # uniform stance with prob eps and store the MIXTURE logprob so the
            # PPO ratio matches the true sampling distribution.
            ego = build_egocentric_obs(obs, T).reshape(
                B * T, EGO_CHANNELS, *obs.shape[-2:])
            _, s_logits, _ = policy(ego)
            slogp_all = s_logits.log_softmax(dim=-1).view(B, T, NUM_STANCES)
            explore = (torch.rand(B, T, device=device) < stance_eps) & alive
            rand_st = torch.randint(0, NUM_STANCES, (B, T), device=device)
            new_st = torch.where(explore, rand_st, stance)
            slogp_old = slogp_all.gather(-1, stance.unsqueeze(-1)).squeeze(-1)
            slogp_new = slogp_all.gather(-1, new_st.unsqueeze(-1)).squeeze(-1)
            mix = torch.log(stance_eps / NUM_STANCES
                            + (1.0 - stance_eps) * slogp_new.exp())
            logprob = torch.where(alive, logprob - slogp_old + mix,
                                  torch.zeros_like(logprob))   # dead teams keep 0
            stance = new_st
        held = stance                                  # carry the held stance forward
        decide_buf[s] = decide
        if n_anchor > 0:                               # anchored opponents follow the heuristic
            h = _heuristic_dydx(engine)
            dydx = torch.where(anchored.unsqueeze(-1), h, dydx)
            # forced stances BEFORE apply_stances so the Maelstrom/Doom
            # variants' knobs + wells flow through the normal path
            stance = torch.where(anch_neutral, torch.zeros_like(stance), stance)
            stance = torch.where(anch_mael, torch.full_like(stance, _A_MAEL_UNDERTOW), stance)
            stance = torch.where(anch_doom, torch.full_like(stance, _A_DOOM_2X), stance)
        move_idx = (dydx[..., 0] + 1) * 3 + (dydx[..., 1] + 1)  # (B,T) in 0..8

        apply_stances(engine, stance, dydx)            # drive each team's stance knobs (self-play)
        if n_anchor > 0:                               # NEUTRAL-variant opponents run bare knobs (eval baseline)
            engine._spin = torch.where(anch_neutral, torch.ones_like(engine._spin), engine._spin)
            engine._burst = torch.where(anch_neutral, torch.zeros_like(engine._burst), engine._burst)
            engine._surge = torch.where(anch_neutral, torch.ones_like(engine._surge), engine._surge)
            engine._drill = torch.where(anch_neutral.unsqueeze(-1), torch.zeros_like(engine._drill), engine._drill)
            engine._wall = torch.where(anch_neutral.unsqueeze(-1), torch.zeros_like(engine._wall), engine._wall)
            engine._fig8 = torch.where(anch_neutral, torch.zeros_like(engine._fig8), engine._fig8)
            for k in ("_node_l", "_node_m", "_node_k", "_node_w", "_node_v", "_ring", "_ring_ecc"):
                setattr(engine, k, torch.where(anch_neutral, torch.zeros_like(getattr(engine, k)), getattr(engine, k)))
            engine._tide = torch.where(anch_neutral.unsqueeze(-1), torch.zeros_like(engine._tide), engine._tide)
            if getattr(engine, "_wells_enabled", False):       # neutral-variant teams cast no wells
                for k in ("_doom_str", "_doom_horizon", "_doom_cap", "_vortex_str"):
                    getattr(engine, k).mul_((~anch_neutral).float())
        _, done, info = engine.step(dydx)

        share = _team_share(engine)
        reward = (share - prev_share)                   # (B,T) dense
        prev_share = share

        newly = done & ~prev_done                       # finished THIS tick
        if newly.any():
            # Terminal reward: winner (most fighters) +1, others -1, ONLY on
            # the tick the game actually finished (done is latched — gating on
            # raw done paid the winner +1 every idle tick after the win).
            fighters = engine.team_oh.sum(dim=(2, 3))   # (B,T)
            winner = fighters.argmax(dim=1)             # (B,)
            term = -torch.ones(B, T, device=device)
            term[torch.arange(B, device=device), winner] = 1.0
            reward = reward + term * newly.float().unsqueeze(1)

        obs_buf.append(obs)
        act_buf[s] = move_idx
        stance_buf[s] = stance
        logp_buf[s] = logprob
        val_buf[s] = value
        # never train on heuristic opponents OR ticks inside finished games
        alive_buf[s] = alive & ~anchored & ~prev_done.unsqueeze(1)
        done_buf[s] = done.float().unsqueeze(1).expand(B, T)
        rew_buf[s] = reward
        deadair = deadair + prev_done.float().sum()

        prev_done = done.clone()
        if done.float().mean().item() > 0.3:
            # Stopgap re-seed (the engine has no per-game reset): refresh the
            # whole batch and cut GAE here — a reset is an episode boundary,
            # so done=1 for EVERY game or values bootstrap across the seam.
            engine.reset()
            prev_done = torch.zeros(B, dtype=torch.bool, device=device)
            prev_share = _team_share(engine)
            done_buf[s] = 1.0

    engine._held_stance = held                          # persists across rollouts
    # Bootstrap value for the final obs.
    with torch.no_grad():
        last_obs = engine.get_observation()
        _, _, _, last_val, _ = act(policy, last_obs, T, engine.team_alive)

    return {
        "obs": torch.stack(obs_buf, dim=0),             # (steps,B,1+3T,H,W)
        "actions": act_buf,
        "stances": stance_buf,
        "stance_decide": decide_buf,
        "logprobs": logp_buf,
        "values": val_buf,
        "rewards": rew_buf,
        "alive": alive_buf,
        "dones": done_buf,
        "last_value": last_val,                         # (B,T)
        # telemetry: fraction of collected (step, game) ticks that were inside
        # an already-finished game (dead air) — should sit <5% after the fix
        "frac_deadair": (deadair / max(1, steps * B)).item(),
    }


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
               ent_coef=0.01, stance_ent_coef=0.015, stance_eps=0.0,
               max_grad_norm=0.5):
    """Clipped PPO update over the collected rollout. Returns a stats dict.

    stance_ent_coef: SEPARATE coefficient for the stance head's entropy,
    normalized over DECISION ticks only — stances decide on ~1/K_HOLD of
    samples, so a mean over all samples made the stated bonus ~45x weaker.
    stance_eps: must match collect_rollout's eps — the fresh stance logprob
    uses the same eps-mixture or the importance ratio is biased.
    """
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
    # sticky stances: the stance log-prob/entropy only exist on DECISION ticks
    # (collection stored move-only logprobs on held ticks — mirror that here
    # or the importance ratio is wrong)
    smask = (rollout["stance_decide"].view(steps, 1, 1)
             .expand(steps, B, T).reshape(-1).float())
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
    stats = {"policy_loss": 0.0, "value_loss": 0.0,
             "move_entropy": 0.0, "stance_entropy": 0.0, "n": float(n)}
    # STANCE TELEMETRY: decision-tick histogram over the alive training rows —
    # top-5 stance ids + fractions, and how many stances still hold >1% mass
    # (a collapsed head shows one id near 1.00 and n_stances_above_1pct == 1).
    dec = smask.bool() & alive
    if dec.any():
        fracs = torch.bincount(stances[dec], minlength=NUM_STANCES).float()
        fracs = fracs / fracs.sum().clamp(min=1)
        topv, topi = fracs.topk(min(5, NUM_STANCES))
        stats["stance_hist"] = " ".join(
            f"{i}:{v:.2f}" for i, v in zip(topi.tolist(), topv.tolist()) if v > 0)
        stats["n_stances_above_1pct"] = float((fracs > 0.01).sum().item())
    else:
        stats["stance_hist"] = ""
        stats["n_stances_above_1pct"] = 0.0
    stats["frac_deadair"] = float(rollout.get("frac_deadair", 0.0))
    count = 0
    for _ in range(epochs):
        perm = idx_alive[torch.randperm(n, device=device)]
        for start in range(0, n, mb_size):
            mb = perm[start:start + mb_size]
            move_logits, stance_logits, value = policy(ego[mb])
            mdist = torch.distributions.Categorical(logits=move_logits)
            sdist = torch.distributions.Categorical(logits=stance_logits)
            m = smask[mb]
            if stance_eps > 0:
                # same eps-mixture as collection (see collect_rollout) — the
                # ratio must compare mixture to mixture to stay unbiased
                sp = sdist.log_prob(stances[mb]).exp()
                slogp = torch.log(stance_eps / NUM_STANCES + (1.0 - stance_eps) * sp)
            else:
                slogp = sdist.log_prob(stances[mb])
            new_logp = mdist.log_prob(actions[mb]) + m * slogp
            ratio = (new_logp - old_logp[mb]).exp()
            a = adv_f[mb]
            l1 = ratio * a
            l2 = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            policy_loss = -torch.min(l1, l2).mean()
            value_loss = (value - ret_f[mb]).pow(2).mean()
            # DECISION-NORMALIZED stance entropy: mean over decision ticks
            # only, with its own coefficient — the move head keeps ent_coef.
            move_ent = mdist.entropy().mean()
            stance_ent = (m * sdist.entropy()).sum() / m.sum().clamp(min=1.0)
            loss = (policy_loss + vf_coef * value_loss
                    - ent_coef * move_ent - stance_ent_coef * stance_ent)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["move_entropy"] += move_ent.item()
            stats["stance_entropy"] += stance_ent.item()
            count += 1

    for k in ("policy_loss", "value_loss", "move_entropy", "stance_entropy"):
        stats[k] /= max(1, count)
    stats["mean_return"] = ret_f[idx_alive].mean().item()
    return stats
