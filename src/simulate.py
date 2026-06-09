"""
Monte Carlo simulation of the 48-team World Cup.

Format (confirmed for 2026): 12 groups of 4. The top two from each group plus
the eight best third-placed teams advance to a Round of 32, then it's single
elimination through to the final.

Each simulated match draws goals from the Poisson model. Running the whole
tournament thousands of times turns the model's per-match rates into
tournament-level probabilities (chance of advancing, reaching the final,
winning it).
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict

from src.model import expected_goals

# Ordered from earliest exit to champion. Used to compute "reached at least
# stage X" probabilities.
STAGE_ORDER = ["group", "R32", "R16", "QF", "SF", "final", "champion"]
STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}


def load_groups(path: str = "config/groups.json") -> dict:
    """Load {group_name: [team, team, team, team]} from JSON, skipping any
    underscore-prefixed metadata keys (e.g. "_comment")."""
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _sim_score(model, a, b, rng):
    lam_a, lam_b = expected_goals(model, a, b)        # neutral venue
    return int(rng.poisson(lam_a)), int(rng.poisson(lam_b))


def _knockout_winner(model, a, b, rng):
    """Single match; a tie is settled by a strength-weighted shootout."""
    lam_a, lam_b = expected_goals(model, a, b)
    ga, gb = int(rng.poisson(lam_a)), int(rng.poisson(lam_b))
    if ga != gb:
        return a if ga > gb else b
    return a if rng.random() < lam_a / (lam_a + lam_b) else b


def _play_group(model, teams, rng):
    """Round-robin; return standings sorted by points, GD, GF."""
    pts = dict.fromkeys(teams, 0)
    gd = dict.fromkeys(teams, 0)
    gf = dict.fromkeys(teams, 0)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            ga, gb = _sim_score(model, a, b, rng)
            gf[a] += ga; gf[b] += gb
            gd[a] += ga - gb; gd[b] += gb - ga
            if ga > gb:
                pts[a] += 3
            elif gb > ga:
                pts[b] += 3
            else:
                pts[a] += 1; pts[b] += 1
    ranked = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return [{"team": t, "pts": pts[t], "gd": gd[t], "gf": gf[t]} for t in ranked]


def _run_group_stage(model, groups, rng):
    """Return the 32 qualifiers (winners, runners-up, 8 best thirds)."""
    winners, runners_up, thirds = [], [], []
    for teams in groups.values():
        standings = _play_group(model, teams, rng)
        winners.append(standings[0])
        runners_up.append(standings[1])
        thirds.append(standings[2])
    best_thirds = sorted(
        thirds, key=lambda s: (s["pts"], s["gd"], s["gf"]), reverse=True
    )[:8]
    return winners + runners_up + best_thirds


def _knockout(model, qualifiers, rng):
    """
    Seeded single elimination. Teams are ranked by group-stage performance and
    paired best-vs-worst each round. This is a deliberate simplification of the
    official bracket (whose pairings depend on which third-placed teams qualify);
    swap in the real bracket for production accuracy.
    """
    seeded = sorted(qualifiers, key=lambda s: (s["pts"], s["gd"], s["gf"]), reverse=True)
    teams = [s["team"] for s in seeded]
    reached = {t: "R32" for t in teams}
    advance_to = ["R16", "QF", "SF", "final", "champion"]
    round_idx = 0
    while len(teams) > 1:
        n = len(teams)
        survivors = []
        for k in range(n // 2):
            winner = _knockout_winner(model, teams[k], teams[n - 1 - k], rng)
            reached[winner] = advance_to[round_idx]
            survivors.append(winner)
        teams = survivors
        round_idx += 1
    return reached


def simulate_once(model, groups, rng):
    """One full tournament; returns {team: furthest stage reached}."""
    qualifiers = _run_group_stage(model, groups, rng)
    return _knockout(model, qualifiers, rng)


def _reach_prob(counts_t, min_stage, n_sims):
    """P(team reached at least `min_stage`)."""
    floor = STAGE_RANK[min_stage]
    return sum(v for s, v in counts_t.items() if STAGE_RANK[s] >= floor) / n_sims


def monte_carlo(model, groups, n_sims: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Run `n_sims` tournaments and summarize each team's stage probabilities."""
    rng = np.random.default_rng(seed)
    all_teams = [t for teams in groups.values() for t in teams]
    counts = {t: defaultdict(int) for t in all_teams}

    for _ in range(n_sims):
        reached = simulate_once(model, groups, rng)
        for t in all_teams:
            counts[t][reached.get(t, "group")] += 1

    rows = [{
        "team": t,
        "p_qualify":   _reach_prob(counts[t], "R32", n_sims),
        "p_r16":       _reach_prob(counts[t], "R16", n_sims),
        "p_qf":        _reach_prob(counts[t], "QF", n_sims),
        "p_sf":        _reach_prob(counts[t], "SF", n_sims),
        "p_final":     _reach_prob(counts[t], "final", n_sims),
        "p_champion":  _reach_prob(counts[t], "champion", n_sims),
    } for t in all_teams]

    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
