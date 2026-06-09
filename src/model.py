"""
A weighted Poisson model of goals scored.

The idea (a standard, interpretable football model): the number of goals a
team scores in a match follows a Poisson distribution whose rate depends on
the team's attacking strength, the opponent's defensive strength, and whether
the team is at home:

    log(expected_goals) = baseline + attack[team] + defense[opponent] + home_edge

Fitting this as a Poisson GLM on the long-format data gives one coefficient
per team (attack and defense) plus a single home-advantage term — all of which
you can read off and explain. That interpretability is the whole point versus a
black-box classifier.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_poisson(long_df: pd.DataFrame):
    """Fit the weighted Poisson GLM and return the fitted model."""
    model = smf.glm(
        formula="goals ~ C(team) + C(opponent) + is_home",
        data=long_df,
        family=sm.families.Poisson(),
        freq_weights=long_df["weight"].values,
    ).fit()
    return model


def known_teams(long_df: pd.DataFrame) -> set:
    """Teams the model has seen — used to validate the group config."""
    return set(long_df["team"].unique())


def expected_goals(model, team_a: str, team_b: str, team_a_home: bool = False):
    """
    Expected goals for each side. At a World Cup almost every match is on
    neutral ground, so `team_a_home` defaults to False; set it True only for
    host-nation matches.
    """
    lam_a = model.predict(pd.DataFrame(
        {"team": [team_a], "opponent": [team_b], "is_home": [int(team_a_home)]}
    )).iloc[0]
    lam_b = model.predict(pd.DataFrame(
        {"team": [team_b], "opponent": [team_a], "is_home": [0]}
    )).iloc[0]
    return float(lam_a), float(lam_b)


def match_probabilities(lam_a: float, lam_b: float, max_goals: int = 10):
    """
    P(team_a win), P(draw), P(team_b win) from the two scoring rates,
    by summing over the joint score matrix. Useful for analysis/EDA;
    the simulation samples scores directly instead (faster).
    """
    from scipy.stats import poisson
    a = poisson.pmf(np.arange(max_goals + 1), lam_a)
    b = poisson.pmf(np.arange(max_goals + 1), lam_b)
    matrix = np.outer(a, b)
    p_a = np.tril(matrix, -1).sum()   # a scores more
    p_draw = np.trace(matrix)         # equal scores
    p_b = np.triu(matrix, 1).sum()    # b scores more
    return p_a, p_draw, p_b
