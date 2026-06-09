"""
Load and prepare international match results for the goals model.

Two weighting ideas drive the model's realism:
  1. Recency  — a match from last year says more about a team's current
     strength than one from 2015, so weights decay with age.
  2. Competitiveness — World Cup and qualifier results are more informative
     than friendlies, so they carry more weight.
"""

import numpy as np
import pandas as pd

# Weight of each match type relative to a friendly. Tune these to taste —
# they encode the assumption "competitive matches are more informative".
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "FIFA World Cup qualification": 0.80,
    "UEFA Euro": 0.80,
    "Copa América": 0.80,
    "UEFA Nations League": 0.70,
    "African Cup of Nations": 0.70,
    "AFC Asian Cup": 0.65,
    "Gold Cup": 0.60,
    "Friendly": 0.30,
}
DEFAULT_WEIGHT = 0.50  # any tournament not listed above


def load_results(path: str = "data/results.csv", since: str = "2015-01-01") -> pd.DataFrame:
    """Read results.csv, keep matches on/after `since`, drop rows missing scores."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] >= pd.Timestamp(since)].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def add_weights(df: pd.DataFrame, half_life_days: int = 730) -> pd.DataFrame:
    """Attach a per-match weight = recency_decay * competitiveness."""
    df = df.copy()
    most_recent = df["date"].max()
    age_days = (most_recent - df["date"]).dt.days
    recency_w = 0.5 ** (age_days / half_life_days)          # halves every ~2 years
    comp_w = df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)
    df["weight"] = recency_w * comp_w
    return df


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape to one row per (team, match) recording goals scored.
    Each match becomes two rows so the model can learn an attacking
    strength for the scorer and a defensive strength for the opponent.
    `is_home` is 1 only for the home side of a non-neutral match.
    """
    home = pd.DataFrame({
        "team": df["home_team"],
        "opponent": df["away_team"],
        "goals": df["home_score"],
        "is_home": np.where(df["neutral"], 0, 1),
        "weight": df["weight"],
    })
    away = pd.DataFrame({
        "team": df["away_team"],
        "opponent": df["home_team"],
        "goals": df["away_score"],
        "is_home": 0,
        "weight": df["weight"],
    })
    return pd.concat([home, away], ignore_index=True)
