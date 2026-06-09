"""
Quick exploratory plots saved to visuals/. Run before modeling to sanity-check
the data and to have figures for the README.
"""

import matplotlib
matplotlib.use("Agg")  # write files without a display
import matplotlib.pyplot as plt
import numpy as np

from src.data_prep import load_results


def main():
    df = load_results()

    # 1. Home advantage: average goal difference at home vs on neutral ground.
    home = df[~df["neutral"]]
    neutral = df[df["neutral"]]
    home_gd = (home["home_score"] - home["away_score"]).mean()
    neutral_gd = (neutral["home_score"] - neutral["away_score"]).mean()

    plt.figure(figsize=(5, 4))
    plt.bar(["At home", "Neutral"], [home_gd, neutral_gd], color=["#2a6f97", "#a3b18a"])
    plt.ylabel("Avg goal difference (home - away)")
    plt.title("Home advantage in international football")
    plt.tight_layout()
    plt.savefig("visuals/home_advantage.png", dpi=120)
    plt.close()

    # 2. Distribution of total goals per match — is Poisson a reasonable fit?
    total_goals = df["home_score"] + df["away_score"]
    plt.figure(figsize=(6, 4))
    plt.hist(total_goals, bins=range(0, 12), align="left", rwidth=0.85, color="#2a6f97")
    plt.xlabel("Total goals in a match")
    plt.ylabel("Number of matches")
    plt.title("Goals per match (since 2015)")
    plt.tight_layout()
    plt.savefig("visuals/goals_distribution.png", dpi=120)
    plt.close()

    print("Saved: visuals/home_advantage.png, visuals/goals_distribution.png")
    print(f"Mean goal diff at home: {home_gd:.2f} | neutral: {neutral_gd:.2f}")


if __name__ == "__main__":
    main()
