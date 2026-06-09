"""
End-to-end pipeline: prep -> fit model -> simulate -> allocate.

Run from the repo root:  python -m src.run_pipeline
"""

import os
import pandas as pd

from src.data_prep import load_results, add_weights, to_long_format
from src.model import fit_poisson, known_teams
from src.simulate import load_groups, monte_carlo
from src.allocate import expected_value

N_SIMS = 10000


def main():
    # 1. Prepare data and fit the goals model.
    df = add_weights(load_results())
    long_df = to_long_format(df)
    model = fit_poisson(long_df)

    # 2. Load the group draw and check every team is one the model has seen.
    groups = load_groups()
    flat = [t for teams in groups.values() for t in teams]
    missing = sorted(set(flat) - known_teams(long_df))
    if missing:
        raise ValueError(
            "These teams aren't in the training data — check spelling against "
            f"results.csv: {missing}"
        )

    # 3. Simulate the tournament and turn the results into an allocation.
    summary = monte_carlo(model, groups, n_sims=N_SIMS)
    allocation = expected_value(summary)

    os.makedirs("output", exist_ok=True)
    allocation.to_csv("output/allocation.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(allocation.head(20).to_string(index=False))
    print("\nFull table written to output/allocation.csv")


if __name__ == "__main__":
    main()
