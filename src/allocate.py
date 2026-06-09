"""
Turn simulation probabilities into a decision: how to allocate across teams.

The model assigns "points" for each round a team reaches (cumulative, like many
pools — you keep earning as your team advances). A team's expected points is the
sum of P(reach stage) * points(stage). Normalizing those gives a fair-value
allocation: the share of the pot each team is "worth" on the model's numbers.

You then compare that fair value to the price/consensus and over- or under-weight
where you disagree. The recommendation is the analytical output; the judgment
stays yours.
"""

import pandas as pd

# Points earned for reaching each stage. Edit to match your pool's scoring.
DEFAULT_PAYOUTS = {
    "R32": 1,
    "R16": 2,
    "QF": 3,
    "SF": 5,
    "final": 8,
    "champion": 13,
}


def expected_value(summary: pd.DataFrame, payouts: dict | None = None) -> pd.DataFrame:
    """Add expected points and a fair-value allocation share per team."""
    payouts = payouts or DEFAULT_PAYOUTS
    df = summary.copy()
    df["expected_points"] = (
        df["p_qualify"]  * payouts["R32"]
        + df["p_r16"]    * payouts["R16"]
        + df["p_qf"]     * payouts["QF"]
        + df["p_sf"]     * payouts["SF"]
        + df["p_final"]  * payouts["final"]
        + df["p_champion"] * payouts["champion"]
    )
    total = df["expected_points"].sum()
    df["fair_share_pct"] = (100 * df["expected_points"] / total).round(2)
    df["expected_points"] = df["expected_points"].round(3)
    return df.sort_values("expected_points", ascending=False).reset_index(drop=True)
