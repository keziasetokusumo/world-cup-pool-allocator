# World Cup 2026 — Predictions & Pool Allocation

**Which teams offer the best value when you have to allocate across all 48?**

Predicting the 2026 World Cup with a weighted Poisson that is modeled on ~150 years of international results. The tournament is simulated thousands of times, and teams are ordered by expected value. 

---

## Data

[International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
(Mart Jürisoo) — ~49,000 men's full international matches with date, teams,
score, tournament, venue, and a `neutral` flag. The repo uses `results.csv`;
`shootouts.csv` and `goalscorers.csv` are available for extensions.

> The `data/` folder is gitignored. Download `results.csv` from the link above
> and place it there.

## Approach

1. **Weight the matches** (`src/data_prep.py`). Recent matches count more (weight
   halves roughly every two years) and competitive matches count more than
   friendlies. This keeps the model anchored to current form.
2. **Fit a Poisson goals model** (`src/model.py`). Goals scored are modeled as
   `log(expected_goals) = baseline + attack[team] + defense[opponent] + home_edge`.
   Every term is readable: you can pull out any team's attacking and defensive
   coefficients and explain them.
3. **Simulate the tournament** (`src/simulate.py`). The 2026 format — 12 groups of
   4, top two plus eight best third-placed teams into a Round of 32, then single
   elimination — is played out 10,000 times, drawing each match's score from the
   model. This converts per-match rates into tournament probabilities.
4. **Recommend an allocation** (`src/allocate.py`). Each team earns points for the
   rounds it reaches; expected points (probability × points) normalized across
   teams gives a fair-value share of the pot. You then over- or under-weight where
   your read differs from the model.

## Pipeline

```
results.csv → weight → Poisson model → 10,000 simulations → expected value → allocation.csv
```

## Key findings

> Run the pipeline against the loaded data and the actual group draw, then fill in
> the numbers below in your own words and add the matching charts to `visuals/`.

- **Title favorites:** The model gives **[team]** the highest title probability at
  **[__]%**, followed by **[team]** and **[team]**. *(See `visuals/title_odds.png`.)*
- **Home advantage:** On non-neutral matches the home side averages **[__]** more
  goals than the away side; the effect is much smaller at neutral venues — which is
  why the simulation drops it for all but host matches. *(See `visuals/home_advantage.png`.)*
- **Best value picks:** After converting odds to expected points, **[team]** offers
  the strongest value relative to its title odds because **[reasoning]** — a team
  the model rates higher than its reputation suggests.
- **Allocation:** The fair-value table concentrates **[__]%** of the pot in the top
  six teams, leaving **[__]%** spread across dark horses.

## How to reproduce

```bash
pip install -r requirements.txt

# 1. Download results.csv into data/ (see link above)
# 2. Copy the example draw and edit it with the real 48 teams / 12 groups
cp config/groups.example.json config/groups.json

# 3. (optional) exploratory plots
python -m src.eda

# 4. Run the full pipeline
python -m src.run_pipeline
```

The pipeline prints the top 20 teams and writes the full table to
`output/allocation.csv`. If a team in your draw isn't in the data, the script
tells you exactly which name to fix.

## Limitations (worth being able to discuss)

- International football is **sparse per team** — even strong sides play relatively
  few competitive matches, so coefficients carry real uncertainty.
- The knockout bracket uses **seeded best-vs-worst pairing**, a simplification of
  the official bracket (whose pairings depend on which third-placed teams qualify).
- The model uses **no player-level information** (injuries, form, squad changes),
  only team results. Tournament-specific factors aren't captured.
- Penalty shootouts are modeled as a strength-weighted coin flip, not a separate
  process.

## Repo structure

```
world-cup-pool-allocator/
├── README.md
├── requirements.txt
├── data/                       # gitignored — download results.csv here
├── config/
│   └── groups.example.json     # template draw; copy to groups.json and edit
├── src/
│   ├── data_prep.py            # load, weight, reshape
│   ├── model.py                # weighted Poisson goals model
│   ├── simulate.py             # Monte Carlo tournament
│   ├── allocate.py             # probabilities → allocation
│   ├── eda.py                  # exploratory plots
│   └── run_pipeline.py         # end-to-end
├── visuals/                    # chart exports for the README
└── output/                     # allocation.csv (gitignored)
```
