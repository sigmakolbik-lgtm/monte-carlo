# Monte Carlo: coin flip and dice roll

[Русская версия](README.md)

An educational project on the Monte Carlo method: a computer simulation of random
experiments, and a check of how closely the empirical results converge to the
predictions of probability theory.

![Simulation chart](coin_dice_monte_carlo.png)

## What the project demonstrates

| # | Experiment | Idea being tested |
|---|---|---|
| 1 | Coin flips | Law of large numbers, standard error `1/(2*sqrt(n))` |
| 2 | One die, 600,000 rolls | Uniform distribution, chi-square goodness-of-fit test |
| 3 | Two dice, 400,000 throws | Distribution of the sum, its triangular shape |
| 4 | Running mean | Expected value `E[X] = 3.5` |
| 5 | A game with negative EV | Why the house wins in the long run |

## Key results

**Coin — the more flips, the closer to 0.5:**

| n (flips) | fraction of heads | error | 2·SE |
|---|---|---|---|
| 10 | 0.40000 | 0.10000 | 0.316 |
| 1,000 | 0.50000 | 0.00000 | 0.032 |
| 1,000,000 | 0.50061 | 0.00061 | 0.001 |

**One die:** chi-square statistic = 6.19 against a critical value of 11.07
(df = 5, 95% confidence level) — the deviation is consistent with chance, so the die
is fair. Mean of 600,000 rolls: 3.50378 versus the theoretical 3.5.

**Two dice:** the sum `s` occurs in `6 - |s - 7|` ways out of 36, so the maximum falls
on 7 (16.67%). The empirical result matched theory to the third decimal place.

**A $1 game:** you bet on one face — the correct face pays `+$4`, a wrong one costs `-$1`.
Theoretical expectation `4*(1/6) - 1*(5/6) = -0.1667 $` per bet. Over 100,000 bets the
simulation returned `-0.16635 $` per bet — a loss predicted by the formula, not by bad luck.

## Running it

```bash
pip install -r requirements.txt
python monte_carlo.py
```

The script prints the tables to the console and saves the chart as
`coin_dice_monte_carlo.png`.

Python 3.10+ is required (the type hints use the `X | None` syntax).

## Changing the experiment

```python
flip_coins(n, p=0.5)     # p=0.6  — a biased coin
roll_dice(n, sides=6)    # sides=20 — a twenty-sided die
```

The random number generator is seeded with a fixed value (`RNG_SEED = 42`), so the
results are reproducible: anyone running the script gets the same numbers. Change the
seed and the numbers shift slightly, while the statistical conclusions stay the same.

## Possible next steps

- Estimating pi with the Monte Carlo method (points inside a square and a circle).
- The birthday paradox and the Monty Hall problem.
- Sums of three or more dice — convergence to the normal distribution (CLT).
- Hypothesis testing: t-test, confidence intervals.
- Geometric Brownian motion as a price model — a step towards mathematical finance.

## License

MIT — see [LICENSE](LICENSE).
