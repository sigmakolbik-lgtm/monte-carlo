# -*- coding: utf-8 -*-
"""
Monte Carlo simulations: coin flip and dice roll.

Sections
  1. Coin flip: law of large numbers
  2. One die: frequencies + chi-square goodness-of-fit test
  3. Two dice: distribution of the sum (empirical vs theory)
  4. Expected value: convergence, and a "casino game" ruin demo
  5. Chart (PNG) drawn with Pillow

Run:  python monte_carlo.py
"""
from __future__ import annotations

import math
import os

import numpy as np

RNG_SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# core model
# --------------------------------------------------------------------------
def flip_coins(n: int, p: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a boolean array: True = heads, with probability p."""
    rng = rng or np.random.default_rng(RNG_SEED)
    return rng.random(n) < p


def roll_dice(n: int, sides: int = 6, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return n fair rolls of a `sides`-sided die (values 1..sides)."""
    rng = rng or np.random.default_rng(RNG_SEED)
    return rng.integers(1, sides + 1, size=n)


def ascii_hist(labels, values, width: int = 42) -> None:
    peak = max(values) if len(values) else 1.0
    for lab, v in zip(labels, values):
        bar = "#" * int(round(v / peak * width)) if peak else ""
        print(f"  {str(lab):>3} | {bar:<{width}} {v:.5f}")


# --------------------------------------------------------------------------
# 1. law of large numbers
# --------------------------------------------------------------------------
def section_lln(rng: np.random.Generator) -> None:
    print("=" * 70)
    print("1. COIN FLIP - the more flips, the closer to the true 50%")
    print("=" * 70)
    print(f"{'n':>12} {'heads':>12} {'frequency':>10} {'|error|':>9} {'2*SE':>8}")
    for n in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
        flips = flip_coins(n, rng=rng)
        heads = int(flips.sum())
        freq = heads / n
        se = math.sqrt(0.25 / n)                 # std error of the proportion
        print(f"{n:>12,} {heads:>12,} {freq:>10.5f} {abs(freq - 0.5):>9.5f} {2 * se:>8.5f}")
    print("\n  Note: 2*SE is the expected size of a random deviation.")
    print("  n = 10 flips can easily give 70% heads; n = 1,000,000 cannot.\n")


# --------------------------------------------------------------------------
# 2. one die
# --------------------------------------------------------------------------
def section_one_die(rng: np.random.Generator, n: int = 600_000) -> None:
    print("=" * 70)
    print(f"2. ONE FAIR DIE - {n:,} rolls, each face should get 1/6 = 0.16667")
    print("=" * 70)
    rolls = roll_dice(n, 6, rng)
    counts = np.bincount(rolls, minlength=7)[1:]          # faces 1..6
    freqs = counts / n
    for face, (c, f) in enumerate(zip(counts, freqs), start=1):
        print(f"  face {face}: {c:>8,} rolls   frequency {f:.5f}   error {f - 1/6:+.5f}")

    expected = np.full(6, n / 6)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    crit = 11.07                                # chi-square, df=5, 95% level
    print(f"\n  chi-square statistic = {chi2:.3f}   (critical value at 95% = {crit})")
    verdict = "looks fair" if chi2 < crit else "suspicious - deviation too large"
    print(f"  verdict: {verdict}\n")
    print(f"  mean of {n:,} rolls = {rolls.mean():.5f}   theoretical E[X] = 3.5\n")


# --------------------------------------------------------------------------
# 3. two dice
# --------------------------------------------------------------------------
def section_two_dice(rng: np.random.Generator, n: int = 400_000):
    print("=" * 70)
    print(f"3. TWO DICE - {n:,} throws, distribution of the sum")
    print("=" * 70)
    a = roll_dice(n, 6, rng)
    b = roll_dice(n, 6, rng)
    sums = a + b
    counts = np.bincount(sums, minlength=13)[2:]          # sums 2..12
    emp = counts / n
    faces = np.arange(2, 13)
    theory = np.array([(6 - abs(s - 7)) / 36 for s in faces])

    print(f"{'sum':>4} {'ways':>5} {'theory':>9} {'empirical':>11} {'error':>9}")
    for s, t, e in zip(faces, theory, emp):
        ways = int(round(t * 36))
        print(f"{s:>4} {ways:>5} {t:>9.5f} {e:>11.5f} {e - t:>+9.5f}")

    chi2 = float((((counts - n * theory) ** 2) / (n * theory)).sum())
    print(f"\n  chi-square = {chi2:.3f}  (df=10, critical value at 95% = 18.31)")
    print(f"  most likely sum is 7 (6 of 36 ways = {6/36:.4f}); "
          f"empirically {emp[faces == 7][0]:.4f}\n")
    print("  empirical shape:")
    ascii_hist(faces, emp)
    print()
    return faces, emp, theory, n


# --------------------------------------------------------------------------
# 4. expected value
# --------------------------------------------------------------------------
def section_expected_value(rng: np.random.Generator) -> None:
    print("=" * 70)
    print("4. EXPECTED VALUE - running mean converges to E[X] = 3.5")
    print("=" * 70)
    rolls = roll_dice(1_000_000, 6, rng)
    running = np.cumsum(rolls) / np.arange(1, rolls.size + 1)
    for n in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
        print(f"  mean of first {n:>9,} rolls = {running[n - 1]:.6f}")
    print()

    print("=" * 70)
    print("5. A FAIR-LOOKING BUT LOSING GAME (the casino always wins)")
    print("=" * 70)
    print("  Rules: you bet $1 on one face. Correct face -> +$4, wrong -> -$1.")
    ev = 4 * (1 / 6) - 1 * (5 / 6)
    print(f"  theoretical expectation per bet = {ev:+.6f} $")
    bets = 100_000
    outcomes = roll_dice(bets, 6, rng)
    target = 6
    profit = np.where(outcomes == target, 4.0, -1.0)
    bankroll = np.cumsum(profit)
    print(f"  after {bets:,} bets: net {bankroll[-1]:+,.0f} $, "
          f"average per bet {profit.mean():+.5f} $")
    print(f"  broke after bet #{int(np.argmax(bankroll < -100)) + 1:,} (bankroll below -$100)"
          if bankroll[-1] < -100 else "  you got lucky this run")
    print()


# --------------------------------------------------------------------------
# 5. chart
# --------------------------------------------------------------------------
def make_chart(path: str, ns, reps_fracs, faces, emp, theory, n_throws: int) -> str:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1240, 680
    BG, FG = (255, 255, 255), (30, 33, 38)
    GRID = (225, 229, 235)
    BLUE, LBLUE = (37, 99, 235), (170, 200, 250)
    ORANGE, GRAY = (234, 88, 12), (203, 209, 218)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    def font(size, bold=False):
        name = "arialbd.ttf" if bold else "arial.ttf"
        try:
            return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", name), size)
        except OSError:
            return ImageFont.load_default()

    f_small, f_axis = font(13), font(14)
    f_title, f_head = font(17, True), font(21, True)

    d.text((W // 2, 28), "Monte Carlo simulation: coin flip and dice roll",
           font=f_head, fill=FG, anchor="mm")

    # ---------------- left panel: convergence ----------------
    x0, y0, x1, y1 = 96, 96, 596, 560
    d.rectangle([x0, y0, x1, y1], outline=GRID)
    d.text(((x0 + x1) / 2, 74), "Coin flip: fraction of heads vs number of flips",
           font=f_title, fill=FG, anchor="mm")

    lon, hin = 2.0, 6.0                      # log10(n) range
    ylo, yhi = 0.40, 0.60

    def X(n):
        return x0 + (math.log10(n) - lon) / (hin - lon) * (x1 - x0)

    def Y(v):
        return y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)

    for v in (0.40, 0.45, 0.50, 0.55, 0.60):
        y = Y(v)
        d.line([x0, y, x1, y], fill=GRID if v != 0.50 else (150, 158, 170))
        d.text((x0 - 10, y), f"{v:.2f}", font=f_axis, fill=FG, anchor="rm")
    for e in range(2, 7):
        xx = X(10 ** e)
        d.line([xx, y0, xx, y1], fill=GRID)
        d.text((xx, y1 + 18), f"$10^{e}$" if False else f"1e{e}", font=f_axis,
               fill=FG, anchor="mm")
    d.text(((x0 + x1) / 2, y1 + 44), "number of flips (log scale)",
           font=f_axis, fill=FG, anchor="mm")
    d.text((x0 - 10, y0 - 16), "heads", font=f_axis, fill=FG, anchor="rm")

    # +/- 2 SE band (theory)
    band_n = np.logspace(lon, hin, 200)
    upper = [Y(min(yhi, 0.5 + 1 / math.sqrt(n))) for n in band_n]
    lower = [Y(max(ylo, 0.5 - 1 / math.sqrt(n))) for n in band_n]
    d.polygon(list(zip([X(n) for n in band_n], upper)) + list(zip([X(n) for n in band_n], lower)),
              fill=(238, 242, 250))

    # dots of individual experiments
    for i, n in enumerate(ns):
        xx = X(n)
        for f in reps_fracs[:, i]:
            yy = Y(f)
            d.ellipse([xx - 2.5, yy - 2.5, xx + 2.5, yy + 2.5], fill=LBLUE)
    # mean of the runs
    pts = [(X(n), Y(reps_fracs[:, i].mean())) for i, n in enumerate(ns)]
    d.line(pts, fill=BLUE, width=3)
    for px, py in pts:
        d.ellipse([px - 4.5, py - 4.5, px + 4.5, py + 4.5], fill=BLUE)
    d.line([x0, Y(0.5), x1, Y(0.5)], fill=(120, 128, 140), width=1)

    ly = y0 + 14
    d.ellipse([x0 + 12, ly - 4, x0 + 20, ly + 4], fill=LBLUE)
    d.text((x0 + 28, ly), "one experiment", font=f_small, fill=FG, anchor="lm")
    ly += 20
    d.ellipse([x0 + 12, ly - 4, x0 + 20, ly + 4], fill=BLUE)
    d.text((x0 + 28, ly), "mean of 30 experiments", font=f_small, fill=FG, anchor="lm")
    ly += 20
    d.rectangle([x0 + 11, ly - 6, x0 + 21, ly + 6], fill=(238, 242, 250), outline=GRID)
    d.text((x0 + 28, ly), "theoretical +/-2 standard errors", font=f_small, fill=FG, anchor="lm")

    # ---------------- right panel: two dice ----------------
    rx0, ry0, rx1, ry1 = 736, 96, 1180, 560
    d.text(((rx0 + rx1) / 2, 74), "Two dice: distribution of the sum",
           font=f_title, fill=FG, anchor="mm")
    rmax = 0.20

    def RY(v):
        return ry1 - v / rmax * (ry1 - ry0)

    for v in (0.0, 0.05, 0.10, 0.15, 0.20):
        y = RY(v)
        d.line([rx0, y, rx1, y], fill=GRID)
        d.text((rx0 - 10, y), f"{v:.2f}", font=f_axis, fill=FG, anchor="rm")
    d.line([rx0, ry1, rx1, ry1], fill=(150, 158, 170))

    slot = (rx1 - rx0) / len(faces)
    bw = slot * 0.62
    for i, s in enumerate(faces):
        cx = rx0 + slot * (i + 0.5)
        # theoretical bar (gray)
        d.rectangle([cx - bw / 2, RY(theory[i]), cx + bw / 2, ry1], fill=GRAY)
        # empirical bar (orange, narrower)
        d.rectangle([cx - bw / 4, RY(emp[i]), cx + bw / 4, ry1], fill=ORANGE)
        d.text((cx, ry1 + 18), str(s), font=f_axis, fill=FG, anchor="mm")
    d.text(((rx0 + rx1) / 2, ry1 + 46), "sum of two dice", font=f_axis, fill=FG, anchor="mm")
    d.text((rx0 - 10, ry0 - 16), "probability", font=f_axis, fill=FG, anchor="rm")

    ly = ry0 + 14
    d.rectangle([rx0 + 11, ly - 6, rx0 + 21, ly + 6], fill=GRAY)
    d.text((rx0 + 28, ly), "theory (ways / 36)", font=f_small, fill=FG, anchor="lm")
    ly += 20
    d.rectangle([rx0 + 11, ly - 6, rx0 + 21, ly + 6], fill=ORANGE)
    d.text((rx0 + 28, ly), f"empirical, {n_throws:,} throws", font=f_small, fill=FG, anchor="lm")

    img.save(path)
    return path


# --------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    section_lln(rng)
    section_one_die(rng)
    faces, emp, theory, n_throws = section_two_dice(rng)
    section_expected_value(rng)

    ns = np.array([100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])
    reps = 30
    reps_fracs = np.vstack([np.array([flip_coins(int(n), rng=rng).mean() for n in ns])
                            for _ in range(reps)])

    out = os.path.join(HERE, "coin_dice_monte_carlo.png")
    make_chart(out, ns, reps_fracs, faces, emp, theory, n_throws)
    print(f"Chart saved to: {out}")


if __name__ == "__main__":
    main()
