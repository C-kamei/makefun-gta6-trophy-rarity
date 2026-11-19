"""GTA VI trophy-rarity forecaster (gta6-trophy-rarity-forecast-01).

Model: damped log-time trend in logit space + honest spread.

Why this shape (from the kit's RULES and a live read of Exophase):
- Exophase refreshes a game's counts hours apart, not continuously. Between
  refreshes the page is frozen; at a refresh the tracked-player count jumps
  and almost every rate moves at once (launch night: 638 players tracked,
  then 3,868 at the first refresh, every rate but the one at 100% lower).
  A live read on 2026-11-19 ~17:20 UTC showed the same pattern over ~41 h:
  median logit move -2.9, mean -2.6, 98% of trophies down. The dominant
  early force is DILUTION: the tracked population explodes after launch
  while only a fraction of the newcomers have progressed.
- So a pure persistence forecast is wrong in a predictable direction, and
  the house curve baseline (straight line of logit rate vs ln(1+hours))
  is the right family but extrapolates its slope raw over 4.5 days.
  This entry fits the same log-time line per trophy over the last 24 h of
  reads, then (a) shrinks the slope by its own statistical significance
  (weak trends fall back toward persistence), (b) damps it further the
  further the target lies beyond the observed range -- launch-week
  dilution is front-loaded (tracked players grew ~100x in the first 41 h,
  and will not repeat that in the next 4.5 days), so the early slope
  overstates the remaining drift -- and (c) caps the total logit move,
  so noisy trophies cannot run away.
- Spread: a random-walk term (the house baseline's 0.5*sqrt(days)), the
  fit's residual std, the slope's standard error times the extrapolation
  distance, and a refresh-jump floor (0.35 logit) for the discrete
  Exophase refreshes. Floored at 0.9 logit: the practice refresh moved
  rates ~1.5 logit on average, and the house baseline's 90-minute spread
  (0.125) covered none of it. The floor is what makes the practice loss
  beat persistence; the trend is what must beat it on the final.

Standard library only, deterministic, no network, no model calls.
Grader: `python3 -P -s /bot/main.py` (Python 3.13), 8 s / case.
Local run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch day begins 2026-11-19T00:00:00Z; the log-time axis is anchored there.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk volatility of a rate's logit, per sqrt-day (house baseline's).
RW_SIGMA = 0.5
# Extra spread for one discrete Exophase refresh jump, in logit units.
REFRESH_FLOOR = 0.35
# Spread is never narrower than this (practice refreshes move ~1.5 logit)
# and never wider than this (a forecast that wide says nothing).
SPREAD_MIN = 0.9
SPREAD_MAX = 1.8
# Largest logit move the trend is allowed to claim from the last read.
# Launch-week dilution is front-loaded (tracked players grew ~100x in the
# first 41 h); extrapolating the early slope raw over 4.5 more days would
# overshoot, so the drift is capped as well as damped.
MAX_DRIFT = 2.0
# Hours of history the trend fit uses (the house curve baseline's window).
FIT_WINDOW_HOURS = 24.0


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as the starter does (JS Math.round)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t: datetime) -> float:
    return max(0.0, (t - LAUNCH).total_seconds() / 3600.0)


def log_time(t: datetime) -> float:
    return math.log1p(hours_since_launch(t))


def ols(xs, ys):
    """Least-squares line y ~ a + b*x. Returns (a, b, resid_std, slope_se)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 3 or sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    s2 = sum(r * r for r in resid) / dof
    s_r = math.sqrt(s2)
    se_b = math.sqrt(s2 / sxx) if sxx > 0 else 0.0
    return a, b, s_r, se_b


def forecast_trophy(series, x_target, horizon_days):
    """series: list of (log_time, logit_rate, read_time) for one trophy.
    Returns the five quantiles (percent) at the case target."""
    last_x, last_y, _ = series[-1]
    dx = x_target - last_x

    # Trend fit on the last FIT_WINDOW_HOURS of reads (at least 3 points).
    cutoff = series[-1][2].timestamp() - FIT_WINDOW_HOURS * 3600
    window = [p for p in series if p[2].timestamp() >= cutoff]
    if len(window) < 3:
        window = series
    xs = [p[0] for p in window]
    ys = [p[1] for p in window]
    _, b, s_r, se_b = ols(xs, ys)

    # Shrink the slope by its significance, damp by extrapolation distance.
    if se_b > 0:
        t_stat = abs(b) / se_b
        shrink = t_stat * t_stat / (t_stat * t_stat + 4.0)
    else:
        shrink = 1.0 if b != 0.0 else 0.0
    damp = 1.0 / (1.0 + 0.5 * max(0.0, dx))
    drift = shrink * damp * b * dx
    drift = max(-MAX_DRIFT, min(MAX_DRIFT, drift))
    y_med = last_y + drift

    # Spread: random walk + fit residual + slope uncertainty + refresh floor.
    rw = RW_SIGMA * math.sqrt(horizon_days)
    spread = math.sqrt(rw * rw + s_r * s_r + (se_b * dx) ** 2 + REFRESH_FLOOR ** 2)
    spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))

    # A trophy pinned at ~100% stays pinned: its rate cannot fall far.
    # Pin the median at the last rate itself, not at 99.9: on the logit
    # scale 99.90 vs 99.99 is a 2.3-unit miss, and this trophy ("Complete
    # Gone Fishing") has sat at 100 -> 99.99 from launch night to now.
    last_rate = sigmoid(last_y)
    if last_rate >= 99.9:
        y_med = last_y
        spread = 0.15

    qs = [sigmoid(y_med + z * spread) for z in Z]
    qs = [round4(min(99.99, max(0.01, q))) for q in qs]
    # Rounding can only produce equal neighbours; enforce non-decreasing.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    x_target = log_time(target)
    horizon_days = case["horizon_minutes"] / 1440.0

    reads = []
    for read in history:
        reads.append((parse_time(read["at"]), read["rates"]))
    reads.sort(key=lambda r: r[0])

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        series = []
        for at, rates in reads:
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            series.append((log_time(at), logit(rate), at))
        if not series:
            # Never observed: neutral 0.5% with a wide spread.
            qs = [round4(sigmoid(logit(0.5) + z * SPREAD_MAX)) for z in Z]
            rows.append({"id": trophy["id"], "q": qs})
            continue
        rows.append({"id": trophy["id"], "q": forecast_trophy(series, x_target, horizon_days)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
