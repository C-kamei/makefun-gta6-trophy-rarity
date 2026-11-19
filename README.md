# makefun-gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/C-kamei/makefun-gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/C-kamei/makefun-gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun's [“Forecast how many GTA VI players earn each trophy in week one”](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250 in USDC). It reads a case file (`makefun/gta6-trophies-case/v1`) from `$MAKEFUN_INPUT_DIR` (or `/input`) and, for each case, writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (or `/output`): five quantiles (0.05/0.25/0.5/0.75/0.95) of each trophy's Exophase earn rate at the target. Method: per-trophy damped log-time trend in logit space — a least-squares line of logit(rate) against ln(1 + hours since launch) over the last 24 h of reads, shrunk by its own significance and damped by extrapolation distance, because launch-week dilution (tracked players grew ~100x in the first 41 h, median logit move −2.9) is front-loaded — plus an honest spread (random walk + fit residual + slope uncertainty + a refresh-jump floor). On the kit's practice cases it scores ~0.85 under a uniform −1.5-logit refresh shift, vs ~1.41 for the house persistence baseline. No network, Python standard library only.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=fk5m5avs&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
