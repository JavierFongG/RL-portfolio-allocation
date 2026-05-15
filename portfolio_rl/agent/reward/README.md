# Reward

This folder defines how the portfolio agent is incentivized.

## Files

### `base_reward.py`

Defines the common reward interface and a reset hook for stateful reward objects.

### `sharpe_reward.py`

Implements a rolling Sharpe-style reward based on realized portfolio returns. It is useful as a classic risk-adjusted baseline.

### `cvar_reward.py`

Implements a return-minus-tail-risk reward using Conditional Value-at-Risk. It penalizes adverse tail outcomes more explicitly than variance-only metrics.

### `composite_reward.py`

Implements the main portfolio reward currently used by the environment:

`R = r_p - lambda_vol * sigma_p - lambda_turn * sum(|delta w|) - lambda_hhi * HHI(w)`

where:

- `r_p = log(1 + sum_i w_i r_i)`
- `sigma_p` is the rolling standard deviation of portfolio log returns over a configurable window
- `sum(|delta w|)` is the L1 turnover between current and previous weights
- `HHI(w) = sum_i w_i^2` measures concentration

Each term is normalized before its coefficient is applied. The normalization divides by a running exponential moving average of absolute magnitude, using decay `0.99`.

## Why this reward design

### Log return

Using log return improves numerical behavior under compounding and makes multi-period aggregation conceptually cleaner than using raw simple returns directly.

### Volatility penalty

A return-only objective often pushes the agent toward unstable allocations. Penalizing realized rolling volatility adds a local risk-awareness term.

### Turnover penalty

Real portfolios pay costs when they rebalance. Even if the environment already tracks transaction costs in wealth evolution, turnover should still appear in the reward when the objective is meant to discourage noisy reallocations.

### HHI concentration penalty

`HHI = sum(w_i^2)` is a compact concentration measure. It is minimum at equal weight `1/N` and maximum at `1` when the portfolio is fully concentrated in one asset. Penalizing HHI encourages diversification without forcing equal weights.

### EMA scale normalization

The raw scales of log return, volatility, turnover, and HHI differ substantially. Without normalization, the choice of `lambda` values would be dominated by unit conventions rather than economic preference. The EMA-based scale tracker makes the terms more comparable while remaining adaptive to regime shifts.

## Important interpretation note

The normalized composite reward is a control objective, not a direct investor utility function. It is designed to make training stable and preferences explicit. In practice, the reported backtest metrics should still include out-of-sample return, Sharpe, drawdown, turnover, and concentration.

## References

- Sharpe (1994), Sharpe ratio: https://doi.org/10.3905/jpm.1994.409501
- Rockafellar and Uryasev (2000), CVaR: https://doi.org/10.21314/JOR.2000.038
- U.S. Department of Justice, HHI overview: https://www.justice.gov/atr/herfindahl-hirschman-index
