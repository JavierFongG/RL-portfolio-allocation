# Environment

This folder contains the market simulator and feasibility logic used during RL training and evaluation.

## Files

### `portfolio_env.py`

Defines the Gymnasium environment. Each step advances one trading day, applies a portfolio weight vector, computes realized returns, evaluates the reward, updates portfolio value, and produces the next state.

### `transaction_cost_model.py`

Implements a linear turnover-based cost:

`cost = c * sum(|w_new - w_old|)`

This is a standard baseline approximation for slippage and fees.

### `risk_constraint.py`

Projects candidate weights onto a feasible region under a maximum portfolio volatility constraint. The implementation uses simplex projection and projected descent rather than a full quadratic programming solver.

## Theory behind the environment design

### Markovian approximation

Real markets are not truly Markovian, but RL environments usually approximate the decision process as one where the current engineered state contains enough information for the next action. The state builder therefore packages recent per-asset history, pooled market context, and risk metadata into one observation.

### Daily stepping

A daily step is a practical granularity:

- data are widely available
- transaction cost modeling is simpler than at intraday frequency
- the signal-to-noise ratio is usually better than at very short horizons

### Linear transaction costs

True trading frictions are nonlinear and depend on liquidity, spread, and market impact. A linear turnover penalty is still useful because it captures the first-order fact that excessive rebalancing is undesirable.

### Volatility feasibility

The risk projection layer exists because a policy can otherwise discover very aggressive allocations that look attractive under short-horizon reward signals. Constraining portfolio volatility is a simple way to inject ex ante risk discipline into the action pipeline.

## Why the environment returns logging information

The `info` dictionary is essential in finance. A reward alone is not enough to diagnose behavior. Logging portfolio value, turnover, gross return, trading cost, and reward terms helps separate “the policy made money” from “the policy overtraded,” “the policy concentrated risk,” or “the policy only performed through one regime.”

## References

- Gymnasium API: https://gymnasium.farama.org/
- Markowitz, H. (1952), portfolio selection: https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
