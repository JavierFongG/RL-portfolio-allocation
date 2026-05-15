# Fusion

This folder turns a set of per-asset latent vectors into the state consumed by the environment and policy.

## Why a fusion layer exists

The encoder produces one latent vector per asset. A policy, however, needs both:

- asset-specific information for deciding relative weights
- market-level context for judging the current cross-sectional regime

The fusion layer is where those two requirements are combined.

## Files

### `attention_pooler.py`

Applies multi-head self-attention to a variable-length set of latent vectors and then pools them into one summary vector. The point is not to model temporal order but cross-sectional interactions between assets at the same decision time.

### `state_builder.py`

Coordinates the entire state assembly process:

1. fetch recent history for each asset
2. normalize using the fitted train-only normalizer
3. encode each window into a latent vector
4. pool the latent set with attention
5. concatenate flattened asset latents, pooled context, and a scalar risk feature

## Theory behind the choices

### Self-attention over assets

Attention is useful here because the relevance of one asset may depend on the rest of the universe. For example, a commodity latent may mean something different in a regime where all risk assets co-move strongly than in a regime with dispersed signals. Self-attention lets each asset representation interact with the others before pooling.

### Variable-length handling

A portfolio universe may change size across experiments or inference scenarios. The attention pooler is written to handle arbitrary `M`, the number of assets, as long as the latent dimension stays fixed.

### Keep both local and global information

The state keeps the per-asset latent block and the pooled summary. This is deliberate:

- the per-asset block is useful for scoring assets independently
- the pooled vector gives the critic or policy a market-wide summary

If only a pooled vector were kept, relative asset identities would be compressed away too aggressively.

## References

- Vaswani et al. (2017), attention: https://arxiv.org/abs/1706.03762
- Recent portfolio RL examples using attention-like cross-sectional modeling: https://arxiv.org/abs/2510.06466
