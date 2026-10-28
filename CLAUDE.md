# TreeGround reporting rules

- Never report or claim a win on a validation split. Validation is for tuning only (thresholds, lambda, checkpoints).
- Headline claims use test splits only, compared with published test numbers under the same protocol.
- Baselines must use the same backbone size as our method (4B vs 4B, 8B vs 8B); larger models are reference rows only.
