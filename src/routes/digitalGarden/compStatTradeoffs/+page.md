# Computational-Statistical Tradeoffs at the Next-Token Prediction Barrier: Autoregressive and Imitation Learning under Misspecification

Next token prediction withh log loss suffers from error amplification where errors compound and generation quality decreases as sequence length or horizon $H$ gets large. Theoretically need not occur in well-specified (realizable) settings but it does in misspecified (agnostic) settings.

Findings:
- Info theoretically possible to avoide error amplification in immitation learning
- Next-token prediction (a special case of behavior cloning) achieves $Theta(H)$ multiplicative approximation error with best hypothesis.
- Autoregressive linear models can trade statistical power for compute.

# Next token prediciton

$$
\pi
$$