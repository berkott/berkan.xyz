One way I think about deep learning, inspired by discussions with [Dhruva](https://dkarkada.xyz/), [Joey](https://github.com/JoeyTurn/), and [Jamie](https://james-simon.github.io/), is that it's just a combination of hyper-parameter selection, a model, an optimizer, and data. That's roughly how this page is organized. Links will be added as I read them and create ntoes

# Core
## Hyper-parameter selection
This section is about scaling, initialization, and hyper-parameter selection.
- LeCun init
- Hyper param transfer
  - muP read TP4
- Scaling laws

## Models
This section is based on the pareto frontier of models of deep learning from [Jamie's](https://james-simon.github.io) thesis. I am just going down the frontier from least realistic to most realistic.
![Pareto frontier](/images/digitalGarden/paretoFrontier.png "Pareto Frontier")
- Linear regression
- Kernel regression
  - Eigenframework
- Random feature models
  - Rahimi and Recht 2007 paper
  - Dimension-free deterministic equivalents and scaling laws for random feature regression
- NNGP / NTK
  - [Deep Neural Networks as Gaussian Processes notes](/digitalGarden/NNGP)
  - [Limitations of kerenels notes](/digitalGarden/kernelLimitations)
  - Jacot's paper
  - The lazy (NTK) and rich (µP) regimes: A gentle tutorial
  - On Lazy Training in Differentiable Programming
- Linear networks
  - The Implicit Bias of Gradient Descent on Separable Data
  - Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning
  - Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity
  - Neural networks and principal component analysis: Learning from examples without local minima.
- RFMs
  - Average gradient outer product as a mechanism for deep neural collapse
- Mean field / muP
  - Look at mei montanari theo + his lecture notes
  - 6 lectures on linearized networks
  - Maybe watch talks
  - TP4
- MLPs
  - Scaling MLPs: A Tale of Inductive Bias
  - On the non-universality of deep learning: quantifying the cost of symmetry
  - SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics
  - Feature emergence via margin maximization: case studies in algebraic tasks
  - Memorization capacity
  - SCALING LAWS FOR ASSOCIATIVE MEMORIES
  - Learning Associative Memories with Gradient Descent
  - Find stuff about storage capacity in models
- Transformers
  - Transformers Learn Shortcuts to Automata
  - Find stuff about storage capacity in models
  - Clayton representational work
  - Will Merrill smth
- SSM
  - Expressivity limitations
  - Modifying by idk smth
  - Figure of variants of SSMs
  - Comp eff on GPU (Damek)

## Optimization
- Understanding all the optimizers
  - SGD
    - Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit
  - RMS Prop
  - Momentum
  - Adam + W
  - Muon
- EoS
  - Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability
  - Central flows
- Loss spikes
  - Small-scale proxies for large-scale Transformer training instabilities
- Other tricks
  - ADDING GRADIENT NOISE IMPROVES LEARNING FOR VERY DEEP NETWORKS

## Imitation learning
- [Classical learning theory with autoregressive models](/digitalGarden/autoRegCLT)
- Black box theoretical studies
  - Computational-Statistical Tradeoffs at the Next-Token Prediction Barrier: Autoregressive and Imitation Learning under Misspecification
  - Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment
  - Taming Imperfect Process Verifiers: A Sampling Perspective on Backtracking
  - On the Query Complexity of Verifier-Assisted Language Generation

## Data
- Pre-training data distribution
- Post-training data distribution

# Other
## Hardware aware
- Albert Gu Flash attention
- Horace He
- Dion: Distributed Orthonormalized Updates
- The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm

## Distillation
- 

## Post-training

## Mechanistic interpretability

## What does it mean to understand? What are we looking for from a theory of deep learning? What could be a unified theory of deep learning?
Eliminating hyper parameters, very good theory empirics match to show we fully understood everything

# References
- https://analyticinterp.github.io/
- ![The Graph](/images/digitalGarden/theGraph.png "The Graph")
- https://surbhi18.github.io/FoMML/calendar/
- https://blakebordelon.github.io/summary.html