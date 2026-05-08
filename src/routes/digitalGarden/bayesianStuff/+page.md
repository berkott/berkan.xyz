# Bayesian inference vs prediction
Bayesian inference is updating your belief about unknown quantities. You start with a prior $p(\theta)$, and then you get data $\mathcal{D}$ and you use it to compute the posterior $p(\theta | \mathcal{D})$.

Bayesian prediction is making a prediction on a test data point $x$. You compute the posterior predictive distribution by integrating the posterior over your parameters and sample from it.