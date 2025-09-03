A Gaussian process (GP) is a stochastic process (a collection of random variables indexed by time or space), such that every finite collection of those random variables has a multivariate normal distribution (copied from [Wikipedia](https://en.wikipedia.org/wiki/Gaussian_process#:~:text=a%20Gaussian%20process%20is%20a%20stochastic%20process%20(a%20collection%20of%20random%20variables%20indexed%20by%20time%20or%20space)%2C%20such%20that%20every%20finite%20collection%20of%20those%20random%20variables%20has%20a%20multivariate%20normal%20distribution.)). 

Some GPs (specifically stationary GPs, meaning it has a constant mean and covariance only depends on relative position of data points), have an explicit representation, where you can write it as 
$$
f(x) = g(R, x),
$$
for some random variables $R$ and some fixed deterministic function $g$ (I think this is true).

# Bayesian inference and prediction with a GP
We have a training set $\mathcal{D} = \{ (x^{(1)}, y^{(1)}), \ldots, (x^{(n)}, y^{(n)}) \}$, and we wish to make a Bayesian prediction (see my notes on [Bayesian stuff](/digitalGarden/bayesianStuff) for the difference between Bayesian inference and prediction) on a test sample $x$ using our GP, which is a distribution over functions.

We consider the following noise model, $y^{(i)} = g(x^{(i)}) + \epsilon^{(i)}$, where $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma_n^2)$.

Our joint prior on labels is
$$
\begin{bmatrix} Y \\ f \end{bmatrix} \sim \mathcal{N} \Big(\begin{bmatrix} m(X) \\ m(x) \end{bmatrix}, \begin{bmatrix} K(X, X) + \sigma_n^2 I_d \ & K(x, X) \\ K(x, X) & K(x, x) \end{bmatrix} \Big).
$$

From here, you just do the standard conditioning a joint Gaussian stuff ([Wikipedia page for this](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bayesian_inference:~:text=Conditional%20distributions)). Then you can write this as a new GP. This is your posterior predictive. 

You can place a GP prior over functions and then compute the posterior after seeing data to have some probability distribution for what the rest of the functin looks like. This is what they call Gaussian process regression. They get these cool visualizations from it like this from on [scikit-learn](https://scikit-learn.org/stable/modules/gaussian_process.html).
![Cool GP Regression](/images/digitalGarden/coolGPReg.png "Cool GP Regression")

But also, the argmax sample from the posterior predictive corresponds to the kernel ridge regression solution,
$$
\hat{f}(x) = K(x, X) (K(X, X) - \sigma_n^2 I_{d} )^{-1} Y,
$$
where $\sigma_n^2$ is your ridge penalty / noise. This is just the mean solution in the plot above, so if you don't care about the uncertainty quantification just do this ^.

TODO: Maybe an example with a linear kernel?