# Fun with bivariate Gaussians

Suppose I give you the following bivariate Gaussian distribution, 
$$
\mathcal{N} \Big(\begin{bmatrix} 0 \\ 1 \end{bmatrix}, \begin{bmatrix} 3 \ & 1 \\ 1 \ & 2 \end{bmatrix} \Big).
$$
Then I say "tell me sonny, if I drew a bunch of points from this distribution and plotted them, where about would they be?" I want a quick answer! You don't even have time to run a Python simulation. What do you do?

## Background
The mean vector is easy to understand, it's just going to be the center of your distribution. The covariance matrix is the tricky part. Let's recall some properties. 
- First off, it's symmetric. Why? It comes straight from the definition of covariance and the commutativity of multiplication. 
- Also, it's positive semi-definite, meaning $v^\top \Sigma v \ge 0$ for any $v$. Why? Because $v^\top \Sigma v = \text{Var}(v^\top X) \ge 0$.

Fun fact, the eigenvectors of a symmetric matrix are orthogonal!

## Eigen stuff
Observe 


Eigenvalue stuff