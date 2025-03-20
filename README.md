# Debunking the Bias Correction Necessity Myth

Implementations of Adam(W) all apply bias-correction to the EMA statistics. The claim is that doing so prevents a bias in the early updates of the optimizer. Most resources quote some variation of the following proof when it comes to claiming the necessity of bias correction:

$$E[m^{t}] = E\left[ (1-\beta_1) \sum_{i=0}^{t-1} {\beta_1^{t-i-1} g_i}\right]$$

If one makes the assumption (which is surely false at early steps) that $E[g_i] \approx E[g_t]$ for small $i$, then the expression simplifies to:

$$E[m^{t}] \approx E[g_t] (1-\beta_1^t)$$

And thus one should divide across by $(1-\beta_1^t)$.

The exact same argument is given to justify the bias correction term for the second moment EMA.

However, even the torch implementation of SGD with momentum just initializes to the gradient rather than zero and does not bias correct.

We demonstrate that across a wide range of models and tasks, simply initializing the moments to their initial values rather than to zero removes the necessity of bias correction.
