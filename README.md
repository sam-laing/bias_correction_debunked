# Debunking the Bias Correction Necessity Myth

Implementations of Adam(W) all apply bias-correction to the exponential moving average (EMA) statistics. The claim is that doing so prevents a bias in the early updates of the optimiser. Most resources [(for example)](https://stats.stackexchange.com/questions/366076/understanding-a-derivation-of-bias-correction-for-the-adam-optimizer) quote some variation of the following proof when it comes to claiming the necessity of bias correction:

$$E[m^{t}] = E\left[ (1-\beta_1) \sum_{i=0}^{t-1} {\beta_1^{t-i-1} g_i}\right]$$

If one makes the assumption (which is surely false at early steps) that $E[g_i] \approx E[g_t]$ for small $i$, then the expression simplifies to:

$$E[m^{t}] \approx E[g_t] (1-\beta_1^t)$$

And thus one should element-wise divide the $t^{\text{th}}$ moment by $(1-\beta_1^t)$ in order to remove said bias. 

The exact same argument is given to justify the bias correction term for the second moment EMA.

We demonstrate that across a wide range of models and tasks, simply initialising $m_0 <- g_0$ and $v_0 <- g_0^2$ removes the bias and allows AdamW to work just as well without bias correcting

## Datasets and Models

This repo currently consists of <fill in models and code here in a pretty way> 

## Hypothesis tests

### Is bias correction needed?

To rigorously prove the desired result, we conduct the following experiment:

- for a set task (defined model and data set), consider a set hyper-parameter configuration. 
- For a number of random seeds (controlling both torch behaviour and data loader sampling) train the exact same model to convergence with and without bias correction. For a random seed $i$, denote the accuracy of the model trained with bias correction $a_i$ and the model without bias correction $b_i$
- Obtain the following table for a collection of random seeds 

| Accuracy without bias correction | Accuracy with bias correction |
| -------------------------------- | ----------------------------- |
| a_1                              | b_1                           |
| a_2                              | b_2                           |
| ...                              | ...                           |
| a_n                              | b_n                           |

- One can then conduct a paired t-test on the samples (or some other statistical test if there are a sufficient number of seeds attempted)
- Formally, let $\theta := \frac{1}{n} \sum_{i=1}^{n} {(a_i - b_i)}$ with null hypothesis $\theta = 0$ and hopefully fail to reject the null hypothesis 
- *One question*: is it any less statistically valid if instead of having the exact same hyperparameter configuration for lots of random seeds if one just obtains lots of comparisons across different random seeds and tests on this instead. The upside here is no need to HP tune each dataset too much and the result might actually be stronger. Intuitively it shouldn't make a difference but somehow the collection of random seeds on one HP configuration feels stronger :////

### Is the $E[g_i] = E[g_t]$ assumption in any way valid? 
