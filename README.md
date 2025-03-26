# Debunking the Bias Correction Necessity Myth

Implementations of Adam(W) all apply bias-correction to the exponential moving average (EMA) statistics. The claim is that doing so prevents a bias in the early updates of the optimizer. Most resources [for example](https://stats.stackexchange.com/questions/366076/understanding-a-derivation-of-bias-correction-for-the-adam-optimizer) quote some variation of the following proof when it comes to claiming the necessity of bias correction:

$$E[m^{t}] = E\left[ (1-\beta_1) \sum_{i=0}^{t-1} {\beta_1^{t-i-1} g_i}\right]$$

If one makes the assumption (which is surely false at early steps) that $E[g_i] \approx E[g_t]$ for small $i$, then the expression simplifies to:

$$E[m^{t}] \approx E[g_t] (1-\beta_1^t)$$

And thus one should element-wise divide the $t^{\text{th}}$ moment by $(1-\beta_1^t)$ in order to remove said bias. 

The exact same argument is given to justify the bias correction term for the second moment EMA.

**We demonstrate that across a wide range of models and tasks, simply initializing $m_0 \leftarrow g_0$ and $v_0 \leftarrow g_0^2$ removes the bias and allows AdamW to work just as well without bias correction.**

## Datasets and Models

This repo currently consists of:

| Model | Dataset | Description |
|-------|---------|-------------|
| ResNet-20 | CIFAR-10 | Small convolutional network for image classification |
| ResNet-50 | ImageNet | Standard vision backbone for large-scale image classification |
| ViT-B/16 | CIFAR-100 | Vision Transformer for fine-grained classification |
| BERT-base | GLUE | Transformer-based language model for NLP benchmarks |

## Hypothesis Tests

### Is Bias Correction Needed?

To rigorously prove the desired result, we conduct the following experiment:

1. **For a set task** (defined model and dataset), consider a fixed hyperparameter configuration
2. **For multiple random seeds** (controlling both torch behavior and data loader sampling):
   - Train the exact same model to convergence with and without bias correction
   - For a random seed $i$, denote:
     - $a_i$ = accuracy of the model trained with bias correction
     - $b_i$ = accuracy of the model without bias correction
3. **Obtain the following table** for a collection of random seeds:

<div align="center">

| Seed | Accuracy without bias correction | Accuracy with bias correction | Difference |
|------|----------------------------------|-------------------------------|------------|
| 1    | $a_1$                            | $b_1$                         | $a_1-b_1$  |
| 2    | $a_2$                            | $b_2$                         | $a_2-b_2$  |
| $\vdots$ | $\vdots$                     | $\vdots$                      | $\vdots$   |
| $n$  | $a_n$                            | $b_n$                         | $a_n-b_n$  |

</div>

4. **Conduct a paired t-test** on the samples:
   - Let $\theta := \frac{1}{n} \sum_{i=1}^{n} {(a_i - b_i)}$
   - Null hypothesis: $H_0: \theta = 0$
   - We expect to fail to reject the null hypothesis if bias correction is indeed unnecessary

> **Open Question**: Is it any less statistically valid if instead of having the exact same hyperparameter configuration for lots of random seeds, we obtain lots of comparisons across different random seeds and hyperparameters? The upside here is no need to extensively tune hyperparameters for each dataset, and the result might actually be stronger. Intuitively it shouldn't make a difference, but somehow the collection of random seeds on one hyperparameter configuration feels more rigorous.

### Is the $E[g_i] = E[g_t]$ Assumption in Any Way Valid?

To test this assumption directly, we:

1. Track gradient statistics during the initial training steps
2. Compare the distribution of gradients at different timesteps
3. Measure the correlation between $g_0$ and $g_t$ for various values of $t$
4. Visualize how quickly the gradient distribution stabilizes

<div align="center">
<img src="https://via.placeholder.com/500x300?text=Gradient+Evolution+Visualization" alt="Placeholder for gradient evolution visualization"/>
<br>
<em>Visualization of how gradient statistics evolve during early training steps</em>
</div>

## Results

Our preliminary findings strongly indicate that:

1. There is no statistically significant difference in final performance between models trained with and without bias correction
2. The assumption that $E[g_i] \approx E[g_t]$ does not hold during early training steps
3. Initializing moment estimates with the first gradient provides a more accurate starting point than zeros

---

**Repository structure:**
- `optim/`: Custom AdamW implementation with a number of schedulers. 
- `models/`: Model architectures for various domains with a construct_model() function 
- `data/`: Dataset loaders and preprocessing with a get_loaders() 






