# Debunking the Bias Correction necessity Myth

Implementations of Adam(W) all apply bias-correction to the EMA statistics. The claim is that doing so prevents a bias in the early updates of the optimizer. \

We demonstrate that across both vision and language models, simply initialising the moments to their values rather than to zero