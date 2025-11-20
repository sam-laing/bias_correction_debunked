# Simplifying Adam: Bias Correction Debunked

This repository contains the code and experiments for the paper *"Simplifying Adam: Bias Correction Debunked"*. The project examines the role of bias correction in the Adam optimizer and shows that it is not necessary for strong performance when modern training practices are used.

---

## Summary

Bias correction has been included in Adam since its introduction, and most deep learning frameworks implement it by default. However, its contribution has not been thoroughly evaluated in practical training settings. This work provides a systematic ablation of bias correction in both language and vision models.

Our main conclusion is that bias correction is not beneficial when an appropriate learning rate schedule is used. It does not act as a true statistical correction but instead modifies the effective learning rate during early optimization.

---

## Key Findings

- With warm-up cosine learning rate schedules, removing bias correction does **not** affect final performance.
- In the commonly used setting where β₁ = β₂ (e.g., 0.95), bias correction can slightly reduce performance.
- Under the default β₁ = 0.9, β₂ = 0.999 setting, bias correction appears helpful **only** when the learning rate is held constant.
- The “bias correction” term primarily functions as an implicit, and often suboptimal, learning rate warm-up.

---

## Mechanistic Insight

The Adam update with bias correction can be factored as:
$$
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} =
\left(\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}\right)
\cdot
\frac{m_t}{\sqrt{v_t}}
$$

The multiplier

$$
\rho(t;\beta_1,\beta_2) = \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
$$

changes the effective learning rate over time. For β₁ = β₂ = 0.95 it creates a sharp spike in the initial steps, which can destabilize training unless an explicit warm-up is applied.

---

## Experimental Setup

### Language Models
- 160M parameter transformer
- SlimPajama dataset (2.5B tokens)
- Weight decay and gradient clipping enabled
- Sweeps over learning rates and β₁, β₂ settings
- Both warm-up cosine scheduling and constant learning rates tested

### Vision Models
- CIFAR-10: ResNet-9
- Tiny ImageNet: ViT and ResNet-50
- Standard data augmentation

Across all settings, bias correction did not provide a measurable benefit when a standard learning rate schedule was used.

---

## Conclusion

Bias correction does not improve results when Adam is used with a proper training pipeline. It can therefore be removed without sacrificing performance, simplifying both implementation and theoretical analyses.

---

## Citation

@misc{adam_bias_correction_2025,
  title        = {Simplifying Adam: Bias Correction Debunked},
  author       = {Sam Laing},
  year         = {2025},
  note         = {Preprint},
}
