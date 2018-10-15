# AIUkraine 2018, day 2

October 14, 2018
	
# 100 Machine Learning Models running live: The Booking.com approach

Lucas Bernardi, Booking.com 

Tips:

- use decoupling predictions from training
- use GLM
- use lookup table

# How to explain predictions of your network?

Vladyslav Kolbasin, Globallogic 

Accuracy vs. interpretability tradeoff

- [LIME](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime) (local individual model-agnostic explanation)
- [SHapley Additive exPlanation](https://github.com/slundberg/shap) (SHAP)
- keras-vis
- Google what-if
- Lucid


# Domain adaptation problem

Alexander Obiednikov, Ring Ukraine 

Style transfer (perceptual loss) doesn't work for this task. Perceptual loss works only with VGG, but not with ResNet. Doesn't work on it's own, only improves the solution.

Types

* Unsupervised cross-domain image generation
* Cycle-consistency loss
* [Progressively growing GANs](https://github.com/tkarras/progressive_growing_of_gans) for improved quality stability and variation
	* https://arxiv.org/pdf/1710.10196.pdf
* Pix2Pix and Vid2Vid
* UNIT and [MUNIT](https://github.com/NVlabs/MUNIT): VAE instead of GANs

[Glow: Better Reversible Generative Models](https://blog.openai.com/glow/)

[Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)

[Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)

# Rare events prediction in the car

Elena Kasianenko, Nazar Sheremeta, CloudMade 

Adaptive cruise control



