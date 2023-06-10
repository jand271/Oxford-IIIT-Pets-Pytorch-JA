# Exploration of attacks on CNN models

Many Convolutional Neural Networks (CNNs) architectures, such as ResNet, provide limited robustness guarantees,
and may be vulnerable to attacks with adversarial samples that significantly degrade their performance.
We investigate methods that attack CNNs used for image classification by minimally changing input images and creating adversarial samples.
Using a ResNet-50 model pretrained on the Oxford-IIIT pets dataset, we compare previously proposed methods of creating adversarial samples,
including the Adversarial Patch method (AG), and the Fast Gradient Sign Method (FGSM). These experiments inform
us to design 3 different neural network structures and training mechanisms to generate adversarial examples.
With the classification inaccuracy ranging from 20% to 99% using our adversarial generators,
our experiment reveal a trade-off between magnitude of change applied to samples and attack effectiveness when generating adversarial samples.
We also create adversarial examples whose deviations from the original images that are either quasi-imperceptible or imperceptible to human eyes.
