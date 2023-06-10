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

## Why is this repo a fork?
While this repository is a fork, we only used the dataset from the original repository.
The dataset was actually created by https://www.robots.ox.ac.uk/~vgg/data/pets/,
and they allow its use under the Attribution-ShareAlike 4.0 International license (https://creativecommons.org/licenses/by-sa/4.0/),
so we are not sure why the author of the repo we forked the dataset from violated the terms of the Creative Commons License, 
but added his own license instead without any attribution to the authors of the dataset (O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar). 
Nevertheless, we are retaining the license he had added, although we did not use any of his work.


## Sections
Will made more info here
