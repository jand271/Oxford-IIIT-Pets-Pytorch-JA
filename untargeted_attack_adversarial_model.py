################################################################################
#                                                                              #
# Adversarial noise generator for producing quasi-imperceptible perturbations  #
# (modifications to an image) for an untargeted attack (the goal of the        #
# adversarial model is to help produce adversarial noise, which when added to  # 
# images, created images that humans have a hard time telling apart from the   #
# original images but they fool a ResNet50 model pretrained on the             #
# Oxford-IIIT pets dataset.                                                    #               
#                                                                              #
# This code runs hyperparameter tuning trials to find the best possible        #
# configuration wherein our objective function (validation set accuracy of the #
# adversarial model divided by the sum of the squares of the pixels of         #
# adversarial noise) is maximized while training an adversarial noise          #
# generator that attempts to fool a ResNet50 model pretrained on the           #
# Oxford-IIIT pets dataset.                                                    #
#                                                                              #
# Note that the validation set inaccuracy for the pre-trained ResNet50 is the  #
# validation set accuracy of the adversarial noise generator.                  #
#                                                                              #
# Authors: Sanchit Jain <sanchit@stanford.edu>,                                #
#          Jason Anderson <jand271@stanford.edu>                               #
#                                                                              #
# Other credits: Cheng Chang <chc012@stanford.edu> provided useful advice      #
#                                                                              #
# Usage: Run python untargeted_attack_adversarial_model.py --help              #
#                                                                              #
#                                                                              #
# Copyright 2023 Sanchit Jain, Jason Anderson, Cheng Chang                     #
#                                                                              #
# Permission is hereby granted, free of charge, to any person obtaining a copy #
# of this software and associated documentation files (the “Software”), to     #
# deal in the Software without restriction, including without limitation the   #
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or  #
# sell copies of the Software, and to permit persons to whom the Software is   #
# furnished to do so, subject to the following conditions:                     #
#                                                                              #
# The above copyright notice and this permission notice shall be included in   #
# all copies or substantial portions of the Software.                          #
#                                                                              #
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,              #
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF           #
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO #
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,        #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR        #
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE    #
# USE OR OTHER DEALINGS IN THE SOFTWARE.                                       #
#                                                                              #
#                                                                              #
################################################################################


import torch
import torch.nn as nn
import optuna
import logging
import argparse
from optuna.trial import TrialState
from matplotlib import pyplot as plt
from common import load_model, load_images, inverse_transform, attempt_gpu_acceleration, Unflatten

# globals
# display all validation set images after 5 epochs
display_validation_set_images = False
# display one batch at the end of each epoch
display_training_set_images = False
# number of epochs per "trial" (an experiment with a unique hyper-parameter config)
num_epochs_per_trial = 5
# display validation set images after every display_every_num_epochs epochs
display_every_num_epochs = 5
# run only one trial with a hyper-parameter config known to work well
known_config_mode = False
# number of trials
num_trials = 100


class AdversarialModel(torch.nn.Module):

    def __init__(self, ):
        super(AdversarialModel, self).__init__()

        # While we could have left the task of figuring out the best configuration
        # for the number of hyperparameters in Hidden Layers, for now,
        # we are using a variant of a model that Jason created.
        # But maybe we can try finding a CNN (with ConvTranspose2D) that would work well.

        self.model = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 336),
            nn.BatchNorm1d(336),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(336, 336),
            nn.BatchNorm1d(336),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(336, 336),
            nn.BatchNorm1d(336),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(336, 3 * 224 * 224),
            nn.Tanh(),
            Unflatten(-1, 3, 224, 224)
        )

    def forward(self, image):
        return self.model(image)


def validate_test_set(
      adversarial_model,
      timm_model,
      test_dataloader,
      device,
      epoch):
    correct = 0
    # number of images we would print
    num_images_printed = 0
    for (images, labels) in test_dataloader:
        images = images.to(device)
        labels = labels.to("cpu")

        with torch.no_grad():
            # get adversarial noise
            adversarial_noise = adversarial_model(images)
            # predict the label of the adversarial image, which is generated by
            # adding the original image to adversarial noise
            label_predicted = timm_model(images + adversarial_noise).to("cpu")

            correct += torch.sum(torch.argmax(label_predicted, axis = 1) != labels).item()

            if display_validation_set_images and (epoch % display_every_num_epochs == 0):
                for i in range(len(labels)):
                    if labels[i] != torch.argmax(label_predicted[i]):
                        print("epoch_" + str(epoch) + "_perturbed_image_" + str(num_images_printed))
                        print(f'original label{labels[i]} : new label{torch.argmax(label_predicted[i])}')
                        plt.imshow(inverse_transform(images[i]))
                        filename = "epoch_" + str(epoch) + "_original_image_" + str(num_images_printed)
                        plt.savefig(filename)
                        plt.imshow(inverse_transform(images[i] + adversarial_noise[i]))
                        filename = "epoch_" + str(epoch) + "_perturbed_image_" + str(num_images_printed)
                        plt.savefig(filename)
                        plt.imshow(inverse_transform(adversarial_noise[i]))
                        filename = "epoch_" + str(epoch) + "_adversarial_noise_" + str(num_images_printed)
                        plt.savefig(filename)
                        num_images_printed += 1
    return correct / len(test_dataloader.dataset)


# TODO: support saving the best model found after hyper-parameter search
def save_best_model(adversarial_model):
    # torch.save(adversarial_model.state_dict(), "adversarial_res.pth")
    pass


def train(trial):
    device = attempt_gpu_acceleration()

    # ResNet50 pre-trained on the Oxford-IIIT pets dataset
    timm_model = load_model()
    timm_model = timm_model.to(device)
    # Oxford-IIIT pets dataset
    train_dataloader, test_dataloader = load_images(timm_model, "datasets/images")
    adversarial_model = AdversarialModel().to(device)

    # Initial validation accuracy
    validation_accuracy = 0

    if known_config_mode:
        # We know this config to perform well (minimal adversarial perturbation with
        # reasonable misclassification accuracy). Gathered from 100 trials on an
        # Nvidia V100 GPU
        lr = 0.0008490532908377003
        weight_decay = 0.018128854482948858
        norm_weight_decay = 0.04947776147536436
    else:
        lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)
        norm_weight_decay = trial.suggest_float("norm_weight_decay", 1e-3, 5e-2, log=True)
    # we can change this later to make the number of epochs a hyper-parameter as well.
    num_epochs = trial.suggest_int("num_epochs", num_epochs_per_trial, num_epochs_per_trial)

    optimizer = torch.optim.Adam(adversarial_model.parameters(),
                                 lr = lr,
                                 betas = (0.1, 0.999),
                                 weight_decay = weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()
    print("lr: %f"%(lr))
    print("weight_decay: %f"%(weight_decay))
    print("norm_weight_decay: %f"%(norm_weight_decay))
    for epoch in range(num_epochs):
        for batch, (images, labels) in enumerate(train_dataloader):

            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)  # default batch-size
            labels_one_hot = torch.zeros((batch_size, 37))  # 37 categories
            labels_one_hot[torch.arange(0, batch_size), labels] = 1
            labels_one_hot = labels_one_hot.to(device)

            optimizer.zero_grad()
            adversarial_noise = adversarial_model(images)
            label_predicted = timm_model(adversarial_noise + images)
            loss = -loss_function(label_predicted, labels_one_hot) + \
                norm_weight_decay * torch.norm(adversarial_noise).mean()

            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [Spoofed Labels: %d/%d]"
                    % (epoch + 1,
                       num_epochs,
                       batch,
                       len(train_dataloader),
                       loss, torch.sum(torch.argmax(label_predicted, axis=1) != labels).item(),
                       batch_size)
                )

        if display_training_set_images:
            for i in range(len(labels)):
                if labels[i] != torch.argmax(label_predicted[i]):
                    print(f'original label{labels[i]} : new label{torch.argmax(label_predicted[i])}')
                    plt.imshow(inverse_transform(images[i]))
                    filename = "training_set_epoch_" + str(epoch) + "_original_image_" + str(i)
                    plt.savefig(filename)
                    plt.imshow(inverse_transform(images[i] + adversarial_noise[i]))
                    filename = "training_set_epoch_" + str(epoch) + "_perturbed_image_" + str(i)
                    plt.savefig(filename)
                    plt.imshow(inverse_transform(adversarial_noise[i]))
                    filename = "training_set_epoch_" + str(epoch) + "_adversarial_noise_" + str(i)
                    plt.savefig(filename)

        validation_accuracy = validate_test_set(
                adversarial_model,
                timm_model,
                test_dataloader,
                device,
                epoch + 1)
        print("[Epoch %d/%d] [Validation accuracy: %f]" % (epoch + 1, num_epochs, validation_accuracy))
    return validation_accuracy / torch.sum(adversarial_noise ** 2)


def training(num_trials):
    # Use Meta's Optuna for hyperparameter search.
    # Adopted from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    study = optuna.create_study(direction="maximize")
    # We wish to maximize validation accuracy
    study.optimize(train, n_trials=num_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logging.info("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--known_config_mode",
                        action="store_true",
                        default=False,
                        help="use one config known to work well. False by default")
    parser.add_argument("--num_trials",
                        default=100,
                        help="number of hyper-parameter search trials. 100 by default")
    parser.add_argument("--num_epochs",
                        default=5,
                        help="number of epochs in each trial")
    parser.add_argument("--display_every_num_epochs",
                        default=5,
                        help="display validation set images after every display_every_num_epochs epochs. 5 by default")
    parser.add_argument("--display_validation_set_images",
                        default=False,
                        action="store_true",
                        help="Whether or not to display validation set images. False by default")
    parser.add_argument("--display_training_set_images",
                        default=False,
                        action="store_true",
                        help="Whether or not to display a batch of training set images after each epoch. False by default")
    parser.add_argument("--verbose",
                        "-v",
                        default=True,
                        action="store_true")
    args, _ = parser.parse_known_args()
    display_validation_set_images = args.display_validation_set_images
    display_training_set_images = args.display_training_set_images
    num_epochs_per_trial = int(args.num_epochs)
    display_every_num_epochs = int(args.display_every_num_epochs)
    if args.known_config_mode:
        display_validation_set_images = True
        known_config_mode = True
        num_trials = 1
    training(num_trials)
