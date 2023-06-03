import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
from matplotlib import pyplot as plt
from common import load_model, load_images, inverse_transform, attempt_gpu_acceleration, Unflatten


class AdversarialModel(torch.nn.Module):

    def __init__(self, ):
        super(AdversarialModel, self).__init__()

        # While we could have left the task of figuring out the best configuration
        # for the number of hyperparameters in Hidden Layers, for now,
        # we are using a variant of a model that Jason created

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


def validate_test_set(adversarial_model, timm_model, test_dataloader, desired_label, device):
    correct = 0
    for (images, _) in test_dataloader:
        images = images.to(device)
        with torch.no_grad():
            adversarial_noise = adversarial_model(images)
            label_predicted = timm_model(images + adversarial_noise)
            correct += torch.sum(torch.argmax(label_predicted, axis = 1) == desired_label).item()
    return correct / len(test_dataloader.dataset)


def train(trial):
    device = attempt_gpu_acceleration()
    # Change it to true manually. This function can't accpet any inputs, so hard-coding for now
    display_images = True
    timm_model = load_model()
    timm_model = timm_model.to(device)
    train_dataloader, test_dataloader = load_images(timm_model, "datasets/images")
    adversarial_model = AdversarialModel().to(device)

    # Initial validation accuracy
    validation_accuracy = 0
    desired_label = 0

    lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 2e-2, log=True)
    gan_weight_decay = trial.suggest_float("gan_weight_decay", 1e-3, 2e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 10, 50)


    optimizer = torch.optim.Adam(adversarial_model.parameters(),
                                 lr = lr,
                                 betas = (0.1, 0.999),
                                 weight_decay = weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch, (images, labels) in enumerate(train_dataloader):

            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)  # default batch-size
            labels_one_hot = torch.zeros((batch_size, 37))  # 37 categories
            labels_one_hot[:, desired_label] = 1
            labels_one_hot = labels_one_hot.to(device)

            optimizer.zero_grad()
            adversarial_noise = adversarial_model(images)
            label_predicted = timm_model(adversarial_noise + images)
            loss = loss_function(label_predicted, labels_one_hot) + \
                gan_weight_decay * torch.norm(adversarial_noise).mean()

            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [Spoofed Label: %d/%d] [Desired Label: %d/%d]"
                    % (epoch,
                       num_epochs,
                       batch,
                       len(train_dataloader),
                       loss, torch.sum(torch.argmax(label_predicted, axis=1) != labels).item(),
                       batch_size, torch.sum(torch.argmax(label_predicted, axis=1) == desired_label).item(),
                       batch_size)
                )

        if display_images:
            print(f'original label{labels[0]} : new label{torch.argmax(label_predicted[0])}')
            plt.imshow(inverse_transform(images[0] + adversarial_noise[0]))
            plt.show()
            plt.imshow(inverse_transform(adversarial_noise[0]))
            plt.show()

        validation_accuracy = validate_test_set(adversarial_model, timm_model, test_dataloader, desired_label, device)
        print("[Epoch %d/%d] [Val: %f]" % (epoch, num_epochs, validation_accuracy))

    torch.save(adversarial_model.state_dict(), "adversarial_res.pth")
    return validation_accuracy


def training(num_epochs, load_save=False, display_images=False):
    # Currently, num_epochs is a hyper-parameter, so we are ignoring it for now.
    if load_save:
        assert False
    # Use Meta's Optuna for hyperparameter search.
    # Adopted from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    study = optuna.create_study(direction="maximize")
    # We wish to maximize validation accuracy
    study.optimize(train, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
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
    training(10, display_images=False)
