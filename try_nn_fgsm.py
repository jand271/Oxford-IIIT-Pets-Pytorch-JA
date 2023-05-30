
from datetime import datetime

from matplotlib import pyplot as plt

import timm
import torch
from common import load_model, load_images, inverse_transform


class AdversarialModel(torch.nn.Module):

    def __init__(self, ):
        super(AdversarialModel, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 224 * 224, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(100, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(100, 3 * 224 * 224),
            torch.nn.Tanh()
        )

    def forward(self, image):
        adversarial_noise = self.model(image)
        return adversarial_noise.view(-1, 3, 224, 224)


def train(adversarial_model, timm_model, dataloader, num_epochs):

    desired_label = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adversarial_model.to(device)
    timm_model.to(device)

    optimizer = torch.optim.Adam(adversarial_model.parameters(), lr=0.001, betas=(0.5, 0.999))
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch, (images, labels) in enumerate(dataloader):

            batch_size = images.size(0)
            labels_one_hot = torch.zeros((batch_size, 37))
            labels_one_hot[:, desired_label] = 1

            optimizer.zero_grad()

            adversarial_noise = adversarial_model(images)
            label_predicted = timm_model(adversarial_noise + images)
            loss = loss_function(label_predicted, labels_one_hot) + 1e2 * torch.max(torch.abs(adversarial_noise))

            loss.backward()
            optimizer.step()

            if batch % 1 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                    % (epoch, num_epochs, batch, len(dataloader), loss)
                )

            if batch % 10 == 0:
                print(f'original label{labels[0]}')
                print(f'new label{torch.argmax(label_predicted[0])}')
                plt.imshow(inverse_transform(images[0] + adversarial_noise[0]))
                plt.show()


def training(num_epochs, load_save=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timm_model = load_model()
    timm_model = timm_model.to(device)

    dataloader = load_images(timm_model, "datasets/images")

    adversarial_model = AdversarialModel()

    if load_save:
        assert False

    train(adversarial_model, timm_model, dataloader, num_epochs)

    torch.save(adversarial_model.state_dict(), "adversarial_res.pth")


if __name__ == '__main__':
    training(100)
