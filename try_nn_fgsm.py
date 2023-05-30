
from datetime import datetime

from matplotlib import pyplot as plt

import timm
import torch
from common import load_model, load_images, inverse_transform, attempt_gpu_acceleration


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
            torch.nn.Tanh(),
        )

    def forward(self, image):
        adversarial_noise = self.model(image)
        return adversarial_noise.view(-1, 3, 224, 224)


def train(adversarial_model, timm_model, dataloader, num_epochs, display_images=False):

    desired_label = 0

    device = attempt_gpu_acceleration()

    adversarial_model.to(device)
    timm_model.to(device)

    optimizer = torch.optim.Adam(adversarial_model.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=0.02)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch, (images, labels) in enumerate(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)
            labels_one_hot = torch.zeros((batch_size, 37))
            labels_one_hot[:, desired_label] = 1
            labels_one_hot = labels_one_hot.to(device)

            optimizer.zero_grad()
            adversarial_noise = adversarial_model(images)
            label_predicted = timm_model(adversarial_noise + images)
            loss = loss_function(label_predicted, labels_one_hot) + 2e-2 * torch.norm(adversarial_noise).mean()

            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [Spoofed Label: %d/%d] [Desired Label: %d/%d]"
                    % (epoch, num_epochs, batch, len(dataloader), loss, torch.sum(torch.argmax(label_predicted, axis=1) != labels).item(), batch_size, torch.sum(torch.argmax(label_predicted, axis=1) == desired_label).item(), batch_size)
                )
        if display_images:
            print(f'original label{labels[0]}')
            print(f'new label{torch.argmax(label_predicted[0])}')
            plt.imshow(inverse_transform(images[0] + adversarial_noise[0]))
            plt.show()


def training(num_epochs, load_save=False, display_images=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timm_model = load_model()
    timm_model = timm_model.to(device)

    dataloader = load_images(timm_model, "datasets/images")

    adversarial_model = AdversarialModel()

    if load_save:
        assert False

    train(adversarial_model, timm_model, dataloader, num_epochs, display_images=display_images)

    torch.save(adversarial_model.state_dict(), "adversarial_res.pth")


if __name__ == '__main__':
    training(100, display_images=True)
