import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from common import inverse_transform, load_model, load_images

class Regressor(nn.Module):
    def __init__(self, out_dim=224):
        super(Regressor, self).__init__()
        self.out_dim = out_dim
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet_layers = list(resnet.children())[:-1]
        self.resnet_embedding = torch.nn.Sequential(*resnet_layers)
        self.head = nn.Linear(512 * 32, 3 * out_dim * out_dim)

    def forward(self, img):
        out = self.resnet_embedding(img)
        out = out.flatten()
        out = self.head(out)
        out = out.reshape((3, self.out_dim, self.out_dim))
        return out

def fgs_noises(model, images, labels,
               batch_size=32, num_classes=37, max_iter=10, lr=1e-3, debug=False):
    images_init = images.clone().detach()
    one_hot_labels = torch.zeros((batch_size, num_classes))
    one_hot_labels[range(batch_size), labels] = 1
    images.requires_grad_()

    for _ in range(max_iter):
        # compute relevant derivatives
        y_pred = model(images)
        loss = torch.nn.functional.cross_entropy(y_pred, one_hot_labels)
        model.zero_grad()
        loss.backward()
        dx_sign = images.grad.sign()
        images.data = images.data + lr * dx_sign.data

    noises = images - images_init

    if debug:
        print(f'Correct Label: {labels[1]}')
        print(f'Former Label: {model(images_init)[1].argmax()}')
        print(f'Current Label: {model(images)[1].argmax()}')

        # Display initial image
        plt.imshow(inverse_transform(images_init[1].squeeze()))
        plt.axis('off')
        plt.show()

        # Display final image
        plt.imshow(inverse_transform(images[1].squeeze()))
        plt.axis('off')
        plt.show()

        # Display noise generated
        plt.imshow(inverse_transform(noises[1].squeeze()))
        plt.axis('off')
        plt.show()

    return noises

def regressor_train(model, regressor, dataloader,
                    num_epochs=100, batch_size=32, lr=1e-3, max_iter=10,
                    verbose=True, debug=False):
    optimizer = AdamW(regressor.parameters(), lr=0.0001, betas=(0.5, 0.999))

    noises = fgs_noises(model, images, labels,
                        batch_size=batch_size, num_classes=37,
                        max_iter=max_iter, lr=lr, debug=debug)

    for epoch in range(num_epochs):
        for batch_num, (images, labels) in enumerate(dataloader):

            
            optimizer.zero_grad()
            noises_pred = regressor(images)
            loss = F.mse_loss(noises, noises_pred)
            loss.backward()
            optimizer.step()
        
        if verbose and epoch % 10 == 0:
            print("[Epoch %d/%d] [Batch %d/%d]"
                % (epoch, num_epochs, batch_num, len(dataloader)))

def regressor_test(model, regressor, dataloader):
    pass
    

def main():
    model = load_model()
    regressor = Regressor(out_dim=224)
    train_dataloader, test_dataloader = load_images(model, "datasets/images")
    regressor_train(model, regressor, train_dataloader, lr=1e-3, max_iter=10, debug=True)
    regressor_test(model, regressor, test_dataloader)

if __name__ == "__main__":
    main()
