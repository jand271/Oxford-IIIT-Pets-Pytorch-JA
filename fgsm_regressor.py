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
from tqdm import tqdm

from common import inverse_transform, load_model

class Regressor(nn.Module):
    def __init__(self, out_dim=224):
        super(Regressor, self).__init__()
        self.out_dim = out_dim
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet_layers = list(resnet.children())[:-1]
        self.resnet_embedding = torch.nn.Sequential(*resnet_layers)
        self.head = nn.Linear(512, 3 * out_dim * out_dim)

    def forward(self, img):
        print(img.shape)
        out = self.resnet_embedding(img)
        out = out.reshape((-1, 512))
        out = self.head(out)
        out = out.reshape((-1, 3, self.out_dim, self.out_dim))
        return out

def load_images(model, data_dir):
    torch.manual_seed(231) # needed for noise from fgsm to be sequentially deterministic

    # Define the image transformations
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    dataset = ImageFolder(data_dir, transform=transform)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def fgs_noises(model, train_dataloader, val_dataloader, test_dataloader,
               batch_size=32, num_classes=37, max_iter=10, lr=1e-3, debug=False):
    noises = {
        "train": [],
        "val": [],
        "test": []
    }

    for dataloader, name in zip([train_dataloader, val_dataloader, test_dataloader],
                                ["train", "val", "test"]):
        for batch_num, (images, labels) in enumerate(dataloader):
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

            current_noises = images - images_init

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
                plt.imshow(inverse_transform(current_noises[1].squeeze()))
                plt.axis('off')
                plt.show()

            noises[name].append(current_noises)
    
    return noises

def regressor_train(model, regressor, train_dataloader, val_dataloader, noises_dict,
                    num_epochs=100, batch_size=32, lr=1e-3, max_iter=10,
                    verbose=True, debug=False):
    optimizer = AdamW(regressor.parameters(), lr=0.0001, betas=(0.5, 0.999))

    train_noises, val_noises = noises_dict["train"], noises_dict["val"]

    val_loss_history = []
    val_inacc_history = []
    best_regressor_state = None

    for epoch in tqdm(range(num_epochs)):
        for batch_num, (images, labels), noises in zip(
                enumerate(train_dataloader), train_noises):
            optimizer.zero_grad()
            noises_pred = regressor(images)
            print(noises_pred.shape)
            loss = F.mse_loss(noises_pred, noises)
            loss.backward()
            optimizer.step()
        
        val_loss, val_inacc = regressor_test(
            model, regressor, val_noises, val_dataloader, verbose=False)
        if (not val_loss_history) or val_inacc > val_inacc_history[-1]:
            best_regressor_state = regressor.state_dict()

        if verbose and epoch % 1 == 0: # TODO: back to 10
            print("[Epoch %d/%d] [Batch %d/%d]:"
                % (epoch, num_epochs, batch_num, len(train_dataloader)))
            print("Val loss: {:.4}".format(val_loss))
            print("Val inaccuracy: {:.4}".format(val_inacc))

    torch.save(best_regressor_state, "models/regressor_res.pth")
    torch.save(val_loss_history, "results/reg_val_loss_hist.pkl")
    torch.save(val_inacc_history, "results/reg_val_inacc_hist.pkl")

    return best_regressor_state, val_loss_history, val_inacc_history

def regressor_test(model, regressor, noises, dataloader, verbose=True):
    loss = 0

    for batch_num, (images, labels), current_noises in zip(
            enumerate(dataloader), noises):
        noises_pred = regressor(images)
        loss += F.mse_loss(noises_pred, current_noises)

        new_images = images + noises_pred
        y_pred = model(new_images)
        inaccuracy = torch.sum(y_pred != labels) / len(labels)
    
    if verbose:
        print("Test loss: {:.4}".format(loss))
        print("Test inaccuracy: {:.4}".format(inaccuracy))

    return loss, inaccuracy

def main():
    model = load_model()
    regressor = Regressor(out_dim=224)
    train_dataloader, val_dataloader, test_dataloader = load_images(model, "datasets/images")
    noises_dict = fgs_noises(model, train_dataloader, val_dataloader, test_dataloader,
                             batch_size=32, num_classes=37, max_iter=10, lr=1e-3, debug=True)

    best_regressor_state, val_loss, val_inacc = regressor_train(
        model, regressor, train_dataloader, val_dataloader, noises_dict,
        num_epochs=2, batch_size=32, lr=1e-3, max_iter=10, verbose=True, debug=True)

    regressor.load_state_dict(best_regressor_state)
    loss, val = regressor_test(model, regressor, test_dataloader, noises_dict, verbose=True)

if __name__ == "__main__":
    main()
