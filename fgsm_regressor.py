################################################################################
#                                                                              #
# One approach we tried is to let the generator learn from the noises created  #
# by FGSM. FGSM creates powerful and almost imperceptible adversarial noises.  #
# Using a stepping method, we created noise for each image in the Oxford-IIIT  #
# pets dataset, and trained the generator as a regressor of these pregenerated #
# noises.                                                                      #
#                                                                              #
# The amount of change that the regressor noises make to the original image    #
# appears to be correlated with the inaccuracy it generates for the spoofed    #
# model. The generated images suggests that the regressor training framework's #
# not practically useful to train the adversarial noise neural networks. The   #
# noise that the resulting regressor generates create too much perturbation to #
# the original image, so the change is clearly visible to human eyes.          #
#                                                                              #
# Authors: Cheng Chang <chc012@stanford.edu>                                   #
#                                                                              #
# Usage: python fgsm_regressor.py                                              #
#                                                                              #
#                                                                              #
# Copyright 2023 Cheng Chang <chc012@stanford.edu>                             #
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
################################################################################

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
from torch.optim import Adam
from torch.optim import SGD
from tqdm import tqdm
import random

from common import inverse_transform, load_model
from targeted_attack_adversarial_generator import AdversarialModel

class ResNetRegressor(nn.Module):
    def __init__(self, out_dim=224):
        super(ResNetRegressor, self).__init__()
        self.out_dim = out_dim
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet_layers = list(resnet.children())[:-1]
        self.resnet_embedding = torch.nn.Sequential(*resnet_layers)
        self.resnet_embedding.requires_grad = False
        
        self.head = nn.Linear(512, 3 * out_dim * out_dim)

    def forward(self, img):
        out = self.resnet_embedding(img)
        out = out.reshape((-1, 512))
        out = self.head(out)
        out = out.reshape((-1, 3, self.out_dim, self.out_dim))
        return out

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), padding=2),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), padding=1),
            nn.Conv2d(32, 3, (3, 3), padding=1),
            torch.nn.Tanh()
        )

    def forward(self, img):
        out = self.model(img)
        return out

def load_images(model, data_dir):
    torch.manual_seed(231) # needed for noise from fgsm to be sequentially deterministic

    # Define the image transformations
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    dataset = ImageFolder(data_dir, transform=transform)
    
    class_to_index_dict = dataset.class_to_idx
    index_to_class_list = [0 for _ in range(len(class_to_index_dict))]
    for cls in class_to_index_dict:
        index_to_class_list[class_to_index_dict[cls]] = cls

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, class_to_index_dict, index_to_class_list

def fgs_noises(model, train_dataloader, val_dataloader, test_dataloader,
               num_classes=37, max_iter=10, lr=1e-3, debug=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    noises = {
        "train": [],
        "val": [],
        "test": []
    }
    count = 0
    former_correct = 0
    after_correct = 0

    for dataloader, name in zip([train_dataloader, val_dataloader, test_dataloader],
                                ["train", "val", "test"]):
        for batch_num, (images, labels) in tqdm(enumerate(dataloader)):
            one_hot_labels = torch.zeros((len(labels), num_classes)).to(device)
            one_hot_labels[range(len(labels)), labels] = 1

            images = images.to(device)
            labels = labels.to(device)

            images_init = images.clone().detach()
            images.requires_grad_()

            for _ in range(max_iter):
                # compute relevant derivatives
                y_pred = model(images)
                loss = torch.nn.functional.cross_entropy(y_pred, labels)
                model.zero_grad()
                loss.backward()
                dx_sign = images.grad.sign()
                images.data = images.data + lr * dx_sign.data

            current_noises = images - images_init
            former_labels = model(images_init).argmax(dim=1)
            current_labels = model(images).argmax(dim=1)

            count += len(labels)
            former_correct += torch.sum(former_labels == labels)
            after_correct += torch.sum(current_labels == labels)

            if debug:
                print(f'Correct Label: {labels}')
                print(f'Former Label: {former_labels}')
                print(f'Current Label: {current_labels}')
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
                return

            noises[name].append(current_noises)

    former_acc = former_correct / count
    after_acc = after_correct /count

    print("Accuracy of model over original images: {:.4f}".format(former_acc))
    print("Accuracy of model over images after FGS: {:.4f}".format(after_acc))

    torch.save(noises, "datasets/fgs_noises.pkl")
    return noises, former_acc, after_acc, count

def regressor_train(model, regressor, train_dataloader, val_dataloader, noises_dict,
                    num_epochs=100, lr=1e-4, reg=0, verbose=True):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(regressor.parameters(), lr=lr, betas=(0.9, 0.999))
    #optimizer = SGD(regressor.parameters(), lr=lr)
    threshold = 0.01


    model = model.to(device)
    regressor = regressor.to(device)
    train_noises, val_noises = noises_dict["train"], noises_dict["val"]
    target = torch.tensor([1]).to(device)

    val_loss_history = []
    val_inacc_history = []
    best_regressor_state = None

    for epoch in tqdm(range(num_epochs)):
        for (batch_num, (images, labels)), noises in zip(
                enumerate(train_dataloader), train_noises):
            images = images.to(device)
            noises = noises.to(device)
            #noises.requires_grad = False

            optimizer.zero_grad()
            noises_pred = regressor(images)
            loss = F.cosine_embedding_loss(
                noises_pred.reshape((-1, 3*224*224)),
                noises.reshape((-1, 3*224*224)),
                target)
            clipped = (noises_pred > threshold).to(float)
            loss += reg * torch.sum(clipped) / len(labels)
            loss.backward()
            optimizer.step()
        
        train_loss, train_inacc = regressor_test(
            model, regressor, train_noises, train_dataloader,
            reg=reg, verbose=False)
        val_loss, val_inacc = regressor_test(
            model, regressor, val_noises, val_dataloader,
            epoch=epoch, reg=reg, verbose=False, display_image=True)
        if (not val_loss_history) or val_inacc > val_inacc_history[-1]:
            best_regressor_state = regressor.state_dict()

        if verbose:
            print("[Epoch {}/{}]:".format(epoch, num_epochs))
            print("[train loss: {:.4f}] [train inaccuracy: {:.4f}]".format(
                train_loss, train_inacc))
            print("[val loss: {:.4f}] [val inaccuracy: {:.4f}]".format(
                val_loss, val_inacc))

    torch.save(best_regressor_state, "models/regressor_res.pth")
    torch.save(val_loss_history, "results/reg_val_loss_hist.pkl")
    torch.save(val_inacc_history, "results/reg_val_inacc_hist.pkl")

    return best_regressor_state, val_loss_history, val_inacc_history

def regressor_test(model, regressor, noises, dataloader,
                   epoch=-1, reg=0, verbose=True, display_image=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    regressor = regressor.to(device)
    target = torch.tensor([1]).to(device)
    threshold = 0.01

    loss = 0
    incorrect = 0
    count = 0

    if display_image:
        #sample_idx = random.randint(0, len(dataloader)-1)
        sample_idx = 0

    for (batch_num, (images, labels)), current_noises in zip(
            enumerate(dataloader), noises):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            current_noises = current_noises.to(device)

            noises_pred = regressor(images)
            loss += F.cosine_embedding_loss(
                noises_pred.reshape((-1, 3*224*224)),
                current_noises.reshape((-1, 3*224*224)),
                target)
            clipped = (noises_pred > threshold).to(float)
            loss += reg * torch.sum(clipped) / len(labels)

            new_images = images + noises_pred
            y_pred = model(new_images).argmax(dim=1)
            incorrect += torch.sum(y_pred != labels)
            count += len(labels)

            if display_image and batch_num == sample_idx:
                sample_correct_label = labels[0]
                sample_former_label = model(images).argmax(dim=1)[0]
                sample_current_label = y_pred[0]
                sample_image_init = images[0]
                sample_image = new_images[0]
                sample_noise = noises_pred[0]

        del images
        del labels
        del current_noises
        del noises_pred
        del new_images
    
    inaccuracy = incorrect / count
    if verbose:
        print("Test loss: {:.4f}".format(loss))
        print("Test inaccuracy: {:.4f}".format(inaccuracy))

    if display_image:
        # print sampled images
        print(f'Correct Label: {sample_correct_label}')
        print(f'Former Label: {sample_former_label}')
        print(f'Current Label: {sample_current_label}')
        # Display initial image
        plt.imshow(inverse_transform(sample_image_init.squeeze()))
        plt.axis('off')
        plt.savefig("results/new/former.png".format(epoch))
        # Display final image
        plt.imshow(inverse_transform(sample_image.squeeze()))
        plt.axis('off')
        plt.savefig("results/new/current{}.png".format(epoch))
        # Display noise generated
        plt.imshow(inverse_transform(sample_noise.squeeze()))
        plt.axis('off')
        plt.savefig("results/new/noise{}.png".format(epoch))

    return loss, inaccuracy

def main():
    model = load_model()
    #regressor = ResNetRegressor(out_dim=224)
    regressor = AdversarialModel()
    #regressor = Regressor()

    train_dataloader, val_dataloader, test_dataloader, class_to_index, index_to_class = load_images(
        model, "datasets/images")
    desired_label = 0

    try:
        noises_path = "datasets/fgs_noises.pkl"
        noises_dict = torch.load(noises_path)
        print("Loaded noises from {}".format(noises_path))
    except:
        # will take 30 min to 1 hour
        noises_dict = fgs_noises(
            model, train_dataloader, val_dataloader, test_dataloader,
            num_classes=37, max_iter=50, lr=1e-3, debug=False)

    # Take about 10-15 min
    best_regressor_state, val_loss, val_inacc = regressor_train(
        model, regressor, train_dataloader, val_dataloader, noises_dict,
        num_epochs=50, lr=1e-3, reg=5e-3, verbose=True)

    regressor.load_state_dict(best_regressor_state)
    test_noises = noises_dict["test"]
    loss, val = regressor_test(
        model, regressor, test_noises, test_dataloader,
        reg=1e-3, verbose=True, display_image=False)

if __name__ == "__main__":
    main()
