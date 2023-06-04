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
from tqdm import tqdm

from common import inverse_transform, load_model
from try_nn_fgsm import AdversarialModel

class Regressor(nn.Module):
    def __init__(self, out_dim=224):
        super(Regressor, self).__init__()
        self.out_dim = out_dim
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet_layers = list(resnet.children())[:-1]
        self.resnet_embedding = torch.nn.Sequential(*resnet_layers)
        self.head = nn.Linear(512, 3 * out_dim * out_dim)

    def forward(self, img):
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

    torch.save(noises, "results/fgs_noises.pkl")
    return noises, former_acc, after_acc, count

def regressor_train(model, regressor, train_dataloader, val_dataloader, noises_dict,
                    num_epochs=100, lr=1e-4, verbose=True):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(regressor.parameters(), lr=lr, betas=(0.9, 0.999))
    model = model.to(device)
    regressor = regressor.to(device)
    train_noises, val_noises = noises_dict["train"], noises_dict["val"]

    val_loss_history = []
    val_inacc_history = []
    best_regressor_state = None

    for epoch in tqdm(range(num_epochs)):
        for (batch_num, (images, labels)), noises in zip(
                enumerate(train_dataloader), train_noises):
            images = images.to(device)
            noises = noises.to(device)

            optimizer.zero_grad()
            noises_pred = regressor(images)
            loss = F.mse_loss(noises_pred, noises, reduction="sum")
            loss.backward()
            optimizer.step()
        
        train_loss, train_inacc = regressor_test(
            model, regressor, train_noises, train_dataloader, verbose=False)
        val_loss, val_inacc = regressor_test(
            model, regressor, val_noises, val_dataloader, verbose=False)
        if (not val_loss_history) or val_inacc > val_inacc_history[-1]:
            best_regressor_state = regressor.state_dict()

        if verbose and epoch % 1 == 0: # TODO: back to 10
            print("[Epoch {}/{}]:".format(epoch, num_epochs))
            print("[train loss: {:.4f}] [train inaccuracy: {:.4f}]".format(
                train_loss, train_inacc))
            print("[val loss: {:.4f}] [val inaccuracy: {:.4f}]".format(
                val_loss, val_inacc))

    torch.save(best_regressor_state, "models/regressor_res.pth")
    torch.save(val_loss_history, "results/reg_val_loss_hist.pkl")
    torch.save(val_inacc_history, "results/reg_val_inacc_hist.pkl")

    return best_regressor_state, val_loss_history, val_inacc_history

def regressor_test(model, regressor, noises, dataloader, verbose=True):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    regressor = regressor.to(device)

    loss = 0
    incorrect = 0
    count = 0
    for (batch_num, (images, labels)), current_noises in zip(
            enumerate(dataloader), noises):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            current_noises = current_noises.to(device)

            noises_pred = regressor(images)
            loss += F.mse_loss(noises_pred, current_noises, reduction="sum")

            new_images = images + noises_pred
            y_pred = model(new_images).argmax(dim=1)
            incorrect += torch.sum(y_pred != labels)
            count += len(labels)

        del images
        del labels
        del current_noises
        del noises_pred
        del new_images
    
    inaccuracy = incorrect / count
    if verbose:
        print("Test loss: {:.4f}".format(loss))
        print("Test inaccuracy: {:.4f}".format(inaccuracy))

    return loss, inaccuracy

def main():
    model = load_model()
    #regressor = Regressor(out_dim=224)
    regressor = AdversarialModel()

    train_dataloader, val_dataloader, test_dataloader, class_to_index, index_to_class = load_images(
        model, "datasets/images")
    desired_label = 0

    try:
        noises_path = "results/fgs_noises.pkl"
        noises_dict = torch.load(noises_path)
        print("Loaded noises from {}".format(noises_path))
    except:
        # will take 30 min to 1 hour
        noises_dict = fgs_noises(
            model, train_dataloader, val_dataloader, test_dataloader,
            num_classes=37, max_iter=50, lr=1e-3, debug=False)

    best_regressor_state, val_loss, val_inacc = regressor_train(
        model, regressor, train_dataloader, val_dataloader, noises_dict,
        num_epochs=10, lr=5e-3, verbose=True)

    regressor.load_state_dict(best_regressor_state)
    test_noises = noises_dict["test"]
    loss, val = regressor_test(model, regressor, test_dataloader, test_noises, verbose=True)

if __name__ == "__main__":
    main()
