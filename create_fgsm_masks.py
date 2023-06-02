import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from common import inverse_transform, load_model

def load_images(model, data_dir):
    # Define the image transformations

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # Load the image dataset
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader

def create_supervised_generator_dataset(model, dataloader, lr=1e-3, max_iter=100, debug=False):
    for _, (images, labels) in enumerate(dataloader):
        for image, label in zip(images, labels):
            image_init = image

            for _ in range(max_iter):
                # compute relevant derivatives
                y_pred = model(image)
                loss = torch.nn.functional.cross_entropy(y_pred, label)
                model.zero_grad()
                loss.backward()
                dx_sign = image.grad.sign()

                image.data = image.data + lr * dx_sign.data

                current_label = y_pred.argmax()

                if current_label != label:
                    break
            
            noise = image - image_init
            
            if debug:
                print(f'learning rate: {lr:0.3e}')
                print(f'New Label: {labels[current_label]}')

                # Display initial image
                plt.imshow(inverse_transform(image_init.squeeze()))
                plt.axis('off')
                plt.show()

                # Display final image
                final_image = inverse_transform(image.squeeze())
                plt.imshow(final_image)
                plt.axis('off')
                plt.show()

                # Display noise generated
                final_image = inverse_transform(noise.squeeze())
                plt.imshow(noise)
                plt.axis('off')
                plt.show()

                return

def main():
    model = load_model()
    train_dataloader, test_dataloader = load_images(model, "datasets/images")
    train_noises = create_supervised_generator_dataset(
        model, train_dataloader, lr=1e-3, max_iter=100, debug=False)
    
    test_noises = create_supervised_generator_dataset(
        model, test_dataloader, lr=1e-3, max_iter=100, debug=False)

if __name__ == "__main__":
    main()
