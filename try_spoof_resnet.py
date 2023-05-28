
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable

from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, CenterCrop, ToPILImage
from torchvision.transforms import Normalize

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, num_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, img_shape, num_classes):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(num_classes + int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


def adversarial_loss_ls(scores, logits):
    return 0.5 * ((scores - logits) ** 2).mean()


# Define the training process
def train(generator, discriminator, dataloader, num_epochs, latent_dim, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adversarial_loss = nn.BCELoss()
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            batch_size = real_images.size(0)

            real_images = real_images.to(device)
            labels = labels.to(device)

            valid = Variable(torch.ones(batch_size, 1)).to(device)
            fake = Variable(torch.zeros(batch_size, 1)).to(device)

            optimizer_G.zero_grad()

            z = Variable(torch.randn(batch_size, latent_dim)).to(device)
            gen_labels = Variable(torch.randint(0, num_classes, (batch_size,))).to(device)

            gen_images = generator(z, gen_labels)

            g_loss = adversarial_loss_ls(discriminator(gen_images, gen_labels), valid)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            real_loss = adversarial_loss_ls(discriminator(real_images, labels), valid)
            fake_loss = adversarial_loss_ls(discriminator(gen_images.detach(), gen_labels), fake)

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

def inverse_transform(image):
    # inverse transform generated by request from ChatGPT by supplying the print(transform) from below
    inv_normalize = Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    image = inv_normalize(image)
    image = ToPILImage()(image)
    image = CenterCrop(size=(235, 235))(image)
    image = Resize(size=(224, 224), interpolation=2)(image)
    return image


def load_model():
    # Load the pre-trained PyTorch model
    model = timm.create_model(
        'hf-hub:nateraw/resnet50-oxford-iiit-pet',
        pretrained=True
    )
    model.eval()
    return model


def load_images(model, data_dir):
    # Define the image transformations

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # Load the image dataset
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader

def training():

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model()
    model = model.to(device)

    # Load the images
    data_dir = "datasets/images"
    dataloader = load_images(model, data_dir)

    # Set hyperparameters
    latent_dim = 100  # Dimension of the noise vector
    num_classes = 37  # Number of classes (labels)
    img_shape = (3, 224, 224)  # Shape of the input images

    # Initialize generator and discriminator
    generator = Generator(latent_dim, img_shape, num_classes)
    discriminator = Discriminator(img_shape, num_classes)

    # Train the GAN
    num_epochs = 50
    train(generator, discriminator, dataloader, num_epochs, latent_dim, num_classes)

    torch.save(generator.state_dict(), "generator_res.pth")
    torch.save(discriminator.state_dict(), "discriminator_res.pth")


def create_image():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = 100  # Dimension of the noise vector
    num_classes = 37  # Number of classes (labels)
    img_shape = (3, 224, 224)

    generator = Generator(latent_dim, img_shape, num_classes)
    generator.load_state_dict(torch.load("generator_res.pth"))

    generator.eval()

    z = torch.randn(1, latent_dim).to(device)
    label = torch.tensor([4]).to(device)  # Replace 5 with the desired label
    with torch.no_grad():
        generated_image = generator(z, label)

    save_image(inverse_transform(generated_image[0]), "generated_image.png")


if __name__ == '__main__':
    training()

