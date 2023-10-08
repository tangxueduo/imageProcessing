# tensorboard usage
# import keyword

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default


def add_image_graph():
    """还有一个add_images"""
    writer = SummaryWriter("./logs")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = datasets.MNIST(
        "mnist_train", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image("images", grid, 0)
    writer.add_graph(model, images)
    writer.close()


def add_scalar():
    """add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)
    还有一个 add_scalars
    """
    writer = SummaryWriter(log_dir="./logs")

    for n_iter in range(100):
        writer.add_scalar("Loss/train", np.random.random(), n_iter)
        writer.add_scalar("Loss/test", np.random.random(), n_iter)
        writer.add_scalar("Accuracy/train", np.random.random(), n_iter)
        writer.add_scalar("Accuracy/test", np.random.random(), n_iter)
    writer.close()


def add_video():
    pass
    # writer = SummaryWriter()


def add_audio():
    # writer = SummaryWriter()
    pass


def add_histogram(
    tag, values, global_step=None, bins="tensorflow", walltime=None, max_bins=None
):
    pass


def add_figure(tag, figure, global_step=None, close=True, walltime=None):
    pass


def add_text(tag, text_string, global_step=None, walltime=None):
    pass


def add_graph(model, input_to_model=None, verbose=False, use_strict_trace=True):
    pass


def add_embedding():
    writer = SummaryWriter("./logs")
    labels = np.random.randint(2, size=100)  # binary label
    predictions = np.random.rand(100)
    writer.add_pr_curve("pr_curve", labels, predictions, 0)
    writer.close()


def main():
    # add_image_graph()
    add_scalar()
    add_embedding()
    # other method: https://pytorch.org/docs/stable/tensorboard.html#


if __name__ == "__main__":
    main()
