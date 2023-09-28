import argparse

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

ROOT = "/media/tx-deepocean/Data/DICOMS/demos/ML_DATA"


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path", type=str, default=f"{ROOT}/cat.jpeg", help="add image abs path"
    )
    return parser.parse_known_args()[0] if known else parser.parse_args()


def convert_img_to_tensor(img_path):
    img = Image.open(img_path)
    print(img)
    convert_tensor = transforms.Compose([transforms.ToTensor()])
    img = convert_tensor(img)

    writer = SummaryWriter("./logs")
    writer.add_image("Tensor Image", img, 0)
    writer.close()


def main(opt):
    print(opt.img_path)
    convert_img_to_tensor(opt.img_path)


if __name__ == "__main__":
    opt = parse_opt(known=True)
    main(opt)
