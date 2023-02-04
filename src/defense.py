import _init_path
import os
import torch
import torchvision as tv
import io
from torch import nn
from scipy import ndimage
from PIL import Image

__all__ = [
    "jpeg", "bitdepth", "guassian_noise", "median_blur", 
    "AutoEncoder1", "AutoEncoder2", "AutoEncoder3",
    "AutoEncoder4", "AutoEncoder5"
]


def jpeg(img, quality=70):
    assert isinstance(img, torch.Tensor)
    device = img.device
    tr1 = tv.transforms.ToPILImage()
    tr2 = tv.transforms.ToTensor()
    res = []
    for x in img:
        x = tr1(x)
        tmp = io.BytesIO()
        x.save(tmp, format='jpeg', quality=quality)
        x = tr2(Image.open(tmp)).to(device)
        res.append(x)
    img = torch.stack(res)
    return img


def bitdepth(img, depth=4):
    assert isinstance(img, torch.Tensor)
    k = 2**depth - 1
    img = torch.round(img * k) / k
    return img


def guassian_noise(img, a=0.03):
    assert isinstance(img, torch.Tensor)
    noise = a * torch.randn_like(img)
    img = torch.clamp(img + noise, 0, 1)
    return img


def median_blur(img, ksize=3):
    assert isinstance(img, torch.Tensor)
    device = img.device
    img = img.cpu().detach().numpy()
    img = ndimage.filters.median_filter(img, size=(1, 1, ksize, ksize), mode='reflect')
    img = torch.from_numpy(img).to(device)
    return img


class AutoEncoder1(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.model.load_state_dict(torch.load("./weights/model1.pth"))
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        x = x.view(-1, 1, *x.shape[2:])
        x = self.model(x)
        x = x.view(-1, 3, *x.shape[2:])
        x = torch.clamp(x, 0, 1)
        return x


class AutoEncoder2(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.model.load_state_dict(torch.load("./weights/model2.pth"))
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        x = x.view(-1, 1, *x.shape[2:])
        x = self.model(x)
        x = x.view(-1, 3, *x.shape[2:])
        x = torch.clamp(x, 0, 1)
        return x


class AutoEncoder3(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.model.load_state_dict(torch.load("./weights/model3.pth"))
        self.model.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        x = self.model(x)
        x = torch.clamp(x, 0, 1)
        return x


class AutoEncoder4(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.model.load_state_dict(torch.load("./weights/model4.pth"))
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        x = self.model(x)
        x = torch.clamp(x, 0, 1)
        return x


class AutoEncoder5(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.model.load_state_dict(torch.load("./weights/model5.pth"))
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        x = self.model(x)
        x = torch.clamp(x, 0, 1)
        return x


