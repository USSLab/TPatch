import cv2
import torch
import math
import torchvision as tv
import torch.nn.functional as F
import random
import logging
import numpy as np
from glob import glob
from PIL import Image
from torch import nn
from torch.utils.data import random_split, DataLoader
from typing import Callable, Tuple


def _gamma_correction(imgs: torch.Tensor, gamma: float) -> torch.Tensor:
    n = imgs.shape[0]
    out = 0
    for img in imgs:
        # 从RGB到光
        out += (1e-6 + img)**gamma
    # 从光到RGB
    out = (1e-6 + out/n)**(1 / gamma)
    return out


def radial_blur(img, s, div, device):
    s = torch.linspace(1 - s, 1, div, device=device)
    zeros = torch.zeros_like(s)
    affine_tensor = torch.stack([
        torch.stack([s, zeros, zeros]),
        torch.stack([zeros, s, zeros]),
    ]).permute(2, 0, 1)
    grid = F.affine_grid(affine_tensor, [div, *img.shape[1:]], align_corners=False)
    imgs = img.unsqueeze(dim=1).expand(-1, div, -1, -1, -1)
    res = []
    for i in range(img.shape[0]):
        samples = F.grid_sample(imgs[i], grid, padding_mode="border", align_corners=False)
        blur_img = torch.mean(samples, dim=0, keepdim=True)
        res.append(blur_img)
    res = torch.cat(res, dim=0)
    return res


def _sine_grid(div: int, device: str) -> torch.Tensor:
    grid = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, div, device=device)
    grid = (torch.sin(grid) + 1) / 2
    return grid


def _sine_grid_new(div: int, device: str, phi: float) -> torch.Tensor:
    grid = torch.linspace(-math.pi, math.pi, div, device=device)
    grid = torch.sin(grid + phi)
    return grid


def _stn_blur(img: torch.Tensor, div: int, x: torch.Tensor, y: torch.Tensor, 
              mean_func: Callable, device: str) -> torch.Tensor:
    ones = torch.ones_like(x, device=device)
    zeros = torch.zeros_like(x, device=device)
    affine_tensor = torch.stack([
        torch.stack([ones, zeros, x]),
        torch.stack([zeros, ones, y]),
    ]).permute(2, 0, 1)
    grid = F.affine_grid(affine_tensor, [div, *img.shape[1:]], align_corners=False)
    imgs = img.unsqueeze(dim=1).expand(-1, div, -1, -1, -1)
    res = []
    for i in range(img.shape[0]):
        samples = F.grid_sample(imgs[i], grid, padding_mode="border", align_corners=False)
        blur_img = mean_func(samples)
        res.append(blur_img)
    res = torch.cat(res, dim=0)
    return res


def _stn_blur_linear(img: torch.Tensor, div: int, x: float, y: float, blur_grid: torch.Tensor,
                     mean_func: Callable, device: str) -> torch.Tensor:
    ones = torch.ones_like(blur_grid, device=device)
    zeros = torch.zeros_like(blur_grid, device=device)
    x = x * blur_grid
    y = y * blur_grid
    affine_tensor = torch.stack([
        torch.stack([ones, zeros, x]),
        torch.stack([zeros, ones, y]),
    ]).permute(2, 0, 1)
    grid = F.affine_grid(affine_tensor, [div, *img.shape[1:]], align_corners=False)
    imgs = img.unsqueeze(dim=1).expand(-1, div, -1, -1, -1)
    res = []
    for i in range(img.shape[0]):
        samples = F.grid_sample(imgs[i], grid, padding_mode="border", align_corners=False)
        blur_img = mean_func(samples)
        res.append(blur_img)
    res = torch.cat(res, dim=0)
    return res


def stn_blur_2d(img: torch.Tensor, x: float, y: float, div: int, device: str) -> torch.Tensor:
    blur_grid = torch.linspace(0, 1, div, device=device)
    mean_func = lambda x: torch.mean(x, dim=0, keepdim=True)
    res = _stn_blur_linear(img, div, x, y, blur_grid, mean_func, device)
    return res


def stn_blur_2d_gamma(img: torch.Tensor,
                      x: float,
                      y: float,
                      div: int,
                      device: str,
                      gamma: float = 2.2) -> torch.Tensor:
    blur_grid = torch.linspace(0, 1, div, device=device)
    mean_func = lambda x: _gamma_correction(x, gamma).unsqueeze(0)
    res = _stn_blur_linear(img, div, x, y, blur_grid, mean_func, device)
    return res


def stn_blur_2d_sin(img: torch.Tensor, x: float, y: float, div: int, device: str) -> torch.Tensor:
    blur_grid = _sine_grid(div, device)
    mean_func = lambda x: torch.mean(x, dim=0, keepdim=True)
    res = _stn_blur_linear(img, div, x, y, blur_grid, mean_func, device)
    return res


def stn_blur_2d_gamma_sin(img: torch.Tensor,
                          x: float,
                          y: float,
                          div: int,
                          device: str,
                          gamma: float = 2.2) -> torch.Tensor:
    blur_grid = _sine_grid(div, device)
    mean_func = lambda x: _gamma_correction(x, gamma).unsqueeze(0)
    res = _stn_blur_linear(img, div, x, y, blur_grid, mean_func, device)
    return res


def stn_blur_general(img: torch.Tensor,
                     x: float,
                     y: float,
                     phi: float,
                     div: int,
                     device: str,
                     gamma: float = 2.2) -> torch.Tensor:
    x = _sine_grid_new(div, device, phi) * x
    y = _sine_grid_new(div, device, 0) * y
    mean_func = lambda x: _gamma_correction(x, gamma).unsqueeze(0)
    res = _stn_blur(img, div, x, y, mean_func, device)
    return res


def load_imagenet_preprocess() -> tv.transforms.Normalize:
    return tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def load_imagenet_val(dataset: str,
                      batch_size: int = 1,
                      size: int = 50000,
                      shuffle: bool = True,
                      inc: bool = False) -> DataLoader:
    imagenet = tv.datasets.ImageFolder(dataset,
                                       transform=tv.transforms.Compose([
                                           tv.transforms.Resize(299),
                                           tv.transforms.CenterCrop((299, 299)),
                                           tv.transforms.ToTensor(),
                                       ]) if inc else tv.transforms.Compose([
                                           tv.transforms.Resize(256),
                                           tv.transforms.CenterCrop((224, 224)),
                                           tv.transforms.ToTensor(),
                                       ]))
    if size != 50000:
        partial = [size, 50000 - size]
        imagenet, _ = random_split(imagenet, partial)
    return DataLoader(
        imagenet,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=5 if batch_size >= 10 else 0,
    )


def read_img(path: str, device: str, crop_size: int = None) -> torch.Tensor:
    if crop_size is None:
        tr = tv.transforms.ToTensor()
    else:
        tr = tv.transforms.Compose([
            tv.transforms.Resize(crop_size),
            tv.transforms.CenterCrop((crop_size, crop_size)),
            tv.transforms.ToTensor(),
        ])
    return tr(Image.open(path)).unsqueeze(0).to(device)


class ImageOnlyLoader:
    def __init__(self, glob_path: str, transform: Callable, shuffle: bool = True) -> None:
        self.img_names = sorted(glob(glob_path))
        self.length = len(self.img_names)
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.img_names)
        self.pil2tensor = tv.transforms.ToTensor()
        self.transform = transform

    def __getitem__(self, key: int) -> torch.Tensor:
        img = self.transform(self.pil2tensor(Image.open(self.img_names[key]))).unsqueeze(dim=0)
        # deal with gray images
        if img.shape[1] == 1:
            img = torch.cat([img] * 3, dim=1)
        return img

    def __len__(self) -> int:
        return self.length


class MIFGSM(nn.Module):
    def __init__(self, m: float, lr: float):
        super().__init__()
        self.m = m
        self.lr = lr
        self.h = 0

    @torch.no_grad()
    def forward(self, t: torch.Tensor) -> None:
        l1 = t.grad.abs().mean()
        if l1 == 0:
            l1 += 1
        self.h = self.h * self.m + t.grad / l1
        t.data -= self.lr * self.h.sign()
        t.grad.zero_()


class TPatch:
    def __init__(self,
                 h: int,
                 w: int,
                 target: int = None,
                 device: str = "cpu",
                 lr: float = 1 / 255,
                 momentum: float = 0.9,
                 eot: bool = False,
                 eot_angle: float = math.pi / 9,
                 eot_scale: float = 0.8,
                 p: float = 0.5):
        if eot:
            self.robust = EoT(angle=eot_angle, scale=eot_scale, p=p)
        self.eot = eot
        self.w = int(w)
        self.h = int(h)
        self.shape = [1, 3, self.h, self.w]
        self.device = device
        self.data = torch.rand(self.shape, device=device, requires_grad=True)
        self.opt = MIFGSM(m=momentum, lr=lr)
        self.pil2tensor = tv.transforms.ToTensor()
        self.last_scale = 1.0
        self.target = target
        self.rotate_mask = None

    def apply(self,
              img: torch.Tensor,
              pos: Tuple[int, int],
              test_mode: bool = False,
              set_rotate: float = None,
              set_resize: float = None,
              do_random_color: bool = True,
              transform: Callable = None) -> torch.Tensor:
        assert len(pos) == 2, "pos should be (x, y)"
        if self.eot:
            if test_mode:
                switch, padding, _ = self.robust(self,
                                                 pos,
                                                 img.shape[-2:],
                                                 do_random_color=do_random_color,
                                                 do_random_rotate=False,
                                                 do_random_resize=False,
                                                 set_rotate=set_rotate,
                                                 set_resize=set_resize,
                                                 rotate_mask=self.rotate_mask)
            else:
                switch, padding, self.last_scale = self.robust(self, pos, img.shape[-2:], 
                                                               do_random_color=do_random_color,
                                                               rotate_mask=self.rotate_mask)
        else:
            switch, padding = self.mask(img.shape, pos)
        if transform:
            padding = transform(padding)
        return (1-switch) * img + switch * padding.clone()

    def update(self, loss: torch.Tensor) -> None:
        loss.backward()
        self.opt(self.data)
        self.data.data.clamp_(0, 1)

    def mask(self, shape: torch.Size, pos: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(shape, dtype=torch.float, device=self.device)
        mask[..., pos[0]:pos[0] + self.h, pos[1]:pos[1] + self.w] = 1
        padding = torch.zeros(shape, dtype=torch.float, device=self.device)
        padding[..., pos[0]:pos[0] + self.h, pos[1]:pos[1] + self.w] = self.data
        return mask, padding

    def random_pos(self, shape: torch.Size) -> Tuple[int, int]:
        h = random.randint(0, shape[-2] - self.h)
        w = random.randint(0, shape[-1] - self.w)
        return h, w

    def save(self, path: str):
        tv.utils.save_image(self.data, path)

    def load(self, path: str):
        self.data = self.pil2tensor(Image.open(path))
        self.data = self.data.unsqueeze(0).to(self.device)
        self.data.requires_grad_()
        self.shape = list(self.data.shape)
        _, _, self.h, self.w = self.shape

    def load_mask(self, path:str):
        self.rotate_mask = self.pil2tensor(Image.open(path))
        if self.rotate_mask.shape[0] != 3:
            self.rotate_mask = self.rotate_mask.expand((3, -1, -1))
        self.rotate_mask = self.rotate_mask.unsqueeze(0).to(self.device)


class EoT(nn.Module):
    def __init__(self, angle=math.pi / 9, scale=0.8, p=0.5, brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1):
        super(EoT, self).__init__()
        self.angle = angle
        self.scale = scale
        self.p = p
        self.color = tv.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def forward(self,
                patch: TPatch,
                pos: Tuple[int, int],
                img_shape: Tuple[int, int],
                do_random_rotate=True,
                do_random_color=True,
                do_random_resize=True,
                set_rotate=None,
                set_resize=None,
                rotate_mask=None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if torch.rand(1) > self.p:
            do_random_rotate = False
            do_random_color = False
            do_random_resize = False

        if do_random_color:
            img = self.color(patch.data)
            img = img + torch.randn_like(img) * 8 / 255
            img = torch.clamp(img, 0, 1)
        else:
            img = patch.data

        if do_random_rotate:
            angle = torch.FloatTensor(1).uniform_(-self.angle, self.angle)
        elif set_rotate is None:
            angle = torch.zeros(1)
        else:
            angle = torch.full((1, ), set_rotate)

        pre_scale = 1 / (torch.cos(angle) + torch.sin(torch.abs(angle)))
        pre_scale = pre_scale.item()

        if do_random_resize:
            min_scale = min(self.scale / pre_scale, 1.0)
            scale_ratio = torch.FloatTensor(1).uniform_(min_scale, 1)
        elif set_resize is None:
            scale_ratio = torch.ones(1)
        else:
            scale_ratio = torch.full((1, ), set_resize)

        scale = scale_ratio * pre_scale
        logging.debug(f"scale_ratio: {scale_ratio.item():.2f}, "
                      f"angle: {angle.item():.2f}, pre_scale: {pre_scale:.2f}, "
                      f"scale: {scale.item():.2f}, ")

        t = -torch.ceil(torch.log2(scale))
        t = 1 << int(t.item())
        if t > 1:
            size = (patch.h // t, patch.w // t)
            img = F.interpolate(img, size=size, mode="area")
            scale *= t

        angle = angle.to(patch.device)
        scale = scale.to(patch.device)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        theta = torch.zeros((1, 2, 3), device=patch.device)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = 0
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = 0

        size = torch.Size((1, 3, patch.h // t, patch.w // t))
        grid = F.affine_grid(theta, size, align_corners=False)
        output = F.grid_sample(img, grid, align_corners=False)

        if rotate_mask is None:
            rotate_mask = torch.ones(size, device=patch.device)
        mask = F.grid_sample(rotate_mask, grid, align_corners=False)

        tw1 = (patch.w - patch.w // t) // 2
        tw2 = patch.w - patch.w // t - tw1
        th1 = (patch.h - patch.h // t) // 2
        th2 = patch.h - patch.h // t - th1

        pad = nn.ZeroPad2d(padding=(
            pos[1] + tw1,
            img_shape[1] - patch.w - pos[1] + tw2,
            pos[0] + th1,
            img_shape[0] - patch.h - pos[0] + th2,
        ))
        mask = pad(mask)
        padding = pad(output)
        mask = torch.clamp(mask, 0, 1)

        return mask, padding, scale_ratio.item()


class TVLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lr = torch.abs(x[..., :, 1:] - x[..., :, :-1]).sum()
        tb = torch.abs(x[..., 1:, :] - x[..., :-1, :]).sum()
        return lr + tb


class ContentLoss(nn.Module):
    def __init__(self, extractor: nn.Module, ref_fp: str, device: str, extract_layer=20) -> None:
        super().__init__()
        self.extractor = extractor
        self.content_hook = extract_layer
        self.preprocess = load_imagenet_preprocess()
        self.resize = tv.transforms.Compose([
            tv.transforms.Resize([224, 224], interpolation=Image.BICUBIC),
            tv.transforms.ToTensor(),
        ])
        self.ref = self.resize(Image.open(ref_fp))[:3]
        self.ref = self.ref.unsqueeze(0).to(device)
        self.ref = self.get_content_layer(self.ref).detach()
        self.upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)

    def get_content_layer(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        for i, m in enumerate(self.extractor.children()):
            x = m(x)
            if i == self.content_hook:
                break
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.get_content_layer(x)
        loss = F.mse_loss(x, self.ref)
        return loss


class TriggerRegion:
    def __init__(self, r_min_max: Tuple[float, float], th_min_max: Tuple[float, float], z_min_max=(0, 90)) -> None:
        self.r_min, self.r_max = r_min_max
        self.th_min, self.th_max = th_min_max
        self.th_min = np.radians(self.th_min)
        self.th_max = np.radians(self.th_max)
        self.z_min, self.z_max = z_min_max
    
    def sample_pos(self) -> Tuple[float, float, float]:
        r = random.uniform(self.r_min, self.r_max)
        th = random.uniform(self.th_min, self.th_max)
        x = r * np.cos(th)
        y = r * np.sin(th)
        z = random.uniform(self.z_min, self.z_max)
        return x, y, z
    
    def sample_neg(self) -> Tuple[float, float, float]:
        if random.random() < 0.5:
            r = random.uniform(0, self.r_min)
            th = random.uniform(self.th_min, self.th_max)
        else:
            while True:
                r = random.uniform(0, self.r_max)
                th = random.uniform(0, math.pi)
                if not self.th_min < th < self.th_max:
                    break
        x = r * np.cos(th)
        y = r * np.sin(th)
        z = random.uniform(self.z_min, self.z_max)
        return x, y, z
