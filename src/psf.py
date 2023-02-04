import _init_path
import os
import cv2
import torch
import torchvision as tv
from tqdm import tqdm
from PIL import Image

def gamma_encode(img, gamma):
    return (1e-6 + img) ** (1/gamma)

def gamma_decode(img, gamma):
    return (1e-6 + img) ** gamma

kernel_size = 45
cutting = kernel_size // 2
device = torch.device(0)
tf = tv.transforms.Compose([
    tv.transforms.CenterCrop([1080, 1080]),
    tv.transforms.Resize([540, 540]),
    tv.transforms.ToTensor(),
])
out_dir = "images_from_video/s20_20220521180615"
for num in tqdm(range(50, 70, 10)):
    layer = torch.nn.Conv2d(in_channels=1, out_channels=1,
                            kernel_size=kernel_size, bias=False)
    layer.to(device)
    img1 = os.path.join(out_dir, "000040.png")
    img2 = os.path.join(out_dir, f"{num:06d}.png")
    img1 = tf(Image.open(img1)).unsqueeze(0).to(device)
    img2 = tf(Image.open(img2)).unsqueeze(0).to(device)
    img1 = gamma_decode(img1, 2.2)
    img2 = gamma_decode(img2, 2.2)
    img2 = img2[..., cutting:-cutting, cutting:-cutting]
    metric = torch.nn.MSELoss()
    opt = torch.optim.SGD(layer.parameters(), lr=1e-3, momentum=0.9)
    for i in range(500):
        img3 = layer(img1.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        loss = metric(img3, img2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        for k in layer.parameters():
            k.data.clamp_(min=0, max=1)
        print(loss)
    for k in layer.parameters():
        print(k.sum())
        tv.utils.save_image((k.div(k.max())).squeeze(),
                            os.path.join(out_dir, f"pred_kernel_g_{num}_.png"))

