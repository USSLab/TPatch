import _init_path
import os
import torchvision as tv
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader


class Background(Dataset):
    def __init__(self, path, size=640) -> None:
        super().__init__()
        self.img_names = sorted(glob(os.path.join(path, "*.png")))
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(size),
            tv.transforms.CenterCrop((size, size)),
            tv.transforms.ToTensor()
        ])
        self.process = lambda x: self.transform(Image.open(x))
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = self.process(img_name)
        return img

def load_kitti(batch_size=1, size=640):
    path = "dataset/KITTI/mask_samples"
    loader = DataLoader(
        Background(path, size=size),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader

