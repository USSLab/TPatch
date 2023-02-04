import _init_path
import os
import math
import random
import torch
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from PIL import Image
from classifier import *
from detector import *
from train_eval import *
from kitti import *
from tpatch import *

class AdaptiveAxes:
    def __init__(self,
                 n_figure: int,
                 n_col: int = 4,
                 fig_size: tuple = (7, 5)) -> None:
        self.n = n_figure
        self.n_col = min(n_figure, n_col)
        self.n_row = (n_figure + n_col - 1) // n_col
        self.fig_size = (fig_size[0] * self.n_col, fig_size[1] * self.n_row)
        self.fig, self.axes = plt.subplots(self.n_row,
                                           self.n_col,
                                           squeeze=False,
                                           figsize=self.fig_size)

    def __iter__(self):
        for i in range(self.n):
            j = i // self.n_col
            k = i % self.n_col
            yield self.axes[j][k]


coco_img = "dataset/mscoco/val2014"
coco_ann = "dataset/mscoco/annotations/instances_val2014.json"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda:0"
out_dir = "demo"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
attack_type = "CA"
target_name = "stop sign"
double_apply = False
model_type = "rcnn"
converter = LabelConverter()
if model_type == "rcnn":
    names = converter.category91
elif model_type == "yolov3":
    names = converter.category80
elif model_type == "yolov5":
    names = converter.category80
target_id = names.index(target_name)
model = get_det_model(device, model_type)
trigger_gen = TriggerRegion((0.02, 0.08), (100., 140.), (0, 20))
trigger_func = lambda x, tr: stn_blur_general(x, tr[0], tr[1], math.radians(tr[2]), 60, device)

bgsize = (200, 200)

psize = (70, 170) if not double_apply else (40, 110)
relpos = (65, 15) if not double_apply else (28, 45)
relpos3 = (132, 45)

eot = True
eot_scale = 0.2
eot_angle = math.pi / 9
epoch = 50
repeat = 20
num = 100
lr = 1e-2
momentum = 0.9
alpha = 0.4 if attack_type == "CA" else 5
beta = 3e-6 if attack_type == "CA" else 3e-6
ceta = 3e-3 if attack_type == "CA" else 3e-3
delta = 1e-6 if attack_type == "CA" else 1e-6


loader1 = load_coco(coco_img, coco_ann)
loader2 = load_kitti()

patch2 = TPatch(bgsize[0], bgsize[1], target_id, device, eot=True, 
                eot_scale=eot_scale, eot_angle=eot_angle, p=1)
resize = tv.transforms.Resize(bgsize)
quick_load = lambda x: resize(patch2.pil2tensor(Image.open(x))).unsqueeze(0).to(device)

patch2.data = quick_load("stop_sign.png")
patch2.load_mask("stop_sign_mask.png")
patch2.rotate_mask = resize(patch2.rotate_mask)

if attack_type == "CA":
    content = "pikachu.jpg"
elif attack_type == "HA":
    if not double_apply:
        content = "stop.png"
    else:
        content = "usenix_text.png"

a = tv.models.vgg19(True).to(device)
content_loss = ContentLoss(a.features, content, device, extract_layer=11)
tv_loss = TVLoss()

filename = f"{random.randint(0, 999999):06d}.png"
while os.path.exists(os.path.join(out_dir, filename)):
    filename = f"{random.randint(0, 999999):06d}.png"

# === eval only ===
# filename = "137498.png"
# === eval only ===

if attack_type == "HA":
    patch = TPatch(psize[0], psize[1], target_id, device=device, lr=lr, momentum=momentum, 
                   eot=eot, eot_scale=0.97, eot_angle=math.pi/60)
else:
    patch = TPatch(bgsize[0], bgsize[1], target_id, device=device, lr=lr, momentum=momentum, 
                   eot=eot, eot_scale=eot_scale, eot_angle=eot_angle, p=1)
save_path = os.path.join(out_dir, filename)

if os.path.exists(os.path.join(out_dir, filename)):
    patch.load(save_path)

logfile = (save_path).replace("png", "txt")
print("=" * 40)
print(datetime.now())
print(save_path)
print(attack_type)
print(model_type)
print(alpha, beta, ceta, delta, psize)

def pn_loss(imgo, gt_box, dummy_box, p_tr, n_tr):
    imgp = trigger_func(imgo, p_tr)
    if attack_type == "CA":
        loss1 = model(imgp, gt_box)
    elif attack_type == "HA":
        loss1 = model(imgp, gt_box, hiding=True)
    imgn = trigger_func(imgo, n_tr)
    if attack_type == "CA":
        loss2 = model(imgn, gt_box, hiding=True)
    elif attack_type == "HA":
        loss2 = model(imgn, gt_box)
    return loss1, loss2


def train(train_loader):
    if model_type.startswith("rcnn"):
        model.train()
    else:
        model.eval()
    t1 = datetime.now()
    log_loss = torch.zeros(4, device=device)
    
    for i, img in enumerate(train_loader, 1):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        for j in range(repeat):
            if attack_type == "CA":
                pos = patch.random_pos((h, w))
                imgo = patch.apply(img, pos)
                gt_box, dummy_box = _make_boxes(patch, pos, model_type[:4].upper())
                last_scale = patch.last_scale
            elif attack_type == "HA":
                pos2 = patch2.random_pos((h, w))
                dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                relpos2 = (relpos[0] + dx, relpos[1] + dy)
                patch2.data = patch.apply(quick_load("stop_sign.png"), relpos2, do_random_color=True)
                if double_apply:
                    dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                    relpos2 = (relpos3[0] + dx, relpos3[1] + dy)
                    patch2.data = patch.apply(patch2.data, relpos2, do_random_color=True)
                imgo = patch2.apply(img, pos2, do_random_color=False)
                gt_box, dummy_box = _make_boxes(patch2, pos2, model_type[:4].upper())
                last_scale = patch2.last_scale

            p_tr = trigger_gen.sample_pos()
            n_tr = trigger_gen.sample_neg()
            
            loss1, loss2 = pn_loss(imgo, gt_box, dummy_box, p_tr, n_tr)
            loss3 = tv_loss(patch.data)
            loss4 = content_loss(patch.data)
            loss = (1/last_scale**2)*(loss1 + alpha*loss2) + beta*loss3 + ceta*loss4
            if torch.isnan(loss).any(): continue
            log_loss += torch.tensor((loss1.item(), loss2.item(), loss3.item(), loss4.item()), device=device)
            patch.update(loss)
            
        if i == 1:
            t2 = datetime.now()
            pred_time = (t2-t1) * (epoch-1)
            print("pred time:", pred_time)
        if i == epoch:
            break
    return log_loss / (epoch * repeat)


def eval(test_loader):
    model.eval()
    t1 = datetime.now()
    success, success_1, success_2 = 0, 0, 0
    logs = []
    
    for i, img in enumerate(test_loader, 1):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        set_resize = random.uniform(eot_scale, 1)
        set_rotate = random.uniform(-eot_angle, eot_angle)
        
        if attack_type == "CA":
            pos = patch.random_pos((h, w))
            imgo = patch.apply(img, pos, test_mode=True, set_resize=set_resize, 
                               set_rotate=set_rotate)
        elif attack_type == "HA":
            pos = patch2.random_pos((h, w))
            dx, dy = 0, 0
            relpos2 = (relpos[0]+dx, relpos[1]+dy)
            patch2.data = patch.apply(quick_load("stop_sign.png"), relpos2, test_mode=True, do_random_color=False)
            if double_apply:
                dx, dy = 0, 0
                relpos2 = (relpos3[0]+dx, relpos3[1]+dy)
                patch2.data = patch.apply(patch2.data, relpos2, test_mode=True, do_random_color=False)
            imgo = patch2.apply(img, pos, test_mode=True, set_resize=set_resize, set_rotate=set_rotate, do_random_color=True)

        p_tr = trigger_gen.sample_pos()
        imgp = trigger_func(imgo, p_tr)
        n_tr = trigger_gen.sample_neg()
        imgn = trigger_func(imgo, n_tr)

        pred1 = model(imgn)[0]
        pred2 = model(imgp)[0]

        if attack_type == "CA":
            w, h = patch.w, patch.h
        elif attack_type == "HA":
            w, h = patch2.w, patch2.h
        gt_box = torch.tensor([[
            pos[1] + (1 - set_resize) * w * 0.5, 
            pos[0] + (1 - set_resize) * h * 0.5, 
            pos[1] + (1 + set_resize) * w * 0.5, 
            pos[0] + (1 + set_resize) * h * 0.5, 
            patch.target,
        ]])

        flag1 = isappear(pred1.cpu(), gt_box)
        flag2 = isappear(pred2.cpu(), gt_box)

        s, s1, s2 = 0, 0, 0
        if attack_type == "HA":
            if flag1 and not flag2:
                s = 1
            if flag1:
                s1 = 1
            if not flag2:
                s2 = 1
        elif attack_type == "CA":
            if not flag1 and flag2:
                s = 1
            if not flag1:
                s1 = 1
            if flag2:
                s2 = 1
        success += s
        success_1 += s1
        success_2 += s2

        p_r = (p_tr[0]**2 + p_tr[1]**2)**0.5
        p_th = np.arctan2(p_tr[0], p_tr[1])
        n_r = (n_tr[0]**2 + n_tr[1]**2)**0.5
        n_th = np.arctan2(n_tr[0], n_tr[1])
        log = [set_resize, np.degrees(set_rotate), p_r, np.degrees(p_th), p_tr[2], 
               n_r, np.degrees(n_th), n_tr[2], s1, s2, s]
        logs.append(log)

        if i == 10:
            t2 = datetime.now()
            pred_time = (t2-t1) * (num-10) / 10
            print("pred time:", pred_time)
        if i == num:
            break
    return (success, success_1, success_2), logs


def main():
    decay_epoch = 2
    n_decay = 3
    for e in range(1, decay_epoch * n_decay + 1):
        print(f"Epoch {e}: start training...")
        losses = train(loader1)
        print(losses)
        print(f"Epoch {e}: start evaluating...")
        sucs, logs = eval(loader2)
        print(sucs)
        patch.save(save_path)
        if e % decay_epoch == 0:
            patch.opt.lr *= 0.3

if __name__ == "__main__":
    main()
