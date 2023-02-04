import random
import torch
import torchvision as tv
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from tpatch import TVLoss, TPatch, ContentLoss

__all__ = [
    "load_coco", "LabelConverter", "eval_det", "eval_det_details", "train_det_CA", "train_det_HA",
    "xyxy2cxcywh", "train_det_CA_full", "eval_det_full", "train_det_HA_full", "cxcywh2xyxy",
    "_make_boxes", "train_det_HA_general", "eval_det_general", "eval_cls_full", "train_cls_AA",
    "isappear"
]


def xyxy2cxcywh(box):
    cx = (box[..., 0] + box[..., 2]) / 2
    cy = (box[..., 1] + box[..., 3]) / 2
    w = box[..., 2] - box[..., 0]
    h = box[..., 3] - box[..., 1]
    return torch.stack((cx, cy, w, h), dim=-1)


def xywh2xyxy(box):
    x1 = box[..., 0]
    x2 = box[..., 0] + box[..., 2]
    y1 = box[..., 1]
    y2 = box[..., 1] + box[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)


def cxcywh2xyxy(box):
    x1 = box[..., 0] - box[..., 2] / 2
    x2 = box[..., 0] + box[..., 2] / 2
    y1 = box[..., 1] - box[..., 3] / 2
    y2 = box[..., 1] + box[..., 3] / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def compute_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, xyxy: bool = True) -> torch.Tensor:
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u
    return iou


def load_coco(img_path, ann_path):
    return DataLoader(
        tv.datasets.CocoDetection(
            img_path,
            ann_path,
            transform=tv.transforms.Compose([
                tv.transforms.Resize(640),
                tv.transforms.CenterCrop((640, 640)),
                tv.transforms.ToTensor(),
            ]),
        ),
        shuffle=True,
    )


class LabelConverter:
    def __init__(self) -> None:
        self.from91to80 = torch.tensor([
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, -1, 24, 25, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, -1, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, -1, 60, -1, -1, 61,
            -1, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, -1, 73, 74, 75, 76, 77, 78, 79
        ])
        self.from80to91 = torch.tensor([i for i, x in enumerate(self.from91to80) if x != -1])
        self.category91 = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
            'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.category80 = [x for x in self.category91[1:] if x != 'N/A']

    def coco2rcnn(self, targets):
        if len(targets) and isinstance(targets[0], dict):
            targets = [targets]
        ds = []
        for target in targets:
            d = {}
            boxes = []
            labels = []
            for ist in target:
                boxes.append(torch.stack(ist["bbox"], dim=1))
                labels.append(ist["category_id"])
            boxes = torch.cat(boxes, dim=0)
            labels = torch.cat(labels, dim=0)
            d["boxes"] = xywh2xyxy(boxes)
            d["labels"] = labels
            ds.append(d)
        return ds

    def rcnn2yolo(self, targets):
        cache = []
        for i, d in enumerate(targets):
            label = self.from91to80[d["labels"].long()].unsqueeze(1)
            boxes = xyxy2cxcywh(d["boxes"]) / 640
            imgid = torch.full_like(label, i)
            label = torch.cat((imgid, label, boxes), dim=1)
            cache.append(label)
        return torch.cat(cache, dim=0) if len(cache) else torch.empty((0, 6))


def isappear(pred, gt):
    select = pred[:, -1] == gt[0, -1]
    pred = pred[select]
    if pred.shape[0] == 0:
        return False
    box1 = pred[:, :4]
    box2 = gt[:, :4]
    iou = compute_iou(box1, box2)
    if (iou > 0.5).any():
        return True
    return False


@torch.no_grad()
def eval_AA(patch: TPatch, trigger_func, p_trs, n_trs, model: nn.Module, loader: DataLoader, num: int):
    success = 0
    success_1 = 0
    success_2 = 0
    model.eval()
    t1 = datetime.now()
    for i, img in enumerate(loader, 1):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]

        img = img.to(patch.device)
        h, w = img.shape[-2:]
        pos = patch.random_pos((h, w))

        imgo = patch.apply(img, pos, test_mode=True, do_random_color=False)
        p_tr = random.choice(p_trs)
        imgp = trigger_func(imgo, p_tr)
        n_tr = random.choice(n_trs)
        imgn = trigger_func(imgo, n_tr)

        pred1 = model(imgn)
        pred2 = model(imgp)

        flag1 = pred1 != patch.target
        flag2 = pred2 == patch.target

        success += (flag1 & flag2).sum().item()
        success_1 += flag1.sum().item()
        success_2 += flag2.sum().item()

        if i == 10:
            t2 = datetime.now()
            pred_time = (t2-t1) * (num-10) / 10
            print("pred time:", pred_time)
        if i == num:
            break
    return success, success_1, success_2


@torch.no_grad()
def eval_HA_or_CA(patch: TPatch,
                  trigger_func,
                  detector: nn.Module,
                  loader: DataLoader,
                  num: int,
                  attack_type: str,
                  defense=None):
    assert attack_type in ["HA", "CA"], "attack should be either HA or AA"

    success = 0
    detector.eval()
    t1 = datetime.now()
    for i, img in enumerate(loader, 1):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        pos = patch.random_pos((h, w))

        img1 = patch.apply(img, pos, test_mode=True)
        if defense is not None:
            img1 = defense(img1)
        img2 = trigger_func(img1)

        pred1 = detector(img1)[0]
        pred2 = detector(img2)[0]

        gt_box = torch.tensor([[pos[1], pos[0], pos[1] + patch.w, pos[0] + patch.h, patch.target]])
        flag1 = isappear(pred1.cpu(), gt_box)
        flag2 = isappear(pred2.cpu(), gt_box)

        if attack_type == "HA":
            if flag1 and not flag2:
                success += 1
        elif attack_type == "CA":
            if not flag1 and flag2:
                success += 1

        if i == 10:
            t2 = datetime.now()
            pred_time = (t2-t1) * (num-10) / 10
            print("pred time:", pred_time)
        if i == num:
            break
    return success


def _make_boxes(patch: TPatch, pos, model_type):
    s = patch.last_scale
    bbox = [
        pos[1] + (1-s) * patch.w * 0.5,
        pos[0] + (1-s) * patch.h * 0.5,
        pos[1] + (1+s) * patch.w * 0.5,
        pos[0] + (1+s) * patch.h * 0.5,
    ]
    # bbox = [pos[1], pos[0], pos[1]+patch.w, pos[0]+patch.h]
    if model_type == "RCNN":
        _box = torch.tensor([bbox], device=patch.device)
        _label = torch.tensor([patch.target], device=patch.device)
        gt_box = [{"boxes": _box, "labels": _label}]
        dummy_box = [{
            "boxes": torch.empty((0, 4), dtype=torch.float, device=patch.device),
            "labels": torch.empty((0, ), dtype=torch.long, device=patch.device),
        }]
    elif model_type == "YOLO":
        gt_box = torch.tensor([[0, patch.target, *bbox]], dtype=torch.float, device=patch.device)
        gt_box[:, 2:] = xyxy2cxcywh(gt_box[:, 2:]) / 640
        dummy_box = torch.empty((0, 6), device=patch.device)
    else:
        raise NotImplementedError
    return gt_box, dummy_box


def train_AA(patch: TPatch,
             trigger_func,
             p_trs,
             n_trs,
             model: nn.Module,
             loader: DataLoader,
             epoch: int,
             alpha=5,
             beta=0.005,
             ceta=0.1,
             content: str = None,
             save_path: str = None):
    tv_loss = TVLoss()
    if content is not None:
        a = tv.models.vgg19(True).to(patch.device)
        content_loss = ContentLoss(a.features, content, patch.device)
    t1 = datetime.now()
    for j in range(1, epoch + 1):
        log_loss = 0

        for i, img in enumerate(loader, 1):
            if isinstance(img, list) or isinstance(img, tuple):
                img = img[0]
            img = img.to(patch.device)
            h, w = img.shape[-2:]
            pos = patch.random_pos((h, w))

            imgo = patch.apply(img, pos)

            p_tr = random.choice(p_trs)
            n_tr = random.choice(n_trs)
            imgp = trigger_func(imgo, p_tr)
            loss1 = model(imgp, patch.target, tr=True)
            imgn = trigger_func(imgo, n_tr)
            loss2 = model(imgn, patch.target, tr=False)

            loss3 = tv_loss(patch.data)
            if content is not None:
                loss4 = content_loss(patch.data)
            else:
                loss4 = 0.
            loss = loss1 + alpha*loss2 + beta*loss3 + ceta*loss4
            if torch.isnan(loss).any():
                continue
            patch.update(loss)
            log_loss += loss.item()

            if i % 500 == 0:
                print(log_loss)
                log_loss = 0

        if save_path is not None:
            patch.save(save_path)

        if j == 1:
            t2 = datetime.now()
            pred_time = (t2-t1) * (epoch-1)
            print("pred time:", pred_time)
        if j % 2 == 0:
            patch.opt.lr *= 0.5
    return patch


def train_CA(patch: TPatch,
             trigger_func,
             p_trs,
             n_trs,
             detector: nn.Module,
             loader: DataLoader,
             num: int,
             repeat: int,
             model_type: str,
             alpha=5,
             beta=0.005,
             ceta=0.1,
             content: str = None,
             save_path: str = None):
    assert model_type in ["RCNN", "YOLO"], "model should be either RCNN or YOLO"

    detector.train()
    tv_loss = TVLoss()
    if content is not None:
        a = tv.models.vgg19(True).to(patch.device)
        content_loss = ContentLoss(a.features, content, patch.device)
    t1 = datetime.now()
    for i, img in enumerate(loader, 1):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        for j in range(repeat):
            pos = patch.random_pos((h, w))
            gt_box, dummy_box = _make_boxes(patch, pos, model_type)

            imgo = patch.apply(img, pos)
            
            p_tr = random.choice(p_trs)
            n_tr = random.choice(n_trs)
            imgp = trigger_func(imgo, p_tr)
            loss1 = detector(imgp, gt_box)
            imgn = trigger_func(imgo, n_tr)
            loss2 = detector(imgn, dummy_box)
            
            loss3 = tv_loss(patch.data)
            if content is not None:
                loss4 = content_loss(patch.data)
            else:
                loss4 = 0.
            loss = loss1 + alpha*loss2 + beta*loss3 + ceta*loss4
            if torch.isnan(loss).any():
                continue
            patch.update(loss)

        if i == 1:
            t2 = datetime.now()
            pred_time = (t2-t1) * (num-1)
            print("pred time:", pred_time)
        if i % 10 == 0:
            if save_path is not None:
                patch.save(save_path)
            print("tv_loss:", loss3)
        if i == num // 4 * 3:
            patch.opt.lr *= 0.5
        if i == num:
            break
    return patch


def train_HA(patch: TPatch,
             patch2: TPatch,
             instances,
             trigger_func,
             p_trs,
             n_trs,
             detector: nn.Module,
             loader: DataLoader,
             num: int,
             repeat: int,
             model_type: str,
             alpha=5,
             beta=0.005,
             ceta=0.1,
             content: str = None,
             save_path: str = None):
    assert model_type in ["RCNN", "YOLO"], "model should either be RCNN or YOLO"

    if model_type == "RCNN":
        detector.train()
    else:
        detector.eval()
    tv_loss = TVLoss()
    if content is not None:
        a = tv.models.vgg19(True).to(patch.device)
        content_loss = ContentLoss(a.features, content, patch.device)
    t1 = datetime.now()
    for i, img in enumerate(loader, 1):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        for j in range(repeat):
            pos = patch2.random_pos((h, w))
            gt_box, dummy_box = _make_boxes(patch2, pos, model_type)
            
            a = random.randint(0, len(instances)-1)
            patch2.data = instances[a].to(patch.device)
            imgo = patch2.apply(img, pos)
            pos2 = [pos[0] + (patch2.h - patch.h) // 2, pos[1] + (patch2.w - patch.w) // 2]
            imgo = patch.apply(imgo, pos2)
            
            p_tr = random.choice(p_trs)
            n_tr = random.choice(n_trs)
            imgp = trigger_func(imgo, p_tr)
            loss1 = detector(imgp, gt_box, hiding=True)
            imgn = trigger_func(imgo, n_tr)
            loss2 = detector(imgn, gt_box)

            loss3 = tv_loss(patch.data)
            if content is not None:
                loss4 = content_loss(patch.data)
            else:
                loss4 = 0.
            loss = loss1 + alpha*loss2 + beta*loss3 + ceta*loss4
            patch.update(loss)

        if i == 1:
            t2 = datetime.now()
            pred_time = (t2-t1) * (num-1)
            print("pred time:", pred_time)
        if i % 10 == 0:
            if save_path is not None:
                patch.save(save_path)
            print("tv_loss:", loss3)
        if i == num // 4 * 3:
            patch.opt.lr *= 0.5
        if i == num:
            break

    return patch
