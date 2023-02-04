import _init_path
import os
import sys
import torch
import math
import torchvision as tv
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
from PIL import Image
from collections import OrderedDict
from torchvision.models.detection.rpn import concat_box_prediction_layers

__all__ = ["Annotator", "FasterRCNN", "YOLOv3", "YOLOv5", "get_det_model", "_simple_read"]


def _simple_read(path, size=640):
    trans = tv.transforms.Compose([
        tv.transforms.Resize(size),
        tv.transforms.CenterCrop((size, size)),
        tv.transforms.ToTensor(),
    ])
    return trans(Image.open(path))


class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im):
        self.im = im if isinstance(im, np.ndarray) else self.tensor2numpy(im)
        self.im = np.ascontiguousarray(self.im)
        self.lw = max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def tensor2numpy(self, t: torch.Tensor):
        return t.detach().cpu().squeeze().mul(255).permute(
            1, 2, 0).numpy().astype("uint8")[..., ::-1]

    def box_label(self,
                  box,
                  label='',
                  color=(128, 128, 128),
                  txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, self.lw, cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label,
                                   0,
                                   fontScale=self.lw / 3,
                                   thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label,
                        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def save(self, path):
        cv2.imwrite(path, self.im)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = tv.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


class ComputeLoss:
    # Compute losses
    def __init__(self, model):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        self.cp, self.cn = 1., 0.

        det = model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp = BCEcls, BCEobj, 1.0, h
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, reverse=False):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                if not reverse:
                    lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    if not reverse:
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    else:
                        t = torch.full_like(ps[:, 0], self.cn, device=device)  # targets
                        attack_target = int(tcls[i][0])
                        lcls += self.BCEcls(ps[:, 5+attack_target], t) / 80
            if not reverse:
                obji = self.BCEobj(pi[..., 4], tobj)
            else:
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
                obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class YOLOv5(nn.Module):
    def __init__(self, device):
        root = os.getcwd()
        sys.path.insert(0, root)
        from yolov5.models.yolo import attempt_load
        sys.path.remove(root)

        super().__init__()
        self.model = attempt_load(f"weights/yolov5m.pt", map_location=device, fuse=True)
        for k in self.model.model.children():
            if "Detect" in str(type(k)):
                k.inplace = False
        self.conf_thres = 0.25
        self.nms = lambda x: non_max_suppression(x, conf_thres=self.conf_thres)
        self.compute_loss = ComputeLoss(self.model)
        
        sys.path = [x for x in sys.path if "yolov5" not in x]

    def forward(self, x, y=None, reverse=False, hiding=False):
        if y is None:
            if self.training:
                raise NotImplementedError
            else:
                pred = self.model(x)[0]
                ret = self.nms(pred)
            return ret
        elif hiding:
            if self.training:
                raise NotImplementedError
            else:
                pred = self.model(x)[0]
                attack_target = int(y[0, 1])
                scores = pred[..., 4] * pred[..., 5+attack_target]
                loss = -torch.log(1 - scores.max())
            return loss
        else:
            if self.training:
                pred = self.model(x)
            else:
                pred = self.model(x)[1]
            loss = self.compute_loss(pred, y, reverse=reverse)[0]
            return loss


class YOLOv3(nn.Module):
    def __init__(self, device):
        root = os.getcwd()
        sys.path.insert(0, root)
        from yolov3.models.yolo import attempt_load
        sys.path.remove(root)
        
        super().__init__()
        self.model = attempt_load(f"weights/yolov3.pt", map_location=device, fuse=True)
        for k in self.model.model.children():
            if "Detect" in str(type(k)):
                k.inplace = False
        self.conf_thres = 0.25
        self.nms = lambda x: non_max_suppression(x, conf_thres=self.conf_thres)
        self.compute_loss = ComputeLoss(self.model)
        
        sys.path = [x for x in sys.path if "yolov3" not in x]

    def forward(self, x, y=None, reverse=False, hiding=False):
        if y is None:
            if self.training:
                raise NotImplementedError
            else:
                pred = self.model(x)[0]
                ret = self.nms(pred)
            return ret
        elif hiding:
            if self.training:
                raise NotImplementedError
            else:
                pred = self.model(x)[0]
                attack_target = int(y[0, 1])
                scores = pred[..., 4] * pred[..., 5+attack_target]
                loss = -torch.log(1 - scores.max())
            return loss
        else:
            if self.training:
                pred = self.model(x)
            else:
                pred = self.model(x)[1]
            loss = self.compute_loss(pred, y, reverse=reverse)[0]
            return loss


class FasterRCNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                 pretrained_backbone=False,
                                                                 min_size=640,
                                                                 max_size=640)
        self.model.roi_heads.score_thresh = 0.5
        
        self.model.load_state_dict(torch.load("weights/fasterrcnn.pth"))
        self.model.to(device)
        
        self.loss_obj = nn.BCELoss()
        self.loss_cls = nn.CrossEntropyLoss()

    def forward(self, x, y=None, hiding=False):
        if y is None:
            if self.training:
                raise NotImplementedError
            preds = self.model(x)
            ret = []
            for pred in preds:
                boxes = pred["boxes"]
                labels = pred["labels"].unsqueeze(1)
                scores = pred["scores"].unsqueeze(1)
                pred_fmt = torch.cat((boxes, scores, labels), dim=1)
                ret.append(pred_fmt)
            return ret
        elif hiding:
            x, y = self.model.transform(x, y)
            attack_target = y[0]["labels"].item()
            features = self.model.backbone(x.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([('0', features)])

            _features = list(features.values())
            objectness, pred_bbox_deltas = self.model.rpn.head(_features)
            anchors = self.model.rpn.anchor_generator(x, _features)

            num_images = len(anchors)
            num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
            num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
            objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
            proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
            proposals = proposals.view(num_images, -1, 4)
            proposals, scores = self.model.rpn.filter_proposals(proposals, objectness, x.image_sizes, num_anchors_per_level)
            
            box_features = self.model.roi_heads.box_roi_pool(features, proposals, x.image_sizes)
            box_features = self.model.roi_heads.box_head(box_features)
            class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

            confs = F.softmax(class_logits)
            scores = scores[0] * confs[:, attack_target]
            loss = -torch.log(1 - scores.max())
            return loss
        else:
            if not self.training:
                raise NotImplementedError
            losses = self.model(x, y)
            return sum((losses[k] for k in losses))


def get_det_model(device, model_type):
    if model_type == "rcnn":
        return FasterRCNN(device)
    elif model_type == "yolov3":
        return YOLOv3(device)
    elif model_type == "yolov5":
        return YOLOv5(device)
    else:
        raise NotImplementedError

