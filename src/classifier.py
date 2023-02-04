import _init_path
import torch
import torchvision as tv
from torch import nn
from tpatch import load_imagenet_preprocess


__all__ = ["BaseClassifier", "EnsembleModel", "get_cls_model", "get_cls_ens_model"]


class BaseClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.compute_conf = nn.Softmax()
        self.preprocess = load_imagenet_preprocess()

    def forward(self, x, y=None, tr=True):
        logits = self.model(self.preprocess(x))
        conf = self.compute_conf(logits)
        if y is None:
            idx = conf.max(dim=1)[1]
            return idx
        elif tr:
            loss = -torch.log(conf[:, y] + 1e-12).mean()
            return loss
        else:
            loss = -torch.log(1 - conf[:, y] + 1e-12).mean()
            return loss

    def top_five(self, x):
        logits = self.model(self.preprocess(x))
        conf = self.compute_conf(logits)
        return torch.topk(conf, 5)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.n = len(models)
    
    def forward(self, x, y=None, tr=True):
        loss = 0
        for model in self.models:
            loss += model(x, y=y, tr=tr)
        return loss / self.n


def get_cls_model(device, model_type) -> BaseClassifier:
    if model_type == "vgg13":
        model = tv.models.vgg13_bn(pretrained=False)
        model.load_state_dict(torch.load("weights/vgg13.pth"))
    elif model_type == "vgg16":
        model = tv.models.vgg16_bn(pretrained=False)
        model.load_state_dict(torch.load("weights/vgg16.pth"))
    elif model_type == "vgg19":
        model = tv.models.vgg19_bn(pretrained=False)
        model.load_state_dict(torch.load("weights/vgg19.pth"))
    elif model_type == "res50":
        model = tv.models.resnet50(pretrained=False)
        model.load_state_dict(torch.load("weights/resnet50.pth"))
    elif model_type == "res101":
        model = tv.models.resnet101(pretrained=False)
        model.load_state_dict(torch.load("weights/resnet101.pth"))
    elif model_type == "res152":
        model = tv.models.resnet152(pretrained=False)
        model.load_state_dict(torch.load("weights/resnet152.pth"))
    elif model_type == "incv3":
        model = tv.models.inception_v3(pretrained=False)
        model.load_state_dict(torch.load("weights/incv3.pth"))
    elif model_type == "mobv2":
        model = tv.models.mobilenet_v2(pretrained=False)
        model.load_state_dict(torch.load("weights/mobv2.pth"))
    else:
        raise NotImplementedError
    model.eval().to(device)
    return BaseClassifier(model)


def get_cls_ens_model(device, model_types):
    models = [get_cls_model(device, model_type) for model_type in model_types]
    return EnsembleModel(models)
