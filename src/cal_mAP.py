import _init_path
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from detmetrics.BoundingBox import BoundingBox
from detmetrics.BoundingBoxes import BoundingBoxes
from detmetrics.Evaluator import Evaluator
from detmetrics.utils import *

folders = [x for x in glob("mAP/yolov5*") if os.path.isdir(x)]
for model in folders:
    print(model)
    if os.path.exists(model+".log"):
        continue
    
    gt_files = sorted(glob("mAP/gt/*.txt"))
    dt_files = sorted(glob(f"{model}/*.txt"))
    gt_files = gt_files[:len(dt_files)]

    bboxes = BoundingBoxes()
    for f in gt_files:
        name = os.path.split(f)[-1].replace(".txt", "")
        with open(f, "r") as fp:
            lines = fp.readlines()
        lines = [x.strip("\n").split() for x in lines]
        for line in lines:
            bbox = BoundingBox(name,
                            line[0],
                            *[float(line[i]) for i in range(1, 5)],
                            CoordinatesType.Absolute, (640, 640),
                            BBType.GroundTruth,
                            format=BBFormat.XYX2Y2)
            bboxes.addBoundingBox(bbox)

    for f in dt_files:
        name = os.path.split(f)[-1].replace(".txt", "")
        with open(f, "r") as fp:
            lines = fp.readlines()
        lines = [x.strip("\n").split() for x in lines]
        for line in lines:
            bbox = BoundingBox(name,
                            line[0],
                            *[float(line[i]) for i in range(2, 6)],
                            CoordinatesType.Absolute, (640, 640),
                            BBType.Detected,
                            float(line[1]),
                            format=BBFormat.XYX2Y2)
            bboxes.addBoundingBox(bbox)
    
    mAPs = []
    iou_threses = np.linspace(0.5, 0.95, 10, endpoint=True)
    for iou_thres in tqdm(iou_threses, ncols=80):
        evaluator = Evaluator()
        metricsPerClass = evaluator.GetPascalVOCMetrics(
            bboxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=iou_thres,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
        mAP = 0
        for mc in metricsPerClass:
            average_precision = mc['AP']
            mAP += average_precision
        mAP /= len(metricsPerClass)
        mAPs.append(mAP)
    with open(model+".log", "w") as f:
        f.write(f"mAP[0.50:0.95:0.05] = {sum(mAPs)/len(mAPs):.3f}\n")
        for mAP, iou_thres in zip(mAPs, iou_threses):
            f.write(f"mAP[{iou_thres:.2f}] = {mAP:.3f}\n")

