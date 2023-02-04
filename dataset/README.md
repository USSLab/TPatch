# Dataset

## Folder Tree

```bash
dataset
├── ImageNet
│   ├── 000
│   ├── 001
│   ├── ...
│   └── 999
├── mscoco
│   ├── annotations
│   ├── crops
│   │   ├── bicycle
│   │   ├── bus
│   │   ├── car
│   │   ├── motorcycle
│   │   ├── person
│   │   ├── stop-sign
│   │   ├── traffic-light
│   │   ├── trash
│   │   └── truck
│   └── val2014
└── KITTI
    └── mask_samples
```

## Downloads

- dataset/ImageNet (ImageNet validation set)
- dataset/mscoco/val2014 [MS COCO val2014](http://images.cocodataset.org/zips/val2014.zip)
- dataset/mscoco/annotations [MS COCO annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- dataset/KITTI/mask_samples [Background Images](https://drive.google.com/drive/folders/1EocHrcuZ70-tOs6kIxLyY8Zr4tCW-Kas?usp=sharing)
- dataset/mscoco/crops (Please run the script `src/crop_bg.py` to generate the background images for evaluation of Hiding Attack.)
