import json
import os
import cv2

coco_img = "dataset/mscoco/val2014"
coco_ann = "dataset/mscoco/annotations/instances_val2014.json"
out_dir = "dataset/mscoco/crops"

with open(coco_ann, "r") as f:
    dt = json.load(f)

class_of_interest = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
]

out_dirs = []
for c in class_of_interest:
    n = os.path.join(out_dir, c.replace(" ", "-"))
    out_dirs.append(n)
    if not os.path.exists(n):
        os.makedirs(n)

ratios = [
    (0.495, 0.505),
    (0.40, 0.60),
    (0.90, 1.10),
    (0.40, 0.60),
    (0.90, 1.10),
    (0.90, 1.10),
    (0.40, 0.60),
    (0.90, 1.10),
]

n = len(class_of_interest)
name2idx = {k["name"]:k["id"] for k in dt["categories"]}
idx2name = {v:k for k, v in name2idx.items()}

for i in range(n):
    out_dir = out_dirs[i]
    target = class_of_interest[i]
    min_bound, max_bound = ratios[i]
    target_id = name2idx[target]
    cnt = 0
    for instance in dt["annotations"]:
        if instance["category_id"] == target_id:
            j = instance["image_id"]
            r = instance["bbox"][2] / instance["bbox"][3]
            s = (instance["bbox"][2] * instance["bbox"][3])**0.5
            ori_img = os.path.join(coco_img, f"COCO_val2014_{j:012d}.jpg")
            if min_bound <= r <= max_bound and s > 100:
                crop_img = os.path.join(out_dir, f"{cnt:06d}.png")
                cnt += 1
                img = cv2.imread(ori_img)
                h, w = img.shape[:2]
                bbox = instance["bbox"]
                a = min(max(int(bbox[1]), 0), h-1)
                b = min(max(int(bbox[1]+bbox[3]), 0), h-1)
                c = min(max(int(bbox[0]), 0), w-1)
                d = min(max(int(bbox[0]+bbox[2]), 0), w-1)
                crop = img[a:b, c:d]
                cv2.imwrite(crop_img, crop)


