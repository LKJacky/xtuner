from typing import Any
import torch
from .refcoco import REFER
from collections import defaultdict
from PIL import Image
import json
import random
import copy
from functools import partial

from torch.utils.data import Dataset


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(
        0, intersection_y2 - intersection_y1 + 1
    )
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


# dataset


class RefCOCOJsonDataset(Dataset):
    """
    "id": data_id,
    "img_id": image_name,
    "sents": sent,
    "bbox": bbox,
    "height": image.height,
    "width": image.width,
    ##
    'instruction_input': ins,
    'answer': ans,
    """

    def __init__(
        self,
        ann_path,
        image_path,
        vis_processor=None,
        reformatter_args={"type": "train_d1"},
    ):
        self.reformatter_map = {
            "train_d1": train_d1,
            "train_c1": train_c1,
            "base_eval": base_eval,
            "train_d2": train_d2,
            "train_c1_construct": train_c1_construct,
        }
        data: dict = json.load(open(ann_path))
        if isinstance(data, dict):
            data = list(data.values())
        self.data = data
        if reformatter_args is not None:
            self.data = self.build_reforder(reformatter_args)(self.data)

        self.image_path = image_path
        self.vis_processor = vis_processor

    def build_reforder(self, reformatter_args):
        assert "type" in reformatter_args, f"{reformatter_args}"
        type = reformatter_args.pop("type")
        return self.reformatter_map[type](**reformatter_args)

    def __getitem__(self, index):
        sample = copy.copy(self.data[index])
        image = self.load_image(sample["img_id"])

        sample["bbox"] = torch.tensor(sample["bbox"])
        sample.update(
            {
                "image": image,
            }
        )
        return sample

    def __len__(self):
        return len(self.data)

    def load_image(self, image_id):
        image = Image.open(self.image_path + "/" + image_id + ".jpg").convert("RGB")
        image = self.vis_processor(image)
        return image

    @classmethod
    def get_data_json(
        cls,
        ann_path,
        image_path,
        dataset="refcoco",
        splitBy="unc",
    ):
        def normalize_bbox(bbox, height, width):
            x, y, w, h = bbox

            bbox = [x / width, y / height, (x + w) / width, (y + h) / height]
            bbox = [int(x * 100) for x in bbox]
            return bbox

        refer = REFER(ann_path, image_path, dataset, splitBy)
        ref_ids = refer.getRefIds(split="train")

        data = {}
        duplicate_data = defaultdict(list)

        for ref_id in ref_ids:
            ref = refer.loadRefs(ref_id)[0]

            image_id = "{:0>12}".format(ref["image_id"])
            sents = [sent["raw"] for sent in ref["sentences"]]
            bbox = refer.getRefBox(ref["ref_id"])

            image = Image.open(image_path + "/" + image_id + ".jpg")
            bbox = normalize_bbox(bbox, image.height, image.width)

            for sent in sents:
                sent_id = "_".join(sent.split(" "))
                data_id = f"{dataset}-{splitBy}-{image_id}-{sent_id}"
                data_item = {
                    "id": data_id,
                    "img_id": image_id,
                    "sents": sent,
                    "bbox": bbox,
                }
                if data_id in data:
                    duplicate_data[data_id].append(data_item)
                else:
                    data[data_id] = data_item

        return list(data.values()), list(duplicate_data.values())
