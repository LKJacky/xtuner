# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import logging
import os
from unittest import TestCase

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.dataset import InvRefCOCOTrainDataset, RefCOCOTrainDataset
from xtuner.dataset.map_fns import refcoco_map_fn
from xtuner.dataset.refcoco import Blip2ImageTrainProcessor
from xtuner.registry import BUILDER
from xtuner.dataset.refcoco_json import RefCOCOJsonDataset

PATH = "xtuner/configs/llava/vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_finetune.py"


class TestRefCOCOJson(TestCase):
    def test_generate(self):
        save = False
        save_path = "data/llava_data/RefCOCOJson"

        data = RefCOCOJsonDataset.get_data_json(
            ann_path="data/refcoco/refcoco_annotations",
            image_path="data/llava_data/llava_images",
        )[0]
        print(data[0])

        if save:
            with open(save_path + "/train.json", "w") as f:
                json.dump(data, f, indent=4)


class TestRef(TestCase):
    def test_ref(self):
        config_path = PATH
        dataset_config = Config.fromfile(config_path)["llava_dataset"]

        refcoco_dataset_config = copy.copy(dataset_config)
        refcoco_dataset_config["type"] = RefCOCOTrainDataset
        refcoco_dataset_config["data_path"] = "data/refcoco/refcoco_annotations"
        refcoco_dataset_config["image_folder"] = "data/refcoco/train2014"
        refcoco_dataset_config["dataset_map_fn"] = refcoco_map_fn
        # refcoco_dataset_config["processor"] = dict(type=Blip2ImageTrainProcessor)
        refcoco_set = BUILDER.build(refcoco_dataset_config)
        item = refcoco_set[0]
        self._print(item)
        print(len(refcoco_set))

    def test_inv_ref(self):
        config_path = PATH
        dataset_config = Config.fromfile(config_path)["llava_dataset"]

        refcoco_dataset_config = copy.copy(dataset_config)
        refcoco_dataset_config["type"] = InvRefCOCOTrainDataset
        refcoco_dataset_config["data_path"] = "data/refcoco/refcoco_annotations"
        refcoco_dataset_config["image_folder"] = "data/refcoco/train2014"
        refcoco_dataset_config["dataset_map_fn"] = refcoco_map_fn

        refcoco_set = BUILDER.build(refcoco_dataset_config)
        item = refcoco_set[0]
        self._print(item)

    def test_llava(self):
        config_path = PATH
        dataset_config = Config.fromfile(config_path)["llava_dataset"]

        dataset = BUILDER.build(dataset_config)
        item = dataset[0]
        self._print(item)

    def _print(self, item):
        for key in item:
            value = item[key]
            if isinstance(value, torch.Tensor):
                print(f"{key}\n{value.shape}")
            elif isinstance(value, list):
                print(f"{key}\n{value}\n{len(value)}")
            else:
                print(f"{key}\n{value}")
            print()
