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
from xtuner.dataset import LLaVADataset
import random
from transformers import AutoTokenizer, CLIPImageProcessor
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory

PATH = "xtuner/configs/llava/vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_finetune.py"


class TestRefCOCOJson(TestCase):
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

    def test_llava_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        dataset = LLaVADataset(
            data_path="data/llava_data/LLaVA-Instruct-150K/complex_reasoning_77k.json",
            image_folder="data/llava_data/llava_images",
            tokenizer=tokenizer,
            image_processor=CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            ),
            max_dataset_length=None,
            dataset_map_fn=llava_map_fn,
            template_map_fn=dict(
                type=template_map_fn_factory, template=PROMPT_TEMPLATE.vicuna
            ),
            max_length=2048,
            pad_image_to_square=False,
        )
        self._print(dataset[0])

    def test_data_load(self):
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        dataset = RefCOCOJsonDataset(
            data_path="data/llava_data/RefCOCOJson/train.json",
            image_folder="data/llava_data/llava_images",
            tokenizer=tokenizer,
            image_processor=CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            ),
            max_dataset_length=None,
            dataset_map_fn=llava_map_fn,
            template_map_fn=dict(
                type=template_map_fn_factory, template=PROMPT_TEMPLATE.vicuna
            ),
            max_length=2048,
            pad_image_to_square=False,
        )
        self._print(dataset[0])
