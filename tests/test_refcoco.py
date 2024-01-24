# Copyright (c) OpenMMLab. All rights reserved.
import json

from unittest import TestCase

import torch


from xtuner.dataset.refcoco_json import (
    RefCOCOJsonDataset,
    RefCOCOJsonEvalDataset,
    InvRefCOCOJsonDataset,
)
from xtuner.dataset import LLaVADataset
from transformers import AutoTokenizer, CLIPImageProcessor
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory


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

    def test_data_load_eval(self):
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        dataset = RefCOCOJsonEvalDataset(
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

    def test_data_load_inv(self):
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        dataset = InvRefCOCOJsonDataset(
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
