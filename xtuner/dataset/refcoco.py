import os
import json
import pickle
import random
import time
import itertools

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
from .huggingface import process_hf_dataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import re
from mmengine.config import Config, ConfigDict
from ..registry import BUILDER


class RefCOCOTrainDataset(Dataset):
    def __init__(
        self,
        data_path,  # path to refcoco path
        image_folder,
        tokenizer,
        image_processor,
        dataset="refcoco",
        splitBy="unc",
        max_dataset_length=None,
        dataset_map_fn=None,
        template_map_fn=None,
        max_length=2048,
        pad_image_to_square=False,
    ):
        self.vis_root = image_folder
        self.text_processor = BlipCaptionProcessor(max_words=50)

        if (
            isinstance(image_processor, dict)
            or isinstance(image_processor, Config)
            or isinstance(image_processor, ConfigDict)
        ):
            self.processor = BUILDER.build(image_processor)
        else:
            self.processor = image_processor

        self.refer = REFER(data_path, image_folder, dataset, splitBy)
        self.ref_ids = self.refer.getRefIds(split="train")

        self.instruction_pool = [
            "[refer] {}",
            "[refer] give me the location of {}",
            "[refer] where is {} ?",
            "[refer] from this image, tell me the location of {}",
            "[refer] the location of {} is",
            "[refer] could you tell me the location for {} ?",
            "[refer] where can I locate the {} ?",
        ]

        def refcoco_prepare_hf(data):
            json_data = DatasetDict({"train": HFDataset.from_list([data])})

            data_set = process_hf_dataset(
                dataset=json_data,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split="train",
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True,
            )
            return data_set[0]

        self.prepare_hf_datasets = refcoco_prepare_hf

    def preprocess(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        image_file = "COCO_train2014_{:0>12}.jpg".format(ref["image_id"])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image_orig_size = image.size
        image = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_new_size = [image.shape[1], image.shape[2]]

        image_new_size = [100, 100]

        sample_sentence = random.choice(ref["sentences"])["raw"]
        refer_sentence = self.text_processor(sample_sentence)

        bbox = self.refer.getRefBox(ref["ref_id"])
        bbox = [
            bbox[0] / image_orig_size[0] * image_new_size[0],
            bbox[1] / image_orig_size[1] * image_new_size[1],
            (bbox[0] + bbox[2]) / image_orig_size[0] * image_new_size[0],
            (bbox[1] + bbox[3]) / image_orig_size[1] * image_new_size[1],
        ]
        bbox = [int(x) for x in bbox]
        bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
        return {
            "image": image,
            "refer_sentence": refer_sentence,
            "bbox": bbox,
            "image_id": ref["image_id"],
        }

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        data = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(
            data["refer_sentence"]
        )

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        image = data.pop("image")
        data = {
            "instruction_input": instruction,
            "answer": data["bbox"],
            "image_id": data["image_id"],
        }

        data = self.prepare_hf_datasets(data)
        data["pixel_values"] = image
        data.pop("instruction_input")
        data.pop("answer")
        return data


class InvRefCOCOTrainDataset(RefCOCOTrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instruction_pool = [
            "[identify] {}",
            "[identify] what object is in this location {}",
            "[identify] identify the object present at this location {}",
            "[identify] what is it in {}",
            "[identify] describe this object in {}",
            "[identify] this {} is",
            "[identify] the object in {} is",
        ]

    def __getitem__(self, index):
        data = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(data["bbox"])

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        image = data.pop("image")
        data = {
            "instruction_input": instruction,
            "answer": self.text_processor(data["refer_sentence"]),
            "image_id": data["image_id"],
        }

        data = self.prepare_hf_datasets(data)
        data["pixel_values"] = image
        return data


def fake_processor(input):
    return input


# below codes are copied form minigpt4: https://github.com/Vision-CAIR/MiniGPT-4


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()


class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(
        self,
        item,
    ):
        return self.transform(item)

    def preprocess(self, item, **kwargs):
        res = self.transform(item)
        return {"pixel_values": [res]}


class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption

