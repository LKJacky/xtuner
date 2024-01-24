from typing import Any
import torch
from .refcoco import REFER
from collections import defaultdict
from PIL import Image
import json
import copy

from .llava import LLaVADataset
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict
from PIL import Image

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
import copy


class RefCOCOJsonDataset(LLaVADataset):
    instruction_pool = [
        "[refer] {}",
        "[refer] give me the location of {}",
        "[refer] where is {} ?",
        "[refer] from this image, tell me the location of {}",
        "[refer] the location of {} is",
        "[refer] could you tell me the location for {} ?",
        "[refer] where can I locate the {} ?",
    ]

    def __init__(
        self,
        data_path,
        image_folder,
        tokenizer,
        image_processor,
        max_dataset_length=None,
        dataset_map_fn=None,
        template_map_fn=None,
        max_length=2048,
        pad_image_to_square=False,
    ):
        json_data = json.load(open(data_path))

        ######################################################
        # Only this part is different from LLaVADataset.__init__
        json_data = self.reformat_data(json_data)
        ######################################################

        for idx in range(len(json_data)):
            if isinstance(json_data[idx]["id"], int):
                json_data[idx]["id"] = str(json_data[idx]["id"])
        json_data = DatasetDict({"train": HFDataset.from_list(json_data)})
        self.text_data = process_hf_dataset(
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

        self.image_folder = image_folder
        if (
            isinstance(image_processor, dict)
            or isinstance(image_processor, Config)
            or isinstance(image_processor, ConfigDict)
        ):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square

    def reformat_data(self, json_data):
        new_json_data = []
        for sample in json_data:
            for instruction_template in self.instruction_pool:
                sample["conversations"] = self.gen_refcoco_conversations(
                    sample, instruction_template
                )
                new_json_data.append(copy.deepcopy(sample))
        return new_json_data

    @classmethod
    def gen_refcoco_conversations(cls, data, instruction_template="{}"):
        """
        build conversition data from refcoco json data as below

        "id": "xxx",
        "image": "xxx.jpg",
        "conversations": [
        {
            "from": "human",
            "value": "xxxx"
        },
        {
            "from": "gpt",
            "value": "xxx"
        }
        """

        conversation = [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": ""},
        ]

        instruction = instruction_template.format(data["sents"])
        answer = "{{<{}><{}><{}><{}>}}".format(
            data["bbox"][0], data["bbox"][1], data["bbox"][2], data["bbox"][3]
        )
        conversation[0]["value"] = instruction + "\n<image>"
        conversation[1]["value"] = answer
        return conversation

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
