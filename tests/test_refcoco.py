# Copyright (c) OpenMMLab. All rights reserved.
import json

from unittest import TestCase

import torch
import os


from xtuner.dataset.refcoco_json import (
    RefCOCOJsonDataset,
    RefCOCOJsonEvalDataset,
    InvRefCOCOJsonDataset,
)
from xtuner.dataset import LLaVADataset
from transformers import AutoTokenizer, CLIPImageProcessor
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.model.llava import LLaVAModel
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from transformers import AutoModelForCausalLM, CLIPVisionModel
from torch.utils.data import DataLoader
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE

from torch.cuda.amp import autocast


def skip_init():
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


skip_init()
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class TestRefCOCOJson(TestCase):
    @classmethod
    def load_refcoco_dataset(
        self,
        data_path="data/llava_data/LLaVA-Instruct-150K/complex_reasoning_77k.json",
        data_type=LLaVADataset,
        tokenizer=None,
        image_size=336,
        **kwargs,
    ):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        dataset = data_type(
            data_path=data_path,
            image_folder="data/llava_data/llava_images",
            tokenizer=tokenizer,
            image_processor=CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336", crop_size=[image_size, image_size]
            ),
            dataset_map_fn=llava_map_fn,
            template_map_fn=dict(
                type=template_map_fn_factory, template=PROMPT_TEMPLATE.vicuna
            ),
            max_length=2048,
            pad_image_to_square=False,
            **kwargs,
        )
        return dataset

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
        save_path = "data/llava_data/RefCOCOJson/"
        data_info = [
            ("refcoco", "unc"),
            ("refcoco+", "unc"),
            ("refcocog", "umd"),
        ]
        all_data = []
        for data_type, split in data_info:
            data = RefCOCOJsonDataset.get_data_json(
                ann_path="data/refcoco/refcoco_annotations",
                image_path="data/llava_data/llava_images",
                dataset=data_type,
                splitBy=split,
            )[0]
            all_data.extend(data)
        print(all_data[0])
        print(len(all_data))

        if save:
            with open(save_path + "/train.json", "w") as f:
                json.dump(all_data, f, indent=4)

    def test_llava_dataset(self):
        dataset = self.load_refcoco_dataset(
            data_path="data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
            data_type=LLaVADataset,
            max_dataset_length=2048,
        )
        print(len(dataset))

    def test_data_load(self):
        dataset = self.load_refcoco_dataset(
            data_path="data/llava_data/RefCOCOJson/train.json",
            data_type=RefCOCOJsonDataset,
            max_dataset_length=10,
        )
        self._print(dataset[0])
        print(len(dataset))

    def test_data_load_eval(self):
        dataset = self.load_refcoco_dataset(
            data_path="data/refcoco/refcoco_annotations/eval_data/refcoco_testA.json",
            data_type=RefCOCOJsonEvalDataset,
        )
        import time

        self._print(dataset[0])

        for i in range(100):
            t0 = time.time()
            a = dataset[i]
            # self._print(dataset[i])
            print(time.time() - t0)
        # loader = DataLoader(dataset, batch_size=1)
        # self._print(loader.__iter__().__next__())

        # print(len(dataset))
        # print(len(loader))

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


class TestModel(TestCase):
    @torch.no_grad()
    def test_build_model(self):
        config_path = "xtuner/configs/llava/vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_refcoco.py"

        from xtuner.registry import BUILDER
        from mmengine import Config

        config = Config.fromfile(config_path)
        model = BUILDER.build(config.model)
        print(model)

    @torch.no_grad()
    def test_generation(self):
        device = torch.device("cuda:0")
        model = LLaVAModel(
            llm=AutoModelForCausalLM.from_pretrained(
                "lmsys/vicuna-7b-v1.5",
                torch_dtype=torch.float16,
            ),
            visual_encoder=CLIPVisionModel.from_pretrained(
                "openai/clip-vit-large-patch14-336",
                torch_dtype=torch.float16,
            ),
        )
        model = model.to(device)

        dataset = TestRefCOCOJson.load_refcoco_dataset(
            data_path="data/refcoco/refcoco_annotations/eval_data/refcoco_testA.json",
            data_type=RefCOCOJsonEvalDataset,
        )

        dataloader = DataLoader(dataset, batch_size=1)
        data = dataloader.__iter__().__next__()
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

        with autocast():

            inputs = data["conversation"][0]["input"][0]
            chunk_encode = []
            for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                if idx == 0:
                    cur_encode = tokenizer.encode(chunk)
                else:
                    cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
                chunk_encode.append(cur_encode)
            assert len(chunk_encode) == 2
            ids = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                ids.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    ids.append(IMAGE_TOKEN_INDEX)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            visual_outputs = model.visual_encoder(
                data["pixel_values"].to(device), output_hidden_states=True
            )
            pixel_values = model.projector(
                visual_outputs.hidden_states[model.visual_select_layer][:, 1:]
            )
            data["pixel_values"] = pixel_values
            data["input_ids"] = ids
            datax = prepare_inputs_labels_for_multimodal(
                llm=model.llm.to(device),
                input_ids=data["input_ids"].to(device),
                pixel_values=data["pixel_values"].to(device),
            )
            generation = model.llm.generate(
                **datax,
                max_new_tokens=10,
                do_sample=False,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        print(tokenizer.decode(generation[0]))
