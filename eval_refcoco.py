# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import re
import string
import time

import numpy as np
import pandas as pd
import torch
import tqdm
from huggingface_hub import snapshot_download
from mmengine import mkdir_or_exist
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
    GenerationConfig,
)
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory

from xtuner.dataset.utils import decode_base64_to_image, expand2square
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto="auto"
)
from xtuner.dataset.refcoco_json import RefCOCOJsonEvalDataset
from xtuner.dataset.map_fns import llava_map_fn
from xtuner.model.llava import LLaVAModel
from torch.utils.data import DataLoader


def skip_init():
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


skip_init()

device = torch.device("cuda:0")


@torch.no_grad()
def main():
    # load model
    config_path = "xtuner/configs/llava/vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_refcoco.py"

    from xtuner.registry import BUILDER
    from mmengine import Config

    config = Config.fromfile(config_path)
    model: LLaVAModel = BUILDER.build(config.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

    # dataset
    dataset = RefCOCOJsonEvalDataset(
        data_path="data/refcoco/refcoco_annotations/eval_data/refcoco_testA.json",
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
    loader = DataLoader(dataset, batch_size=1)

    # generation config
    gen_config = GenerationConfig(
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )
    stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=["</s>"])

    # generation
    answers = []
    for data in loader:
        visual_outputs = model.visual_encoder(
            data["pixel_values"].to(device), output_hidden_states=True
        )
        pixel_values = model.projector(
            visual_outputs.hidden_states[model.visual_select_layer][:, 1:]
        )
        data["pixel_values"] = pixel_values
        data["input_ids"] = torch.tensor(data["input_ids"]).reshape([1, -1])
        datax = prepare_inputs_labels_for_multimodal(
            llm=model.llm.to(device),
            input_ids=data["input_ids"].to(device),
            pixel_values=data["pixel_values"].to(device),
        )

        generation = model.llm.generate(
            **datax,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria,
        )
        ans = tokenizer.decode(generation[0])
        answers.append(
            {
                "ans": ans,
                "id": data["id"][0],
                "bbox": data["bbox"].tolist(),
            }
        )
        json.dump(answers, open("answers.json", "w"))
        break


if __name__ == "__main__":
    main()
