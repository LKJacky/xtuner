# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import re
import string
import time
import torch
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    GenerationConfig,
)
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory

from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto="auto"
)
from xtuner.dataset.refcoco_json import RefCOCOJsonEvalDataset
from xtuner.dataset.map_fns import llava_map_fn
from xtuner.model.llava import LLaVAModel
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist


def skip_init():
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


skip_init()


from mmengine.dist import init_dist


def merge_outputs(otuputs):
    new_outputs = [None for _ in range(dist.get_world_size())]

    assert dist.is_initialized()

    dist.all_gather_object(new_outputs, otuputs)
    new_dict = []
    for output in new_outputs:
        new_dict.extend(output)
    return new_dict


@torch.no_grad()
def main():
    init_dist(launcher="pytorch")
    print(f"{dist.get_rank()} / {dist.get_world_size()}")
    device = torch.device(f"cuda:{dist.get_rank()}")

    # load model
    config_path = "xtuner/configs/llava/vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_refcoco.py"

    from xtuner.registry import BUILDER
    from mmengine import Config

    config = Config.fromfile(config_path)
    model: LLaVAModel = BUILDER.build(config.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    state_dict = torch.load(
        "work_dirs/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_refcoco/epoch_1.pth/mp_rank_00_model_states.pt",
        map_location="cpu",
    )["module"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # dataset
    dataset = RefCOCOJsonEvalDataset(
        data_path="data/llava_data/RefCOCOJson/eval_data/refcoco_testA.json",
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
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=False, seed=0),
    )
    loader.sampler.set_epoch(0)

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
    print(len(dataset))
    # generation
    answers = []
    for i, data in enumerate(loader):
        t0 = time.time()
        # prepare inputs
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

        # generation
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
                "bbox": torch.tensor(data["bbox"]).tolist(),
            }
        )
        if i % 100 == 0:
            print(f"{i}/{len(dataset)}: {time.time()-t0}")
    merged_outputs = merge_outputs(answers)
    json.dump(merged_outputs, open("answers.json", "w"))


if __name__ == "__main__":
    main()
