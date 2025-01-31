# 数据集准备

- [HuggingFace 数据集](#huggingface-数据集)
- [其他](#其他)
  - [Arxiv Gentitle 生成题目](#arxiv-gentitle-生成题目)
  - [MOSS-003-SFT](#moss-003-sft)
  - [Chinese Lawyer](#chinese-lawyer)

## HuggingFace 数据集

针对 HuggingFace Hub 中的数据集，比如 [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)，用户可以快速使用它们。更多使用指南请参照[单轮对话文档](./single_turn_conversation.md)和[多轮对话文档](./multi_turn_conversation.md)。

## 其他

### Arxiv Gentitle 生成题目

Arxiv 数据集并未在 HuggingFace Hub上发布，但是可以在 Kaggle 上下载。

**步骤 0**，从 https://kaggle.com/datasets/Cornell-University/arxiv 下载原始数据。

**步骤 1**，使用 `xtuner preprocess arxiv ${DOWNLOADED_DATA} ${SAVE_DATA_PATH} [optional arguments]` 命令处理数据。

例如，提取从 `2020-01-01` 起的所有 `cs.AI`、`cs.CL`、`cs.CV` 论文：

```shell
xtuner preprocess arxiv ${DOWNLOADED_DATA} ${SAVE_DATA_PATH} --categories cs.AI cs.CL cs.CV --start-date 2020-01-01
```

**步骤 2**，所有的 Arixv Gentitle 配置文件都假设数据集路径为 `./data/arxiv_data.json`。用户可以移动并重命名数据，或者在配置文件中重新设置数据路径。

### MOSS-003-SFT

MOSS-003-SFT 数据集可以在 https://huggingface.co/datasets/fnlp/moss-003-sft-data 下载。

**步骤 0**，下载数据。

```shell
# 确保已经安装 git-lfs (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/fnlp/moss-003-sft-data
```

**步骤 1**，解压缩。

```shell
cd moss-003-sft-data
unzip moss-003-sft-no-tools.jsonl.zip
unzip moss-003-sft-with-tools-no-text2image.zip
```

**步骤 2**, 所有的 moss-003-sft 配置文件都假设数据集路径为 `./data/moss-003-sft-no-tools.jsonl` 和 `./data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl`。用户可以移动并重命名数据，或者在配置文件中重新设置数据路径。

### Chinese Lawyer

Chinese Lawyer 数据集有两个子数据集，它们可以在 https://github.com/LiuHC0428/LAW-GPT 下载。

所有的 Chinese Lawyer 配置文件都假设数据集路径为 `./data/CrimeKgAssitant清洗后_52k.json` 和 `./data/训练数据_带法律依据_92k.json`。用户可以移动并重命名数据，或者在配置文件中重新设置数据路径。
