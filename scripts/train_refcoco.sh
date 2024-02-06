export N=2
export GPUS_PER_NODE=8
export SRUN_ARGS="--quotatype=auto"
export MASTER_PORT=$((60000 + $RANDOM % 1000))

export NPROC_PER_NODE=8

export NNODES=${N}
export PORT=${MASTER_PORT}
export ADDR=${host}


export TRANSFORMERS_OFFLINE=1

export COMMAND="xtuner train xtuner/configs/llava/vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_refcoco.py --deepspeed deepspeed_zero2"

sh scripts/srun.sh llm_razor refcoco \
sh scripts/storchrun.sh 
