export N=1
export GPUS_PER_NODE=8
export SRUN_ARGS="--quotatype=auto"
export MASTER_PORT=$((60000 + $RANDOM % 1000))

export NPROC_PER_NODE=8

sh scripts/srun.sh llm_razor refcoco \
torchrun --master-port ${MASTER_PORT} --nproc_per_node 8 eval_refcoco.py \
lmsys/vicuna-7b-v1.5 \
--visual-encoder openai/clip-vit-large-patch14-336 \
--llava xtuner/llava-v1.5-7b-xtuner \
--prompt-template internlm_chat \
--work-dir ./tmp