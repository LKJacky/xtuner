
host=$(scontrol show hostname ${SLURM_NODELIST} | head -n1)
# echo ${SLURM_PROCID}
echo ${host}

export NODE_RANK=${SLURM_PROCID}
export ADDR=${host}

${COMMAND}