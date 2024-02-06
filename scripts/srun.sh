#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-24}
N=${N:-1}


SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:3}


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${N} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    -N ${N} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    $PY_ARGS