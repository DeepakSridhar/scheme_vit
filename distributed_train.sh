#!/bin/bash
NUM_PROC=$1
minimum=2000
maximum=4000
#Generate the random number
randomNumber=$(($minimum + $RANDOM % $maximum))
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$randomNumber train.py "$@"

## Using torchrun to distribute over multi-nodes
# torchrun \
# --nproc_per_node=$NUM_PROC \
# --nnodes=2 \
# --node_rank=0 \
# --rdzv_id=456 \
# --rdzv_backend=c10d \
# --rdzv_endpoint=172.31.43.139:8888 \
# train.py "$@"
