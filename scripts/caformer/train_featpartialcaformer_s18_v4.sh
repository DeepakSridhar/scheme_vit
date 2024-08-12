DATA_PATH=/data8/imgDB/DB/ILSVRC/2012/
CODE_PATH=/home/deepak/featattentionvit/ # modify code path here

MODEL=featpartialcaformer_s18
OUT_PATH=/data8/deepak/poolformer/output/train/
NAME=featpartialcaformer_s18_8gpus_bs512_lr_4em3

ALL_BATCH_SIZE=512
NUM_GPU=8
GRAD_ACCUM_STEPS=1 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


#while true; do \
cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt lamb --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 --head-dropout 0.0 \
--output $OUT_PATH --experiment $NAME #&& break; done