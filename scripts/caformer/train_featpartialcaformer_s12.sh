DATA_PATH=/data8/imgDB/DB/ILSVRC/2012/
CODE_PATH=/home/deepak/featattentionvit/ # modify code path here

MODEL=featpartialcaformer_s12
OUT_PATH=/data8/deepak/poolformer/output/train/

ALL_BATCH_SIZE=4096
NUM_GPU=16
GRAD_ACCUM_STEPS=1 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


#while true; do \
cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt lamb --lr 8e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 --head-dropout 0.0 \
--output $OUT_PATH --experiment $MODEL #&& break; done