# Scheme MLP Mixer


## Installation
Install anaconda on your server and then install conda environment using the provided yml file in other branch
```bash
conda create -n python=3.8
```

## Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; fvcore; [timm](https://github.com/rwightman/pytorch-image-models) (`pip install timm==0.6.11`)

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## Inference Speed
To evaluate the inference speed of the models

```bash
python run_benchmark.py 
```

## Validation

To evaluate CAFormer-S18 models, run:

```bash
MODEL=schemeformer_ppaa_s12_224
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --checkpoint /path/to/checkpoint 
```



### Train
To train models on 8 GPUs. The relation between learning rate and batch size is lr=bs/1024*1e-3.
For convenience, assuming the batch size is 1024, then the learning rate is set as 1e-3 (for batch size of 1024, setting the learning rate as 2e-3 sometimes sees better performance). 

To use bash script on single node via command line
```bash
DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/ # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model featcaformer_s18 --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.2 --head-dropout 0.0
```

Alternatively, use the following command to run on multi-node by changing the options in the bash script
```bash
bash scripts/caformer/train_featcaformer_s18.sh
```
```bash
DATA_PATH=/path/to/imagenet/data/  # absolute path
CODE_PATH=/absolute/path/to/code/ # modify code path here

OUT_PATH=/absolute/path/to/output/

ALL_BATCH_SIZE=1024 #Batch Size across all gpus
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
```

**For multi-node training comment the distributed.launch command and uncomment the torchrun command with appropriate options set**


Training scripts of other models are shown in [scripts](/scripts/).
