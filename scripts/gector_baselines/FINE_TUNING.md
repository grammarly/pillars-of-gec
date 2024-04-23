# GECTOR fine-tuning approach

### 1. Install gector-large models. 

#### gector large repository
```
git clone https://github.com/MaksTarnavskyi/gector-large.git
cd gector
pip install -r requirements.txt
pip install overrides==3.1.0
```
We used `p3.2xlarge` istance for training.

### 2. Preprocess data

Prepare data in the correct format: `python scripts/gector_baselines/preprocess_data_for_gector.py`

#### Preprocess BEA only data
```
TRAIN_SOURCE=data/preprocessed_for_gector/bea/train_src
TRAIN_TARGET=data/preprocessed_for_gector/bea/train_tgt
TRAIN_PREPROCESSED=data/preprocessed_for_gector/bea/train_tagged.txt
python gector-large/utils/preprocess_data.py -s $TRAIN_SOURCE -t $TRAIN_TARGET -o $TRAIN_PREPROCESSED

DEV_SOURCE=data/preprocessed_for_gector/bea/dev_src
DEV_TARGET=data/preprocessed_for_gector/bea/dev_tgt
DEV_PREPROCESSED=data/preprocessed_for_gector/bea/dev_tagged.txt
python gector-large/utils/preprocess_data.py -s $DEV_SOURCE -t $DEV_TARGET -o $DEV_PREPROCESSED
```

#### Preprocess some GEC data (troy data is downsampled to 20%)
```
TRAIN_SOURCE=data/preprocessed_for_gector/some_gec/train_src
TRAIN_TARGET=data/preprocessed_for_gector/some_gec/train_tgt
TRAIN_PREPROCESSED=data/preprocessed_for_gector/some_gec/train_tagged.txt
python gector-large/utils/preprocess_data.py -s $TRAIN_SOURCE -t $TRAIN_TARGET -o $TRAIN_PREPROCESSED

DEV_SOURCE=data/preprocessed_for_gector/some_gec/dev_src
DEV_TARGET=data/preprocessed_for_gector/some_gec/dev_tgt
DEV_PREPROCESSED=data/preprocessed_for_gector/some_gec/dev_tagged.txt
python gector-large/utils/preprocess_data.py -s $DEV_SOURCE -t $DEV_TARGET -o $DEV_PREPROCESSED
```

### 3. Fine-tuning gector models

#### Stage 1. Fine-tuning on all GEC data (with downsampled troy).
```
TRAIN_PREPROCESSED=data/preprocessed_for_gector/some_gec/train_tagged.txt
DEV_PREPROCESSED=data/preprocessed_for_gector/some_gec/dev_tagged.txt
PRETRAIN=roberta-large_1_pie_1bw_st3
PRETRAIN_FOLDER=models/gector/
BATCH_SIZE=8
MAX_LEN=50
AC_SIZE=32
UPE=10000
TP_PROB=1
TN_PROB=1
COLD_STEPS=0
GECTOR_REPO=gector-large
VOCAB_PATH=$GECTOR_REPO/data/output_vocabulary/
MODEL_DIR=models/gector/stage1/


python train.py --vocab_path=$VOCAB_PATH --train_set=$TRAIN_PREPROCESSED --dev_set=$DEV_PREPROCESSED --pretrain=$PRETRAIN --model_dir=$MODEL_DIR  --batch_size=$BATCH_SIZE  --accumulation_size=$AC_SIZE --n_epoch=20 --patience=3 --updates_per_epoch=$UPE --cold_steps_count=$COLD_STEPS --tn_prob=$TN_PROB --tp_prob=$TP_PROB  --pretrain_folder=$PRETRAIN_FOLDER --transformer_model=roberta-large
```


#### Stage 2. Fine-tuning of stage1 model on bea-only data.
```
TRAIN_PREPROCESSED=data/preprocessed_for_gector/bea/train_tagged.txt
DEV_PREPROCESSED=data/preprocessed_for_gector/bea/dev_tagged.txt
MODEL_DIR=models/gector/stage2/
PRETRAIN=roberta-large_stage1
PRETRAIN_FOLDER=models/gector/
BATCH_SIZE=16
MAX_LEN=50
AC_SIZE=16
UPE=0
TP_PROB=1
TN_PROB=1
COLD_STEPS=0
GECTOR_REPO=gector-large
VOCAB_PATH=$GECTOR_REPO/data/output_vocabulary/


python train.py --vocab_path=$VOCAB_PATH --train_set=$TRAIN_PREPROCESSED --dev_set=$DEV_PREPROCESSED --pretrain=$PRETRAIN --model_dir=$MODEL_DIR  --batch_size=$BATCH_SIZE  --accumulation_size=$AC_SIZE --n_epoch=20 --patience=3 --updates_per_epoch=$UPE --cold_steps_count=$COLD_STEPS --tn_prob=$TN_PROB --tp_prob=$TP_PROB  --pretrain_folder=$PRETRAIN_FOLDER --transformer_model=roberta-large
```