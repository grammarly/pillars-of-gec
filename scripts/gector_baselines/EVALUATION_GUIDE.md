# GECTOR pretrained model to evaluate

### 1. Download models. 

download the model to the model’s directory: models/gector/
cd models/gector/

#### roberta-large
download from google drive
https://drive.google.com/drive/folders/1Si2hwmskb7QxqSFtPBsivl_FujkR3p6l


### 2. Install repositories

#### gector large repository
```
git clone https://github.com/MaksTarnavskyi/gector-large.git
cd gector
pip install -r requirements.txt
pip install overrides==3.1.0
```

#### M2 scorer
`git clone git@ssh.gitlab.grammarly.io:nlp-research/m2scorer.git`

#### ERRANT
```
git clone https://github.com/chrisjbryant/errant.git
install errant from the source
```

### 3. Generate predictions

```
GECTOR_REPO=gector-large
INPUT_FILE=evaluaiton_sets/nucle14-2a.txt

PRED_FILE=pred_gector_tmp.txt
MODEL_PATH=models/gector/roberta-large_1_pie_1bw_st3.th
VOCAB_PATH=$GECTOR_REPO/data/output_vocabulary/

MIE=0.65
CONF=0.1
TMODEL=roberta-large

python $GECTOR_REPO/predict.py --model_path=$MODEL_PATH --vocab_path=$VOCAB_PATH --input_file=$INPUT_FILE --output_file=$PRED_FILE --min_error_probability=$MIE --additional_confidence=$CONF --transformer_model=$TMODEL
```

### 4. Get scores
#### M2
```
M2_SCORER=path to m2scorer/m2scorer
M2_FILE=data/evaluation_sets/nucle14-2a.m2

PRED_FILE=pred_gector_tmp.txt
$M2_SCORER $PRED_FILE $M2_FILE 
```

#### ERRANT
```
INPUT_FILE=data/evaluation_sets/bea-dev.txt
M2_FILE=data/evaluation_sets/bea-dev.m2
PRED_FILE=pred_gector_tmp.txt
TMP_FILE=pred_gector_tmp.m2


errant_parallel -orig $INPUT_FILE -cor $PRED_FILE -out $TMP_FILE
errant_compare -hyp $TMP_FILE -ref $M2_FILE 
```