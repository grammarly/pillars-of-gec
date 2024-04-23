# EditScorer

*source:* https://github.com/AlexeySorokin/EditScorer

*paper:* https://aclanthology.org/2022.emnlp-main.785.pdf


### installation & models

To prepare the codebase:
```
git clone https://github.com/AlexeySorokin/EditScorer.git
cd EditScorer
pip install -r requirements.txt
```

To get the models

```
cd data
mkdir -p bea_reranking && cd bea_reranking
wget https://www.dropbox.com/s/m5dot9rp0vwkcc8/gector_variants.tar.gz
tar -xzvf gector_variants.tar.gz
cd ../..
```

### Generate predications with staged processing

For `nucle` and `bea-dev`:

```
for DS_NAME in "nucle14-2a" "bea-dev"; do
  for M_NAME in "roberta_1_gector" "xlnet_0_gector"; do
  
    INPUT_FILE="data/evaluation_sets/${DS_NAME}.m2"
    GENDALF_PRED_PATH="outputs/${DS_NAME}___${M_NAME}.txt"
    PRED_PATH="EditScorer/dump/reranking"
    MODEL_PATH="EditScorer/checkpoints/pie_bea_ft2-gector"
    MODEL_NAME="checkpoint_2.pt"
  
    python apply_staged_model.py -c $MODEL_PATH -C $MODEL_NAME -i $INPUT_FILE -v $GENDALF_PRED_PATH -O ${DS_NAME}___${M_NAME} -s 8 -a 0.1
  
  done
done
```

For `bea-test`:
```
for M_NAME in "roberta_1_gector" "xlnet_0_gector"; do

  INPUT_FILE="data/evaluation_sets/bea-test.txt"
  GENDALF_PRED_PATH="outputs/bea-test___${M_NAME}.txt"
  PRED_PATH="EditScorer/dump/reranking"
  MODEL_PATH="EditScorer/checkpoints/pie_bea_ft2-gector"
  MODEL_NAME="checkpoint_2.pt"

  python apply_staged_model.py -c $MODEL_PATH -C $MODEL_NAME -i $INPUT_FILE -v $GENDALF_PRED_PATH -O bea-test___${M_NAME} -s 8 -a 0.1 -r
  
done
```