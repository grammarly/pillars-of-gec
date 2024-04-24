
# pillars-of-gec
This repository provides code, state-of-the art predictions and links to the pretrained Grammatical Error Correction models for **"Pillars of Grammatical Error Correction: Comprehensive Inspection Of Contemporary Approaches In The Era of Large Language Models"** paper which was accepted for publication at <a href="https://sig-edu.org/bea/2024">BEA-2024</a> (19th Workshop on Innovative Use of NLP for Building Educational Applications; co-located with <a href="https://2024.naacl.org/">NAACL 2024</a>).


# Structure
`Scripts` directory contain required code to reproduce some of the baselines and build ensembles.  
`Data` directory contain single systems and ensembles outputs on 3 main GEC benchmarks.  
Table bellow contain single system scores and links to trained models available for download.  

## Pretrained models and results
<table>
  <tr>
    <th>Model name</th>
    <th colspan="3">CoNNL-2014 (test)</th>
    <th colspan="3">BEA-2019 (dev)</th>
    <th colspan="3">BEA-2019 (test)</th>
  </tr>
  <tr>
    <th>  </th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F05</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F05</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F05</th>
  </tr>

  <tr>
    <td>CTC-copy <a href="https://github.com/yzhangcs/ctc-copy">[repo]</a></td>
    <td>72.6</td>
    <td>47.0</td>
    <td>65.5</td> 
    <td>58.2</td>
    <td>38.0</td>
    <td>52.7</td> 
    <td>71.7</td>
    <td>59.9</td>
    <td>69.0</td> 
  </tr>
  <tr>
    <td>GECToR-2024 <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/GECToR-2024/gector-2024-roberta-large.th">[link]</a></td>
    <td>75.0</td>
    <td>44.7</td>
    <td>66.0</td> 
    <td>64.6</td>
    <td>37.2</td>
    <td>56.3</td> 
    <td>77.7</td>
    <td>59.0</td>
    <td>73.1</td> 
  </tr>
  <tr>
    <td>EditScorer <a href="https://github.com/AlexeySorokin/EditScorer">[repo]</a></td>
    <td>78.5</td>
    <td>39.4</td>
    <td>65.5</td> 
    <td>67.3</td>
    <td>36.1</td>
    <td>57.4</td> 
    <td>81.0</td>
    <td>56.1</td>
    <td>74.4</td> 
  </tr>
  <tr>
    <td>T5-11B <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/T5-11B.tar">[link]</a></td>
    <td>70.9</td>
    <td><strong>56.5</strong></td>
    <td>67.5</td> 
    <td>60.9</td>
    <td><strong>51.1</strong></td>
    <td>58.6</td> 
    <td>73.2</td>
    <td><strong>71.2</strong></td>
    <td>72.8</td> 
  </tr>
  <tr>
    <td>UL2-20B <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/ul2-20b.tar">[link]</a></td>
    <td>73.8</td>
    <td>50.4</td>
    <td>67.5</td> 
    <td>60.5</td>
    <td>48.6</td>
    <td>57.7</td> 
    <td>75.2</td>
    <td>70.0</td>
    <td>74.1</td> 
  </tr>
  <tr>
    <td>Chat-LLaMa-2-7B-FT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/llama7b.tar">[link]</a></td>
    <td>75.5</td>
    <td>46.8</td>
    <td>67.2</td> 
    <td>58.3</td>
    <td>46.0</td>
    <td>55.3</td> 
    <td>72.3</td>
    <td>67.4</td>
    <td>71.2</td> 
  </tr>
  <tr>
    <td>Chat-LLaMa-2-13B-FT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/llama13b.tar">[link]</a></td>
    <td>77.2</td>
    <td>45.6</td>
    <td>67.9</td> 
    <td>59.8</td>
    <td>46.1</td>
    <td>56.4</td> 
    <td>74.6</td>
    <td>67.8</td>
    <td>73.1</td> 
  </tr>
 <tr>
    <td>Majority-voting ensemble (best 7)</a></td>
    <td><strong>83.7</strong></td>
    <td>45.7</td>
    <td><strong>71.8</strong></td> 
    <td><strong>71.7</strong></td>
    <td>42.2</td>
    <td><strong>62.9</strong></td> 
    <td><strong>87.3</strong></td>
    <td>64.1</td>
    <td><strong>81.4</strong></td> 
  </tr>

</table>


## Evaluation

There are 3 evaluation sets that we are using for GEC:

1. CoNLL-2014 (`nucle14-2a`, m2 file is available; [m2scorer](https://gitlab.grammarly.io/nlp-research/m2scorer) is official scorer)
2. BEA19-dev (`bea-dev`, m2 file is available; [errant](https://github.com/chrisjbryant/errant) is official scorer)
3. BEA19-test (`bea-test`, m2 file is NOT available; score can be got only through [codelab](https://codalab.lisn.upsaclay.fr/competitions/4057#results
) sumbission)

### Examples of evaluation

Evalsest directory: `data/evaluation_sets`.

1. Example of evaluation with Errant

```
ERRANT_SCORER=path_to_errant_scorer_directory
INPUT_FILE=data/evaluation_sets/bea-dev.txt
M2_FILE=data/evaluation_sets/bea-dev.m2
PRED_FILE=YOUR_PRED_FILE.txt
TMP_FILE=YOUR_TMP_FILE.m2


python $ERRANT_SCORER/parallel_to_m2.py -orig $INPUT_FILE -cor $PRED_FILE -out $TMP_FILE
python $ERRANT_SCORER/compare_m2.py -hyp $TMP_FILE -ref $M2_FILE >> {{result}}
```


2. Example of evaluation with m2scorer
```
M2_SCORER=path_to_m2scorer
M2_FILE=data/evaluation_sets/nucle14-2a.m2
PRED_FILE=YOUR_PRED_FILE.txt
$M2_SCORER $PRED_FILE $M2_FILE >> {{reslut}}
```

## Citation
[to be updated once proceedings are published]
```
@misc{omelianchuk2024pillars,
      title={Pillars of Grammatical Error Correction: Comprehensive Inspection Of Contemporary Approaches In The Era of Large Language Models}, 
      author={Kostiantyn Omelianchuk and Andrii Liubonko and Oleksandr Skurzhanskyi and Artem Chernodub and Oleksandr Korniienko and Igor Samokhin},
      year={2024},
      eprint={2404.14914},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
