# SPN4RE-NEREL

## Setup environment
```shell
git clone https://github.com/dartt0n/SPN4RE-NEREL.git
cd SPN4RE-NEREL
uv sync
```

## Generate data
```shell
uv run scripts/generate_data.py
```

## Run training
```shell
uv run main.py --no-gpu # in case you don't have GPU
uv run main.py --gpu    # in case you have GPU
```

Full list of options:
```shell
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --dataset-name                                       TEXT     [default: NEREL]                                                                                                                                               │
│ --train-file                                         TEXT     [default: ./data/NEREL/train.json]                                                                                                                             │
│ --valid-file                                         TEXT     [default: ./data/NEREL/dev.json]                                                                                                                               │
│ --test-file                                          TEXT     [default: ./data/NEREL/test.json]                                                                                                                              │
│ --generated-data-directory                           TEXT     [default: ./data/generated_data/]                                                                                                                              │
│ --generated-param-directory                          TEXT     [default: ./data/generated_data/model_param/]                                                                                                                  │
│ --bert-directory                                     TEXT     [default: DeepPavlov/rubert-base-cased]                                                                                                                        │
│ --partial                        --no-partial                 [default: no-partial]                                                                                                                                          │
│ --model-name                                         TEXT     [default: Set-Prediction-Networks]                                                                                                                             │
│ --num-generated-triples                              INTEGER  [default: 10]                                                                                                                                                  │
│ --num-decoder-layers                                 INTEGER  [default: 3]                                                                                                                                                   │
│ --matcher                                            TEXT     [default: avg]                                                                                                                                                 │
│ --na-rel-coef                                        FLOAT    [default: 1]                                                                                                                                                   │
│ --rel-loss-weight                                    FLOAT    [default: 1]                                                                                                                                                   │
│ --head-ent-loss-weight                               FLOAT    [default: 2]                                                                                                                                                   │
│ --tail-ent-loss-weight                               FLOAT    [default: 2]                                                                                                                                                   │
│ --freeze-bert                    --no-freeze-bert             [default: freeze-bert]                                                                                                                                         │
│ --batch-size                                         INTEGER  [default: 8]                                                                                                                                                   │
│ --max-epoch                                          INTEGER  [default: 50]                                                                                                                                                  │
│ --gradient-accumulation-steps                        INTEGER  [default: 1]                                                                                                                                                   │
│ --decoder-lr                                         FLOAT    [default: 2e-05]                                                                                                                                               │
│ --encoder-lr                                         FLOAT    [default: 1e-05]                                                                                                                                               │
│ --lr-decay                                           FLOAT    [default: 0.01]                                                                                                                                                │
│ --weight-decay                                       FLOAT    [default: 1e-05]                                                                                                                                               │
│ --max-grad-norm                                      FLOAT    [default: 0]                                                                                                                                                   │
│ --optimizer                                          TEXT     [default: AdamW]                                                                                                                                               │
│ --n-best-size                                        INTEGER  [default: 100]                                                                                                                                                 │
│ --max-span-length                                    INTEGER  [default: 12]                                                                                                                                                  │
│ --refresh                        --no-refresh                 [default: no-refresh]                                                                                                                                          │
│ --gpu                            --no-gpu                     [default: gpu]                                                                                                                                                 │
│ --visible-gpu                                        INTEGER  [default: 1]                                                                                                                                                   │
│ --random-seed                                        INTEGER  [default: 1]                                                                                                                                                   │
│ --help                                                        Show this message and exit.                                                                                                                                    │
╰──
```

---- 
original readme:
# Joint Entity and Relation Extraction with Set Prediction Networks
[![GitHub stars](https://img.shields.io/github/stars/DianboWork/SPN4RE?style=flat-square)](https://github.com/DianboWork/SPN4RE/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/DianboWork/SPN4RE?style=flat-square&color=blueviolet)](https://github.com/DianboWork/SPN4RE/network/members)

Source code for [Joint Entity and Relation Extraction with Set Prediction Networks](https://arxiv.org/abs/2011.01675). We would appreciate it if you cite our paper as following:

```
@article{sui2020joint,
  title={Joint Entity and Relation Extraction with Set Prediction Networks},
  author={Sui, Dianbo and Chen, Yubo and Liu, Kang and Zhao, Jun and Zeng, Xiangrong and Liu, Shengping},
  journal={arXiv preprint arXiv:2011.01675},
  year={2020}
}
```
##  Model Training
### Requirement:
```
Python: 3.7   
PyTorch: >= 1.5.0 
Transformers: 2.6.0
```

###  NYT Partial Match
* Note That: Replacing BERT_DIR in the command line with the actual directory of BERT-base-cased in your machine!!!
* 注意：需将命令行中的BERT_DIR替换为你机器中实际存储BERT的目录！！！

```shell
python -m main --bert_directory BERT_DIR --num_generated_triples 15 --na_rel_coef 1 --max_grad_norm 1 --max_epoch 100 --max_span_length 10
```

###  NYT Exact Match

```shell
python -m main --bert_directory BERT_DIR --num_generated_triples 15 --max_grad_norm 2.5 --na_rel_coef 0.25 --max_epoch 100 --max_span_length 10
```
or 
```shell
python -m main --bert_directory BERT_DIR --num_generated_triples 15 --max_grad_norm 1 --na_rel_coef 0.5 --max_epoch 100 --max_span_length 10
```

### WebNLG Partial Match
```shell
python -m main --bert_directory BERT_DIR --batch_size 4 --num_generated_triples 10 --na_rel_coef 0.25 --max_grad_norm 20  --max_epoch 100 --encoder_lr 0.00002 --decoder_lr 0.00005 --num_decoder_layers 4 --max_span_length 10 --weight_decay 0.000001 --lr_decay 0.02
```
## Trained Model Parameters
Model parameters can be download in [Baidu Pan](https://pan.baidu.com/s/1nL-qZs16x684d98APVn8FQ) (key: SetP) :sunglasses:
