from dataclasses import dataclass


@dataclass
class Config:
    dataset_name: str = "NEREL"
    train_file: str = "./data/NEREL/train.json"
    valid_file: str = "./data/NEREL/dev.json"
    test_file: str = "./data/NEREL/test.json"
    generated_data_directory: str = "./data/generated_data/"
    generated_param_directory: str = "./data/generated_data/model_param/"
    bert_directory: str = "DeepPavlov/rubert-base-cased"
    partial: bool = False
    model_name: str = "Set-Prediction-Networks"
    num_generated_triples: int = 10
    num_decoder_layers: int = 3
    matcher: str = "avg"  # "avg", "min"
    na_rel_coef: float = 1
    rel_loss_weight: float = 1
    head_ent_loss_weight: float = 2
    tail_ent_loss_weight: float = 2
    freeze_bert: bool = True
    batch_size: int = 8
    max_epoch: int = 50
    gradient_accumulation_steps: int = 1
    decoder_lr: float = 2e-5
    encoder_lr: float = 1e-5
    lr_decay: float = 0.01
    weight_decay: float = 1e-5
    max_grad_norm: float = 0
    optimizer: str = "AdamW"  # "Adam", "AdamW"
    n_best_size: int = 100
    max_span_length: int = 12
    refresh: bool = False
    gpu: bool = True
    visible_gpu: int = 1
    random_seed: int = 1
