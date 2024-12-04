import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import typer

from config import Config
from models.setpred4RE import SetPred4RE
from trainer.trainer import Trainer
from utils.data import build_data


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class Main(Config):
    def __post_init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.visible_gpu)
        set_seed(self.random_seed)
        data = build_data(self)
        model = SetPred4RE(self, data.relational_alphabet.size())
        trainer = Trainer(model, data, self)
        trainer.train_model()


if __name__ == "__main__":
    typer.run(Main)
