import gc
import random

import torch
from rich import print
from rich.progress import track
from torch import nn, optim
from transformers import AdamW

from config import Config
from models.setpred4RE import SetPred4RE
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold
from utils.metric import metric, num_metric, overlap_metric


class Trainer(nn.Module):
    def __init__(self, model: SetPred4RE, data, cfg: Config):
        super().__init__()
        self.args = cfg
        self.model = model
        self.data = data

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        component = ["encoder", "decoder"]
        grouped_params = [
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if not any(nd in name for nd in no_decay) and component[0] in name
                ],
                "weight_decay": cfg.weight_decay,
                "lr": cfg.encoder_lr,
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if any(nd in name for nd in no_decay) and component[0] in name
                ],
                "weight_decay": 0.0,
                "lr": cfg.encoder_lr,
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if not any(nd in name for nd in no_decay) and component[1] in name
                ],
                "weight_decay": cfg.weight_decay,
                "lr": cfg.decoder_lr,
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if any(nd in name for nd in no_decay) and component[1] in name
                ],
                "weight_decay": 0.0,
                "lr": cfg.decoder_lr,
            },
        ]
        if cfg.optimizer == "Adam":
            self.optimizer = optim.Adam(grouped_params)
        elif cfg.optimizer == "AdamW":
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        if cfg.gpu:
            self.cuda()

    def train_model(self):
        best_f1 = 0
        best_result_epoch = -1
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1
        # result = self.eval_model(self.data.test_loader)
        for epoch in range(self.args.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            avg_loss = AverageMeter()
            random.shuffle(train_loader)
            for batch_id in track(range(total_batch), description=f"Epoch {epoch:03d}/{self.args.max_epoch:03d}"):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                train_instance = train_loader[start:end]
                # print([ele[0] for ele in train_instance])
                if not train_instance:
                    continue
                input_ids, attention_mask, targets, _ = self.model.batchify(train_instance)
                loss, _ = self.model(input_ids, attention_mask, targets)
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 100 == 0 and batch_id != 0:
                    print(f"     Instance: {start}; loss: {avg_loss.avg:.4f}")
            gc.collect()
            torch.cuda.empty_cache()
            # Validation
            print(f"=== Epoch {epoch} Validation ===")
            result = self.eval_model(self.data.valid_loader)
            # Test
            # print("=== Epoch %d Test ===" % epoch, flush=True)
            # result = self.eval_model(self.data.test_loader)
            f1 = result["f1"]
            if f1 > best_f1:
                print("Achieving Best Result on Validation Set.", flush=True)
                torch.save(
                    {"state_dict": self.model.state_dict()},
                    self.args.generated_param_directory
                    + f"{self.model.name}_{self.args.dataset_name}_epoch_{epoch}_f1_{result['f1']:.4f}.model",
                )
                best_f1 = f1
                best_result_epoch = epoch

            gc.collect()
            torch.cuda.empty_cache()
        print(f"Best result on validation set is {best_f1} achieving at epoch {best_result_epoch}.")

    def eval_model(self, eval_loader):
        self.model.eval()
        # print(self.model.decoder.query_embed.weight)
        prediction, gold = {}, {}
        with torch.no_grad():
            batch_size = self.args.batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, target, info = self.model.batchify(eval_instance)
                gold.update(formulate_gold(target, info))
                # print(target)
                gen_triples = self.model.gen_triples(input_ids, attention_mask, info)
                prediction.update(gen_triples)
        num_metric(prediction, gold)
        overlap_metric(prediction, gold)
        return metric(prediction, gold)

    def load_state_dict(self, state_dict):  # type: ignore
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
