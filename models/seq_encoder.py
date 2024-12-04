import torch.nn as nn
from transformers import BertModel

from config import Config


class SeqEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained(cfg.bert_directory)
        if cfg.freeze_bert:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.bert_config = self.bert.config

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        return last_hidden_state, pooler_output
