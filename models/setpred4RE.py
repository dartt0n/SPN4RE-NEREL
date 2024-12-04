import torch
import torch.nn as nn

from config import Config
from models.seq_encoder import SeqEncoder
from models.set_criterion import SetCriterion
from models.set_decoder import SetDecoder
from utils.functions import generate_triple


class SetPred4RE(nn.Module):
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = SeqEncoder(cfg)
        config = self.encoder.bert_config
        self.num_classes = num_classes
        self.decoder = SetDecoder(
            config,
            cfg.num_generated_triples,
            cfg.num_decoder_layers,
            num_classes,
            return_intermediate=False,
        )
        self.criterion = SetCriterion(
            num_classes,
            loss_weight=self.get_loss_weight(cfg),
            na_coef=cfg.na_rel_coef,
            losses=["entity", "relation"],
            matcher=cfg.matcher,
        )

    def forward(self, input_ids, attention_mask, targets=None):
        last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask)
        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(
            encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask
        )
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        head_start_logits = head_start_logits.squeeze(-1).masked_fill(
            (1 - attention_mask.unsqueeze(1)).bool(),
            -10000.0,
        )
        head_end_logits = head_end_logits.squeeze(-1).masked_fill(
            (1 - attention_mask.unsqueeze(1)).bool(),
            -10000.0,
        )
        tail_start_logits = tail_start_logits.squeeze(-1).masked_fill(
            (1 - attention_mask.unsqueeze(1)).bool(),
            -10000.0,
        )
        tail_end_logits = tail_end_logits.squeeze(-1).masked_fill(
            (1 - attention_mask.unsqueeze(1)).bool(),
            -10000.0,
        )  # [bsz, num_generated_triples, seq_len]
        outputs = {
            "pred_rel_logits": class_logits,
            "head_start_logits": head_start_logits,
            "head_end_logits": head_end_logits,
            "tail_start_logits": tail_start_logits,
            "tail_end_logits": tail_end_logits,
        }
        if targets is not None:
            loss = self.criterion(outputs, targets)
            return loss, outputs
        else:
            return outputs

    def gen_triples(self, input_ids, attention_mask, info):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            # print(outputs)
            pred_triple = generate_triple(outputs, info, self.cfg, self.num_classes)
            # print(pred_triple)
        return pred_triple

    def batchify(self, batch_list):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens, strict=False)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
        if self.cfg.gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = [
                {key: torch.tensor(value, dtype=torch.long, requires_grad=False).cuda() for key, value in target.items()}  # noqa: E501
                for target in targets
            ]
        else:
            targets = [
                {key: torch.tensor(value, dtype=torch.long, requires_grad=False) for key, value in target.items()}
                for target in targets
            ]
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info

    @staticmethod
    def get_loss_weight(args):
        return {
            "relation": args.rel_loss_weight,
            "head_entity": args.head_ent_loss_weight,
            "tail_entity": args.tail_ent_loss_weight,
        }
