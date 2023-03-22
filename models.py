import pytorch_lightning as pl
import torch
from torch import nn
import warnings
from torchmetrics import MeanMetric
from typing import Tuple, Dict, List, Union
from sklearn.metrics import f1_score

from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn import PBL2Norm
from ptls.data_load.padded_batch import PaddedBatch
from ptls.custom_layers import StatPooling, GEGLU
from ptls.nn.seq_step import LastStepEncoder
from torchmetrics import MeanMetric
from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss
from ptls.data_load.padded_batch import PaddedBatch


class Head(nn.Module):   
    def __init__(self, input_size, n_classes, hidden_size=64, drop_p=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, n_classes)
        )
   
    def forward(self, x):
        x = self.head(x)
        return x

class NextItemPredictionModule(pl.LightningModule):
    """GPT2 Language model
    Original sequence are encoded by `TrxEncoder`.
    Model `seq_encoder` predicts embedding of next transaction.
    Heads are used to predict each feature class of future transaction.
    Parameters
    ----------
    trx_encoder:
        Module for transform dict with feature sequences to sequence of transaction representations
    seq_encoder:
        Module for sequence processing. Generally this is transformer based encoder. Rnn is also possible
        Should works without sequence reduce
    head_hidden_size:
        Hidden size of heads for feature prediction
    seed_seq_len:
         Size of starting sequence without loss 
    total_steps:
        total_steps expected in OneCycle lr scheduler
    max_lr:
        max_lr of OneCycle lr scheduler
    weight_decay:
        weight_decay of Adam optimizer
    pct_start:
        % of total_steps when lr increase
    norm_predict:
        use l2 norm for transformer output or not
    inference_pooling_strategy:
        'out' - `seq_encoder` forward (`is_reduce_requence=True`) (B, H)
        'out_stat' - min, max, mean, std statistics pooled from `seq_encoder` layer (B, H) -> (B, 4H)
        'trx_stat' - min, max, mean, std statistics pooled from `trx_encoder` layer (B, H) -> (B, 4H)
        'trx_stat_out' - min, max, mean, std statistics pooled from `trx_encoder` layer + 'out' from `seq_encoder` (B, H) -> (B, 5H)
    """

    def __init__(self,
                 trx_encoder: torch.nn.Module,
                 seq_encoder: AbsSeqEncoder,
                 target_col: str, 
                 head_hidden_size: int = 64,
                 total_steps: int = 64000,
                 seed_seq_len: int = 16,
                 max_lr: float = 0.00005,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = False,
                 ):

        super().__init__()
        self.save_hyperparameters(ignore=['trx_encoder', 'seq_encoder'])

        self.target_col = target_col

        self.trx_encoder = trx_encoder
        assert not self.trx_encoder.numeric_values, '`numeric_values` parameter of `trx_encoder` should be == {}. Discretize all numerical features into categorical to use Tabformer model!'
        assert self.trx_encoder.embeddings, '`embeddings` parameter for `trx_encoder` should contain at least 1 feature!'

        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False

        self.head = Head(
            input_size=self._seq_encoder.embedding_size, 
            hidden_size=head_hidden_size, 
            n_classes=self.trx_encoder.embeddings[target_col].num_embeddings
        )

        if self.hparams.norm_predict:
            self.fn_norm_predict = PBL2Norm()

        self.loss = nn.CrossEntropyLoss(ignore_index=0)

        self.train_gpt_loss = MeanMetric()
        self.valid_gpt_loss = MeanMetric()

    def forward(self, batch: PaddedBatch):
        z_trx = self.trx_encoder(batch) 
        out = self._seq_encoder(z_trx)
        if self.hparams.norm_predict:
            out = self.fn_norm_predict(out)
        return out

    def loss_gpt(self, predictions, labels, is_train_step):
        y_pred = self.head(predictions[:, self.hparams.seed_seq_len:-1, :])
        y_pred = y_pred.view(-1, y_pred.size(-1))

        y_true = labels[self.target_col][:, self.hparams.seed_seq_len+1:]
        y_true = torch.flatten(y_true.long())

        loss = self.loss(y_pred, y_true)
        return loss

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)  # PB: B, T, H
        out = out.payload if isinstance(out, PaddedBatch) else out
        labels = batch.payload

        loss_gpt = self.loss_gpt(out, labels, is_train_step=True)
        self.train_gpt_loss(loss_gpt)
        self.log(f'gpt/loss', loss_gpt, sync_dist=True)

        return loss_gpt

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)  # PB: B, T, H
        out = out.payload if isinstance(out, PaddedBatch) else out
        labels = batch.payload

        loss_gpt = self.loss_gpt(out, labels, is_train_step=False)
        self.valid_gpt_loss(loss_gpt)

        y_pred = self.head(out[:, self.hparams.seed_seq_len:-1, :]).argmax(dim=2)
        y_pred_masked = y_pred[batch.seq_len_mask[:, self.hparams.seed_seq_len:-1].bool()].cpu()
 
        y_true = labels[self.target_col][:, self.hparams.seed_seq_len+1:]
        y_true_masked = y_true[batch.seq_len_mask[:, self.hparams.seed_seq_len:-1].bool()].cpu()

        return y_pred_masked, y_true_masked

    def test_step(self,batch, batch_idx):
        out = self.forward(batch)  # PB: B, T, H
        out = out.payload if isinstance(out, PaddedBatch) else out
        labels = batch.payload

        y_pred = self.head(out[:, self.hparams.seed_seq_len:-1, :]).argmax(dim=2)
        y_pred_masked = y_pred[batch.seq_len_mask[:, self.hparams.seed_seq_len:-1].bool()].cpu()
 
        y_true = labels[self.target_col][:, self.hparams.seed_seq_len+1:]
        y_true_masked = y_true[batch.seq_len_mask[:, self.hparams.seed_seq_len:-1].bool()].cpu()

        return y_pred_masked, y_true_masked

    def training_epoch_end(self, _):
        self.log(f'gpt/train_gpt_loss', self.train_gpt_loss, prog_bar=False, sync_dist=True, rank_zero_only=True)
        # self.train_gpt_loss reset not required here

    def validation_epoch_end(self, outputs):
        preds, labels = zip(*outputs)
        preds, labels = torch.cat(preds), torch.cat(labels)
        self.log(f'gpt/valid_gpt_loss', self.valid_gpt_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log(f'gpt/valid_f1_weighted', f1_score(labels.numpy(), preds.numpy(), average='weighted'), prog_bar=True)
        # self.valid_gpt_loss reset not required here

    def test_epoch_end(self, outputs):
        preds, labels = zip(*outputs)
        preds, labels = torch.cat(preds), torch.cat(labels)
        
        self.log(f'gpt/test_f1_weighted', f1_score(labels.numpy(), preds.numpy(), average='weighted'))

    def configure_optimizers(self):
        optim = torch.optim.NAdam(self.parameters(),
                                 lr=self.hparams.max_lr,
                                 weight_decay=self.hparams.weight_decay,
                                 )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optim], [scheduler]


class MLMPretrainModule(pl.LightningModule):
    """Masked Language Model (MLM) from [ROBERTA](https://arxiv.org/abs/1907.11692)
    Original sequence are encoded by `TrxEncoder`.
    Randomly sampled trx representations are replaced by MASK embedding.
    Transformer `seq_encoder` reconstruct masked embeddings.
    The loss function tends to make closer trx embedding and his predict.
    Negative samples are used to avoid trivial solution.
    Parameters
    ----------
    trx_encoder:
        Module for transform dict with feature sequences to sequence of transaction representations
    seq_encoder:
        Module for sequence processing. Generally this is transformer based encoder. Rnn is also possible
        Should works without sequence reduce
    hidden_size:
        Size of trx_encoder output.
    loss_temperature:
         temperature parameter of `QuerySoftmaxLoss`
    total_steps:
        total_steps expected in OneCycle lr scheduler
    max_lr:
        max_lr of OneCycle lr scheduler
    weight_decay:
        weight_decay of Adam optimizer
    pct_start:
        % of total_steps when lr increase
    norm_predict:
        use l2 norm for transformer output or not
    replace_proba:
        probability of masking transaction embedding
    neg_count:
        negative count for `QuerySoftmaxLoss`
    log_logits:
        if true than logits histogram will be logged. May be useful for `loss_temperature` tuning
    """

    def __init__(self,
                 trx_encoder: torch.nn.Module,
                 seq_encoder: AbsSeqEncoder,
                 total_steps: int,
                 hidden_size: int = None,
                 loss_temperature: float = 20.0,
                 max_lr: float = 0.001,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = True,
                 replace_proba: float = 0.1,
                 neg_count: int = 1,
                 log_logits: bool = False,
                 ):

        super().__init__()
        self.save_hyperparameters(ignore=['trx_encoder', 'seq_encoder'])

        self.trx_encoder = trx_encoder
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False

        if self.hparams.norm_predict:
            self.fn_norm_predict = PBL2Norm()

        if hidden_size is None:
            hidden_size = trx_encoder.output_size

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)

        self.loss_fn = QuerySoftmaxLoss(temperature=loss_temperature, reduce=False)

        self.train_mlm_loss = MeanMetric()
        self.valid_mlm_loss = MeanMetric()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams.max_lr,
                                 weight_decay=self.hparams.weight_decay,
                                 )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optim], [scheduler]

    def get_mask(self, attention_mask):
        last_ind = attention_mask.sum(dim=1) - 1 
        mask = torch.zeros_like(attention_mask)
        mask[:, last_ind] = 1
        return mask.bool()

    def mask_x(self, x, attention_mask, mask):
        shuffled_tokens = x[attention_mask.bool()]
        B, T, H = x.size()
        ix = torch.multinomial(torch.ones(shuffled_tokens.size(0)), B * T, replacement=True)
        shuffled_tokens = shuffled_tokens[ix].view(B, T, H)

        rand = torch.rand(B, T, device=x.device).unsqueeze(2).expand(B, T, H)
        replace_to = torch.where(
            rand < 0.8,
            self.token_mask.expand_as(x),  # [MASK] token 80%
            torch.where(
                rand < 0.9,
                shuffled_tokens,  # random token 10%
                x,  # unchanged 10%
            )
        )
        return torch.where(mask.bool().unsqueeze(2).expand_as(x), replace_to, x)

    def forward(self, z: PaddedBatch):
        out = self._seq_encoder(z)
        if self.hparams.norm_predict:
            out = self.fn_norm_predict(out)
        return out

    def get_neg_ix(self, mask):
        """Sample from predicts, where `mask == True`, without self element.
        sample from predicted tokens from batch
        """
        mask_num = mask.int().sum()
        mn = 1 - torch.eye(mask_num, device=mask.device)
        neg_ix = torch.multinomial(mn, self.hparams.neg_count)
    
        b_ix = torch.arange(mask.size(0), device=mask.device).view(-1, 1).expand_as(mask)[mask][neg_ix]
        t_ix = torch.arange(mask.size(1), device=mask.device).view(1, -1).expand_as(mask)[mask][neg_ix]
        return b_ix, t_ix

    def loss_mlm(self, x: PaddedBatch, is_train_step):
        mask = self.get_mask(x.seq_len_mask)
        masked_x = self.mask_x(x.payload, x.seq_len_mask, mask)

        out = self.forward(PaddedBatch(masked_x, x.seq_lens)).payload
        
        target = x.payload[mask].unsqueeze(1)  # N, 1, H
        predict = out[mask].unsqueeze(1)  # N, 1, H
        neg_ix = self.get_neg_ix(mask)
        negative = out[neg_ix[0], neg_ix[1]]  # N, nneg, H
        
        loss = self.loss_fn(target, predict, negative)

        if is_train_step and self.hparams.log_logits:
            with torch.no_grad():
                logits = self.loss_fn.get_logits(target, predict, negative)
            self.logger.experiment.add_histogram('mlm/logits',
                                                 logits.flatten().detach(), self.global_step)
        return loss

    def training_step(self, batch, batch_idx):
        x_trx = batch
        z_trx = self.trx_encoder(x_trx)  # PB: B, T, H
        loss_mlm = self.loss_mlm(z_trx, is_train_step=True)
        self.train_mlm_loss(loss_mlm)
        loss_mlm = loss_mlm.mean()
        self.log(f'mlm/loss', loss_mlm)
        return loss_mlm

    def validation_step(self, batch, batch_idx):
        x_trx = batch
        z_trx = self.trx_encoder(x_trx)  # PB: B, T, H
        loss_mlm = self.loss_mlm(z_trx, is_train_step=False)
        self.valid_mlm_loss(loss_mlm)

    def training_epoch_end(self, _):
        self.log(f'mlm/train_mlm_loss', self.train_mlm_loss, prog_bar=False)
        # self.train_mlm_loss reset not required here

    def validation_epoch_end(self, _):
        self.log(f'mlm/valid_mlm_loss', self.valid_mlm_loss, prog_bar=True)
        # self.valid_mlm_loss reset not required here
    

class GptPretrainContrastiveModule(MLMPretrainModule):
    def get_mask(self, attention_mask):
        last_ind = attention_mask.sum(dim=1) - 1 
        mask = torch.zeros_like(attention_mask)
        mask[:, last_ind] = 1
        return mask.bool()