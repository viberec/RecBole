# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.sample_num = int(
            config.get("custom_train_neg_sample_args", {}).get("sample_num", 0)
        )

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.hidden_dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True,    # CRITICAL: SASRec data is [Batch, Seq]
            norm_first=True      # Pre-Norm is generally more stable for deep SASRec
        )
        self.trm_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
            enable_nested_tensor=False
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # OPTIMIZATION 1: Pre-compute static Position IDs
        # We create a buffer [1, max_seq_length] that lives on the GPU.
        # This prevents creating a new tensor every single forward step.
        self.register_buffer(
            "position_ids",
            torch.arange(self.max_seq_length).unsqueeze(0)
        )
        
        # OPTIMIZATION 2: Pre-compute Causal Mask (Upper Triangular -inf)
        # Used for native TransformerEncoder
        # Shape: [MAX_LEN, MAX_LEN]
        causal_mask = torch.triu(
            torch.full(
                (self.max_seq_length, self.max_seq_length), 
                float("-inf")
            ), 
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        # 1. Embedding (Broadcasting Optimization)
        seq_length = item_seq.size(1)
        
        position_ids = self.position_ids[:, :seq_length]
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # 2. Masking (Native PyTorch Logic)
        # src_key_padding_mask: [Batch, Seq] -> True where padding (0) exists
        padding_mask = (item_seq == 0)

        # mask: [Seq, Seq] -> Causal mask (prevent peeking ahead)
        # We slice our pre-computed buffer.
        attn_mask = self.causal_mask[:seq_length, :seq_length]

        # 3. Transformer Pass (Fused Kernel)
        # This is where the speedup happens.
        output = self.trm_encoder(
            src=input_emb, 
            mask=attn_mask, 
            src_key_padding_mask=padding_mask
        )
        
        # 4. Gather Last Item
        # Optimized indexing for batch gathering
        batch_indices = torch.arange(output.size(0), device=output.device)
        output = output[batch_indices, item_seq_len - 1]
        
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            if self.sample_num > 0:
                n_neg = self.sample_num
                batch_size = pos_items.size(0)

                # A. Calculate Positive Scores (Cheap)
                pos_emb = self.item_embedding(pos_items)
                # Dot Product: [Batch, H] * [Batch, H] -> Sum -> [Batch, 1]
                pos_logits = torch.sum(seq_output * pos_emb, dim=-1, keepdim=True)

                # B. Calculate Negative Scores (Heavy - Optimized)
                # Generate Negatives on GPU (SHARED NEGATIVES)
                # We sample one set of negatives for the whole batch.
                # This allows using a single GEMM (MatMul) which is much faster than BMM.
                neg_items = torch.randint(
                    1, self.n_items,
                    (n_neg,),
                    device=pos_items.device
                )
                neg_emb = self.item_embedding(neg_items)  # [Neg, Hidden]

                # USE TENSOR CORES: General Matrix Multiplication (GEMM)
                # [Batch, Hidden] @ [Hidden, Neg] -> [Batch, Neg]
                neg_logits = torch.matmul(seq_output, neg_emb.transpose(0, 1))

                # 3. Combine Scores: [Batch, 1] + [Batch, Neg] -> [Batch, 1+Neg]
                logits = torch.cat([pos_logits, neg_logits], dim=1)

                # 4. Target is always index 0
                targets = torch.zeros(
                    batch_size, dtype=torch.long, device=pos_items.device
                )

                loss = self.loss_fct(logits, targets)
                return loss
            else:
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = self.loss_fct(logits, pos_items)
                return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
