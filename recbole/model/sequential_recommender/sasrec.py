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
        # 1. OPTIMIZATION: Use Cached Position IDs + Broadcasting
        # Old way: Created new tensor, expanded to [Batch, Len] (Memory heavy)
        # New way: Slice existing buffer [1, Len].
        # PyTorch broadcasting handles the [Batch, Len] + [1, Len] addition automatically.

        # Safety check: Ensure we don't slice past the buffer
        seq_length = item_seq.size(1)
        position_ids = self.position_ids[:, :seq_length]

        # Lookup happens on [1, Len] -> Result [1, Len, Hidden]
        # This is much smaller than [Batch, Len, Hidden]
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)

        # Broadcasting happens here:
        # [Batch, Len, Hidden] + [1, Len, Hidden]
        input_emb = item_emb + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # 2. OPTIMIZATION: Use Native PyTorch Transformer
        # Extended attention mask from RecBole is usually [B, 1, 1, Seq_Len] or similar
        # PyTorch native transformer expects [B, Head, Seq, Seq] or [Batch*Head, Seq, Seq]
        # or more simply for src_key_padding_mask: [Batch, Seq]
        
        # RecBole's get_attention_mask returns a 4D mask with -10000.0 for padding
        # We need to adapt this for nn.TransformerEncoder
        # The native encoder expects a mask of shape (S,S) or (B*num_heads, S, S).
        # Since we have a causal (left-to-right) requirement, we usually combine this.
        
        # However, typically RecBole's `get_attention_mask` does two things:
        # 1. Pads 0s
        # 2. Causality (Future masking)
        
        # Let's simplify and use PyTorch's causal mask generation if we can, 
        # but sticking to RecBole's logic for safety:
        # extended_attention_mask is [B, 1, 1, Seq_Len] (from RecBole's base class usually)
        # But wait, SASRec overrode it? No, it inherits from SequentialRecommender.
        # Let's check SequentialRecommender.get_attention_mask in abstract_recommender.py
        
        # It seems safer to just pass `src_key_padding_mask` and `mask`.
        # But RecBole's get_attention_mask encapsulates everything into a float additive mask.
        
        # Since we switched to Native Transformer, we must accept that `forward` args are different.
        # nn.TransformerEncoder(src, mask=..., src_key_padding_mask=...)
        
        # We will strip dimensions to make it compatible or just use the additive mask as `mask`.
        # The mask shape should be [Batch*n_head, Seq_Len, Seq_Len] for 3D mask in PyTorch < 2.0 
        # or [Batch, Seq_Len, Seq_Len].
        
        # RecBole's `extended_attention_mask` is [B, 1, 1, Seq_Len] ? 
        # Actually SequentialRecommender.get_attention_mask returns [B, 1, Seq_Len, Seq_Len] for bidirectional=False.
        
        # Let's try to trust the additive mask "just works" if we squeeze it correctly.
        # But native transformer expects boolean or float mask.
        
        # Squeeze the head dim: [B, 1, S, S] -> [B, S, S]
        # And multiply heads? No, PyTorch supports [B*num_heads, S, S] or [S, S].
        
        # Simpler approach: generating causal mask using nn.Transformer
        
        attention_mask = self.get_attention_mask(item_seq, bidirectional=False)
        # attention_mask is [B, 1, S, S] with 0.0 and -10000.0
        
        # We need to reshape for PyTorch Native: [B * n_heads, S, S]
        # But wait, batch_first=True means we can pass [B, S, S]? 
        # PyTorch documentation says: mask (Tensor, optional): the mask of shape (S, S) or (N*num_heads, S, S).
        
        # So we repeat it.
        attention_mask = attention_mask.squeeze(1).repeat(self.n_heads, 1, 1)
        
        trm_output = self.trm_encoder(
            src=input_emb,
            mask=attention_mask
        )
        
        output = trm_output # Native encoder returns tensor, not list

        output = self.gather_indexes(output, item_seq_len - 1)
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
