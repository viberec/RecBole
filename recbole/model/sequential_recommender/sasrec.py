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
    
    OPTIMIZED VERSION:
    - Native FlashAttention (Boolean Masks)
    - Full-Sequence Training (Auto-constructs shifted targets)
    - GPU-Accelerated Sampling
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # 1. Load Parameters
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.sample_num = int(
            config.get("custom_train_neg_sample_args", {}).get("sample_num", 0)
        )

        # 2. Embeddings
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # 3. Native PyTorch Transformer (FlashAttention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.hidden_dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True,
            norm_first=True
        )
        self.trm_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
            enable_nested_tensor=False
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # 4. Loss Functions
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # 5. Static Buffers
        self.register_buffer(
            "position_ids",
            torch.arange(self.max_seq_length).unsqueeze(0)
        )
        # Boolean Causal Mask
        causal_mask = torch.triu(
            torch.ones(self.max_seq_length, self.max_seq_length), 
            diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        seq_length = item_seq.size(1)
        
        # Embedding
        position_ids = self.position_ids[:, :seq_length]
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = self.LayerNorm(self.dropout(item_emb + position_embedding))

        # Native Masking
        padding_mask = (item_seq == 0)
        attn_mask = self.causal_mask[:seq_length, :seq_length]

        # Transformer Pass
        output = self.trm_encoder(
            src=input_emb, 
            mask=attn_mask, 
            src_key_padding_mask=padding_mask
        )
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]      # [Batch, Seq_Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        last_item = interaction[self.POS_ITEM_ID]  # [Batch] (Next item)
        
        # 1. Forward Pass (Full Sequence)
        # Shape: [Batch, Seq_Len, Hidden]
        seq_output = self.forward(item_seq, item_seq_len)

        # ----------------------------------------------------------------------
        # FIX: Construct Full Target Sequence (Shifted)
        # ----------------------------------------------------------------------
        # item_seq: [A, B, C, 0]
        # target  : [B, C, D, 0] (where D is last_item)
        
        # Create a placeholder for targets with the same shape as input
        pos_items = torch.zeros_like(item_seq)
        
        # Shift inputs to the left: [A, B, C, 0] -> [B, C, 0]
        pos_items[:, :-1] = item_seq[:, 1:]
        
        # Append the actual "Next Item" (ground truth) to the end of the non-zero sequence
        # We use scatter to place 'last_item' at the correct 'len-1' index for each user
        # Note: RecBole usually pads at the end. The last valid item is at index `len-1`.
        # The target for the item AT `len-1` is `last_item`.
        
        # A Simpler Strategy for RecBole's default padding (Right Padding):
        # If input is [A, B, C, 0], len=3.
        # We want prediction at index 0 (A) -> B
        # We want prediction at index 1 (B) -> C
        # We want prediction at index 2 (C) -> last_item
        
        # Overwrite the shifted logic with the 'last_item' at the correct position
        # However, a simple Shift+Append works if we assume the sequence was contiguous.
        # Let's assume right-padding for simplicity, which is standard.
        pos_items[:, :-1] = item_seq[:, 1:]
        
        # Now we need to insert 'last_item' at the position of the last valid token.
        # We can use scatter_ to put 'last_item' into the 'item_seq_len - 1' position.
        # But wait, we shifted left. So the position is 'item_seq_len - 1' in the NEW tensor.
        # Actually, let's look at the mask logic.
        
        # The easiest way that works for ALL padding types (assuming contiguous history):
        # We just need to ensure that for every non-zero input, we have a target.
        # The 'pos_items' tensor we just made (shift left) is correct for t=0..T-2.
        # We just need to fix t=T-1.
        
        # Using advanced indexing to place last_item at [row, len-1]
        indices = item_seq_len - 1
        # Check bounds (prevent -1 index issues if len is 0, though unlikely)
        indices = indices.clamp(min=0) 
        pos_items.scatter_(1, indices.unsqueeze(1), last_item.unsqueeze(1))
        
        # ----------------------------------------------------------------------
        # OPTIMIZATION: Flatten & Mask (Train on All Valid Steps)
        # ----------------------------------------------------------------------
        
        # We calculate loss only where the INPUT (item_seq) was not padding.
        # (Because we are predicting the target FOR that input)
        mask = (item_seq != 0)
        
        # Flatten: [Total_Valid_Tokens, Hidden]
        seq_output = seq_output[mask]
        pos_items = pos_items[mask]

        # ----------------------------------------------------------------------
        # Loss Calculation
        # ----------------------------------------------------------------------
        if self.loss_type == "BPR":
            # GPU Optimized BPR
            neg_items = torch.randint(
                1, self.n_items, 
                (pos_items.size(0),), 
                device=pos_items.device
            )
            
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            
            pos_score = torch.sum(seq_output * pos_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_emb, dim=-1)
            
            loss = self.loss_fct(pos_score, neg_score)
            return loss

        elif self.loss_type == 'CE':
            if self.sample_num > 0:
                # Sampled Softmax (GEMM)
                n_neg = self.sample_num
                
                pos_emb = self.item_embedding(pos_items)
                pos_logits = torch.sum(seq_output * pos_emb, dim=-1, keepdim=True)
                
                neg_items = torch.randint(
                    1, self.n_items, (n_neg,), device=pos_items.device
                )
                neg_emb = self.item_embedding(neg_items)
                neg_logits = torch.matmul(seq_output, neg_emb.transpose(0, 1))
                
                logits = torch.cat([pos_logits, neg_logits], dim=1)
                targets = torch.zeros(pos_items.size(0), dtype=torch.long, device=pos_items.device)
                return self.loss_fct(logits, targets)
            
            else:
                # Full Softmax
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = self.loss_fct(logits, pos_items)
                return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        # Get full sequence
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Gather LAST item only
        batch_indices = torch.arange(seq_output.size(0), device=seq_output.device)
        seq_output = seq_output[batch_indices, item_seq_len - 1]
        
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Gather LAST item only
        batch_indices = torch.arange(seq_output.size(0), device=seq_output.device)
        seq_output = seq_output[batch_indices, item_seq_len - 1]
        
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
