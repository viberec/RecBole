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
    - Uses Native PyTorch Transformer (FlashAttention support)
    - Implements Full-Sequence Training (trains on all timesteps, not just the last)
    - Supports BPR, Sampled Softmax (Shared Negatives), and Full Softmax
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
        
        # Check for custom sampling args, usually found in config or added manually
        self.sample_num = int(
            config.get("custom_train_neg_sample_args", {}).get("sample_num", 0)
        )

        # 2. Embeddings
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # 3. OPTIMIZATION: Native PyTorch Transformer
        # Replaces RecBole's slow Python implementation.
        # Triggers FlashAttention on supported GPUs (L4, A100, etc.)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.hidden_dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True,    # CRITICAL: SASRec uses [Batch, Seq]
            norm_first=True      # Pre-Norm is more stable for deep networks
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
        
        # 5. OPTIMIZATION: Pre-compute Static Buffers
        # Position IDs: [1, Max_Len]
        self.register_buffer(
            "position_ids",
            torch.arange(self.max_seq_length).unsqueeze(0)
        )
        
        # Causal Mask (Upper Triangular True)
        # We use a Boolean mask to enable the fastest FlashAttention kernels.
        causal_mask = torch.triu(
            torch.ones(self.max_seq_length, self.max_seq_length), 
            diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
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
        padding_mask = (item_seq == 0)
        attn_mask = self.causal_mask[:seq_length, :seq_length]

        # 3. Transformer Pass
        # Returns [Batch, Seq, Hidden] - THE FULL SEQUENCE
        # We do NOT gather the last item here anymore.
        output = self.trm_encoder(
            src=input_emb, 
            mask=attn_mask, 
            src_key_padding_mask=padding_mask
        )
        
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Get full sequence output: [Batch, Seq_Len, Hidden]
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Target: The 'pos_items' from RecBole are typically shifted (next item)
        pos_items = interaction[self.POS_ITEM_ID] # [Batch, Seq_Len]

        # ----------------------------------------------------------------------
        # OPTIMIZATION: Full Sequence Training (Convergence Speedup)
        # We flatten the batch and sequence dimensions to train on EVERY valid timestep.
        # ----------------------------------------------------------------------
        
        # 1. Identify valid steps (Mask out padding)
        mask = (pos_items != 0)
        
        # 2. Flatten: [Total_Valid_Tokens, Hidden]
        seq_output = seq_output[mask]
        pos_items = pos_items[mask]

        if self.loss_type == "BPR":
            # --- BPR Loss (Optimized) ---
            # Paper Logic: 1 Negative per Positive
            # GPU Optimization: Generate negatives on GPU instantly
            
            neg_items = torch.randint(
                1, self.n_items, 
                (pos_items.size(0),), 
                device=pos_items.device
            )
            
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            
            # Calculate scores (Dot Product)
            pos_score = torch.sum(seq_output * pos_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_emb, dim=-1)
            
            loss = self.loss_fct(pos_score, neg_score)
            return loss

        elif self.loss_type == 'CE':
            if self.sample_num > 0:
                # --- Sampled Softmax (Shared Negatives / GEMM) ---
                # Optimization: 1 Shared Pool for the whole batch
                
                n_neg = self.sample_num
                batch_size = pos_items.size(0) # Effectively Total_Valid_Tokens

                # A. Positive Scores
                pos_emb = self.item_embedding(pos_items)
                pos_logits = torch.sum(seq_output * pos_emb, dim=-1, keepdim=True)

                # B. Negative Scores (GEMM)
                neg_items = torch.randint(
                    1, self.n_items,
                    (n_neg,),
                    device=pos_items.device
                )
                neg_emb = self.item_embedding(neg_items)
                neg_logits = torch.matmul(seq_output, neg_emb.transpose(0, 1))

                # C. Combine & Loss
                logits = torch.cat([pos_logits, neg_logits], dim=1)
                targets = torch.zeros(
                    batch_size, dtype=torch.long, device=pos_items.device
                )
                loss = self.loss_fct(logits, targets)
                return loss
            
            else:
                # --- Full Softmax (Exact) ---
                # Heavy computation: [Total_Valid_Tokens, N_Items]
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
        
        # Gather LAST item only for prediction
        # [Batch, Hidden]
        batch_indices = torch.arange(seq_output.size(0), device=seq_output.device)
        seq_output = seq_output[batch_indices, item_seq_len - 1]
        
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Get full sequence
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Gather LAST item only for prediction
        batch_indices = torch.arange(seq_output.size(0), device=seq_output.device)
        seq_output = seq_output[batch_indices, item_seq_len - 1]
        
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
