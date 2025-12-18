import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)   #row-wise
        self.softmax_col = nn.Softmax(dim=-2)   #column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        input_tensor_norm = self.LayerNorm(input_tensor)
        mixed_query_layer = self.query(input_tensor_norm)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

#         # Standard Attention
#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer)
#         # attention_scores = 0
#         attention_scores = attention_scores / self.sqrt_attention_head_size
#         X=self.softmax(attention_scores)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         # [batch_size heads seq_len seq_len] scores
#         # [batch_size 1 1 seq_len]
#         attention_scores = attention_scores + attention_mask
#         # Normalize the attention scores to probabilities.
#         attention_probs = self.softmax(attention_scores)
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.attn_dropout(attention_probs)    
#         context_layer = torch.matmul(attention_probs, value_layer)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)       
        query_norm_inverse = 1/torch.norm(elu_query, dim=3,p=2) #(L2 norm)
        key_norm_inverse = 1/torch.norm(elu_key, dim=2,p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij',elu_query,query_norm_inverse)
        normalized_key_layer = torch.einsum('mnij,mnj->mnij',elu_key,key_norm_inverse)
        context_layer = torch.matmul(normalized_query_layer,torch.matmul(normalized_key_layer,value_layer))/ self.sqrt_attention_head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = hidden_states + input_tensor

        return hidden_states

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec_linrec(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_linrec, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            # new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            # self.attention_layernorms.append(new_attn_layernorm)

            # new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
            #                                                 args.num_heads,
            #                                                 args.dropout_rate)
            
            # self.attention_layers.append(new_attn_layer)
            new_attn_layer =  MultiHeadAttention(args.num_heads, args.hidden_units, args.dropout_rate, args.dropout_rate, layer_norm_eps=1e-8)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs_idx = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)

        # 2️⃣ 检查范围并立即同步 GPU
        min_val = int(seqs_idx.min().item())
        max_val = int(seqs_idx.max().item())
        torch.cuda.synchronize()

        if (min_val < 0) or (max_val >= self.item_emb.num_embeddings):
            print(f"[ERROR] seqs.min={min_val}, seqs.max={max_val}, "
                f"embedding_size={self.item_emb.num_embeddings}")
            raise ValueError("Embedding index out of range")

        # 3️⃣ 尝试 lookup，并捕获所有类型异常（包括 CUDA）
        try:
            seqs = self.item_emb(seqs_idx)
        except Exception as e:
            print(f"[CUDA ERROR during embedding lookup] {e}")
            print(f"seqs_idx device: {seqs_idx.device}, "
                f"embedding device: {self.item_emb.weight.device}")
            import pdb; pdb.set_trace()
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            # seqs = torch.transpose(seqs, 0, 1)
            # Q = self.attention_layernorms[i](seqs)
            # mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
            #                                 attn_mask=attention_mask)
            #                                 # need_weights=False) this arg do not work?
            # seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1)
            seqs = self.attention_layers[i](seqs, attention_mask)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, max_item): # for training    
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]  # b x unit

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # item x unit
        logits = log_feat.matmul(item_embs.T) # b x item

        return log_feat, logits

    def predict(self, log_seqs, max_item): # for inference
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]  # b x unit

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # item x unit
        logits = log_feat.matmul(item_embs.T) # b x item
        
        return logits

class SASRec(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs_idx = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)

        # 2️⃣ 检查范围并立即同步 GPU
        min_val = int(seqs_idx.min().item())
        max_val = int(seqs_idx.max().item())
        torch.cuda.synchronize()

        if (min_val < 0) or (max_val >= self.item_emb.num_embeddings):
            print(f"[ERROR] seqs.min={min_val}, seqs.max={max_val}, "
                f"embedding_size={self.item_emb.num_embeddings}")
            raise ValueError("Embedding index out of range")

        # 3️⃣ 尝试 lookup，并捕获所有类型异常（包括 CUDA）
        try:
            seqs = self.item_emb(seqs_idx)
        except Exception as e:
            print(f"[CUDA ERROR during embedding lookup] {e}")
            print(f"seqs_idx device: {seqs_idx.device}, "
                f"embedding device: {self.item_emb.weight.device}")
            import pdb; pdb.set_trace()
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, max_item): # for training    
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]  # b x unit

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # item x unit
        logits = log_feat.matmul(item_embs.T) # b x item

        return log_feat, logits

    def predict(self, log_seqs, max_item): # for inference
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]  # b x unit

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # item x unit
        logits = log_feat.matmul(item_embs.T) # b x item
        
        return logits

class SASRec_ADER(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_ADER, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        mask = (torch.LongTensor(log_seqs) != 0).float().unsqueeze(-1).to(self.dev)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1,log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)
        seqs *= mask

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= mask

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, max_item): # for training        
        item_indices = [i for i in range(1, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = log_feat.matmul(item_embs.T)

        return log_feat, logits

    def predict(self, log_seqs, max_item): # for inference
        item_indices = [i for i in range(1, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = log_feat.matmul(item_embs.T)
        return logits

class SASRec_pos_neg_seq_logits(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_pos_neg_seq_logits, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, log_seqs, max_item): # for inference
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = log_feat.matmul(item_embs.T)
        return logits


class SASRec_continue_learning_period_emb(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_continue_learning, self).__init__()

        self.item_num = item_num
        self.dev = args.device
        self.beta = args.beta

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.period_emb = torch.nn.Embedding(1, args.hidden_units)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # self.mlp_predictor = torch.nn.Sequential(
        #     torch.nn.Linear(args.hidden_units, args.hidden_units),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(args.hidden_units, self.item_num + 1)  # 输出维度为 item_num + 1
        # )
        # self.mlp_predictor = torch.nn.Linear(args.hidden_units, self.item_num + 1)

        # self.concat_compressor = torch.nn.Sequential(
        #     torch.nn.Linear(args.hidden_units * 2, args.hidden_units),
        # )

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, max_item): # for training    
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]  # b x unit
        period_emb = self.period_emb(torch.LongTensor([0]).to(self.dev)) # 1 x unit
        # log_feat += self.period_emb(torch.LongTensor([0]).to(self.dev))

        # period_emb_expanded = period_emb.expand(log_feat.size(0), -1)
        # log_feat_with_period = torch.cat([log_feat, period_emb_expanded], dim=1)  # shape: (batch_size, hidden_dim * 2)
 
        # log_feat_compressed = self.concat_compressor(log_feat_with_period)
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # item x unit
        # logits = log_feat_compressed.matmul(item_embs.T) 
        logits = log_feat.matmul(item_embs.T) # b x item
        period_logits = period_emb.matmul(item_embs.T)
        period_logits = period_logits.expand(logits.size(0), -1)
        final_logits = logits + self.beta * period_logits
        # print(f'logits:{logits}, period_logits:{period_logits}, beta:{self.beta}, final_logits:{final_logits}')
        # import pdb
        # pdb.set_trace()
        # period_logits = period_emb.expand(period_emb.size(-1), -1) # unit x unit
        # log_feat_with_period = log_feat.matmul(period_logits)
        # final_logits = log_feat_with_period.matmul(item_embs.T)

        return log_feat, final_logits

    def predict(self, log_seqs, max_item): # for inference
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]  # b x unit
        period_emb = self.period_emb(torch.LongTensor([0]).to(self.dev)) # 1 x unit
        # log_feat += self.period_emb(torch.LongTensor([0]).to(self.dev))

        # period_emb_expanded = period_emb.expand(log_feat.size(0), -1)
        # log_feat_with_period = torch.cat([log_feat, period_emb_expanded], dim=1)  # shape: (batch_size, hidden_dim * 2)
 
        # log_feat_compressed = self.concat_compressor(log_feat_with_period)
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # item x unit
        # logits = log_feat_compressed.matmul(item_embs.T) 
        logits = log_feat.matmul(item_embs.T) # b x item
        period_logits = period_emb.matmul(item_embs.T)
        period_logits = period_logits.expand(logits.size(0), -1)
        final_logits = logits + self.beta * period_logits

        # period_logits = period_emb.expand(period_emb.size(-1), -1) # unit x unit
        # log_feat_with_period = log_feat.matmul(period_logits)
        # final_logits = log_feat_with_period.matmul(item_embs.T)
        
        return final_logits

class SASRec_DPO_sequence(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_DPO_sequence, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, flag, length, seq, label_seqs, max_item): # for training        
        log_feats = self.log2feats(seq) # user_ids hasn't been used yet

        label_embs = self.item_emb(torch.LongTensor(label_seqs).to(self.dev))
        item_indices = [i for i in range(0, max_item + 1)]
        label_logits = (log_feats * label_embs).sum(dim=-1)
        
        win_logits_sum, lose_logits_sum = 0.0, 0.0
        win_logits_num, lose_logits_num = 0, 0
        new_data_logits = []
        new_data_labels = []
        for idx, len_ in enumerate(length):
            if flag[idx] == 1:
                log_feat = log_feats[idx, :, :]
                item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
                new_data_logits.extend(log_feat.matmul(item_embs.T))
                new_data_labels.extend(F.one_hot(torch.LongTensor(label_seqs[idx]), num_classes=max_item + 1).float().to(self.dev))
            else:
                
                len_tol = len(seq[idx])
                st = 0
                while(seq[idx][st] == 0): st += 1
                cut = len_tol - len_

                pre_len = min(cut - st, len_)
                win_logits_sum += label_logits[idx][cut:cut+pre_len].sum()
                win_logits_num += len(label_logits[idx][cut:cut+pre_len])
                lose_logits_sum += label_logits[idx][cut-pre_len:cut].sum()
                lose_logits_num += len(label_logits[idx][cut-pre_len:cut])
        
        lose_logits = torch.log(torch.sigmoid(win_logits_sum / win_logits_num))
        win_logits = torch.log(torch.sigmoid(lose_logits_sum / lose_logits_num))

        if len(new_data_logits) != 0:
            return win_logits, lose_logits, torch.cat(new_data_logits, dim=0), torch.cat(new_data_labels, dim=0) 
        else:
            return win_logits, lose_logits, [], []
        
    def predict(self, log_seqs, max_item): # for inference
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = log_feat.matmul(item_embs.T)
        return logits

class SASRec_DPO_argmax(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_DPO_argmax, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, flag, length, seq, label_seqs, max_item): # for training        
        log_feats = self.log2feats(seq) # batch x len x units
        item_indices = [i for i in range(0, max_item + 1)] # itemnum
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # itemnum x units
        logits = log_feats.matmul(item_embs.T)  # batch x len x itemnum
        
        win_logits_sum, lose_logits_sum = 0.0, 0.0
        logits_num = 0
        tol_num = 0
        new_data_logits = []
        new_data_labels = []
        for i, len_ in enumerate(length):
            for j in range(1, len_):
                logit = logits[i][len(seq[i]) - j]
                logit = torch.softmax(logit, dim=-1)
                label = label_seqs[i][len(seq[i]) - j]
                max_index = torch.argmax(logit)
                if label != max_index:
                    win_logits_sum += logit[label]
                    lose_logits_sum += logit[max_index]
                    logits_num += 1
                
                tol_num += 1
        
        lose_logits = torch.log(win_logits_sum / logits_num)
        win_logits = torch.log(lose_logits_sum / logits_num)
        
        return win_logits, lose_logits
        
    def predict(self, log_seqs, max_item): # for inference
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = log_feat.matmul(item_embs.T)
        return logits

class SASRec_DPO_label(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_DPO_label, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, seq, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(seq) # user_ids hasn't been used yet
    
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits
        
        
    def predict(self, log_seqs, max_item): # for inference
        item_indices = [i for i in range(0, max_item + 1)]
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        log_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = log_feat.matmul(item_embs.T)
        return logits