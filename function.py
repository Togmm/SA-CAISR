import os
import time
import torch
import argparse
from typing import Union, Iterable, Tuple
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils_0617 import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def get_periods(dataset: str) -> list:
    """ Get list of periods for continue learning.
        Args:
            dataset (str): Name of dataset, in ['DIGINETICA', 'YOOCHOOSE'].
        Returns:
            periods (list): list of period in the form of [1, 2, ..., period_num].
    """
    # For continue learning: periods = [1, 2, ..., period_num]
    datafiles = os.listdir(os.path.join('./data', dataset))
    period_num = int(len(list(filter(lambda file: file.endswith(".txt"), datafiles))))
    periods = range(1, period_num)
    #  create dictionary to save model
    for period in periods:
        if not os.path.isdir(os.path.join('model', 'period%d' % period)):
            os.makedirs(os.path.join('model', 'period%d' % period))
    return periods

def save_model(model, args, period, epoch, last_ckpt_path):
    if last_ckpt_path != None:
        os.remove(last_ckpt_path)
    folder = f'result/{args.dataset}_tol/' + args.dataset + '_' + args.train_dir + f'/period{period}'
    os.makedirs(folder, exist_ok=True)
    fname = 'epoch=%d.ckpt' % (epoch)
    torch.save(model.state_dict(), os.path.join(folder, fname))
    last_ckpt_path = os.path.join(folder, fname)
    return last_ckpt_path

def init_model(model):
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

def evaluate_result(num_tmp_batch, tmp_sampler, model, max_item):
    model.eval()
    with torch.no_grad():
        MRR_10, RECALL_10, NDCG_10, MRR_20, RECALL_20, NDCG_20, tol_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for step in range(num_tmp_batch):
            seq, pos = tmp_sampler.sampler()
            seq, pos = np.array(seq), np.array(pos)
            result_logits = -model.predict(seq, max_item)
            result = evaluate_continue_learning(seq, result_logits, max_item, pos)
            MRR_10 += result[0]; RECALL_10 += result[1]; NDCG_10 += result[2]; MRR_20 += result[3]; RECALL_20 += result[4]; NDCG_20 += result[5]; tol_num += result[6]
        print(f'Num of Test/Valid: {tol_num}')
    model.eval()
    return MRR_10 / tol_num, RECALL_10 / tol_num, NDCG_10 / tol_num, MRR_20 / tol_num, RECALL_20 / tol_num, NDCG_20 / tol_num

def dpo_loss(beta, win_logits, lose_logits, win_logits_ref, lose_logits_ref):
    diff = win_logits - lose_logits
    diff_ref = win_logits_ref - lose_logits_ref
    # diff = win_logits - win_logits_ref
    dpo_loss = - F.logsigmoid(beta * (diff - diff_ref))
    return dpo_loss.mean()

def load_exemplars(exemplar_pre: dict) -> list:
    """ Load exemplar in previous cycle.
        Args:
            exemplar_pre (dict): Exemplars from previous cycle in the form of {item_id: [session, session,...], ...}
        Returns:
            exemplars (list): Exemplars list in the form of [session, session]
    """
    exemplars = []
    for item in exemplar_pre.values():
        if isinstance(item, list):
            exemplars.extend([i for i in item if i])
    return exemplars

def InFoNCE(logits_a, logits_b, temperature=0.07):
    """
    logits_a: [B, item_num], The predicted score of the target model (not softmax)
    logits_b: [B, item_num], The predicted score of the reference model (not softmax)
    """
    # Normalize the predicted scores (optional, it's important depending on whether the logits are normalized)
    z_a = F.normalize(logits_a, dim=1)
    z_b = F.normalize(logits_b, dim=1)

    # similarity matrix（dot-product similarity）
    sim_matrix = torch.matmul(z_a, z_b.T)  # [B, B]
    sim_matrix /= temperature

    # Construct labels (positive samples are the diagonals)
    batch_size = logits_a.size(0)
    labels = torch.arange(batch_size).to(logits_a.device)

    # Contrast loss（InfoNCE）
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def HardNegativeInfoNCE(logits_a, logits_b, temperature=0.07, top_k=128):
    """
    logits_a: [B, item_num], The predicted score of the target model (not softmax)
    logits_b: [B, item_num], The predicted score of the reference model (not softmax)
    top_k: Represents how many most similar negative samples are used for each sample
    """
    # [1] Normalized prediction representation
    z_a = F.normalize(logits_a, dim=1)  # Update Model
    z_b = F.normalize(logits_b, dim=1).detach()   # Reference Model

    # [2] similarity matrix：sim(i,j) = z_a[i] @ z_b[j]
    sim_matrix = torch.matmul(z_a, z_b.T)  # [B, B]

    batch_size = sim_matrix.size(0)

    # [3] Exclude the positive sample diagonal (mask) to avoid being selected as a hard negative
    mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix_masked = sim_matrix.masked_fill(mask, -1e9)

    # [4] Get the similarity and index of top-k hardest negatives
    topk_vals, topk_indices = torch.topk(sim_matrix_masked, k=min(top_k, batch_size), dim=1)  # [B, k]


    # [5] Get the similarity of positive samples (diagonal)
    sim_pos = torch.diag(sim_matrix).unsqueeze(1)  # [B, 1]

    # [6] Splicing positive samples + top-k hard negatives
    logits = torch.cat([sim_pos, topk_vals], dim=1)  # [B, k+1]

    # [7] Build labels: Positive sample is the 0th position of each row
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    # [8] Calculate InfoNCE Loss (only positive samples and hard negatives)
    loss = F.cross_entropy(logits / temperature, labels)

    return loss

def contrastive_loss(logits, logits_dp, labels, margin=0.5):
    """
    logits, logits_dp: (batch_size, num_classes)
    labels: (batch_size,) LongTensor
    margin: Minimum distance boundary (when far away)
    """
    logits = F.normalize(logits, dim=1)
    logits_dp = F.normalize(logits_dp, dim=1)

    # Compute the index of the top 20 maximum values ​​of logits_dp (i.e. top-20 predictions)
    topk = 20
    topk_preds = logits_dp.topk(topk, dim=1).indices  # (batch_size, 20)

    # Determine whether the true label of each sample appears in the corresponding top-20
    # labels: (batch_size,) -> (batch_size, 1) 
    labels_expanded = labels.unsqueeze(1)  # (batch_size, 1)

    # mask indicates whether it is within top-20, that is, the prediction is correct (within the grace range)
    mask = (topk_preds == labels_expanded).any(dim=1)  # (batch_size,)

    # Calculate cosine similarity (can also be changed to L2)
    sim = F.cosine_similarity(logits, logits_dp, dim=1)  # (batch_size,)

    # For the correct sample, we hope that the two are close => high similarity => loss = 1 - sim
    positive_loss = (1 - sim)[mask].mean() if mask.any() else 0.0

    # For wrong samples, we hope that the two are far away => low similarity => loss = max(0, sim - margin)
    negative_loss = F.relu(sim[~mask] - margin).mean() if (~mask).any() else 0.0

    return positive_loss + negative_loss

def distillation_loss(logits, logits_dp, temperature=1.0):
    """
    logits: Student model output,shape=[B, N]
    logits_dp: Teacher model output,shape=[B, N]
    temperature: distillation temperature,一般设为1.0~5.0之间
    """
    # Use temperature softmax to get soft targets
    soft_target = F.softmax(logits_dp / temperature, dim=1)
    log_pred = F.log_softmax(logits / temperature, dim=1)
    
    # KL divergence (student imitates teacher)
    loss_kl = F.kl_div(log_pred, soft_target, reduction='batchmean') * (temperature ** 2)
    return loss_kl

def ewc_loss(model_new, model_old, fisher_dict):
    """
    computer Elastic Weight Consolidation (EWC) loss。

    参数:
        model_new: Model in the current training stage (needs to be updated)
        model_old: Old model in the reference stage (frozen)
        fisher_dict: dict, {parameter name: Fisher information tensor}
        lambda_ewc: float, EWC loss weight
    返回:
        torch.Tensor: Total EWC loss
    """

    loss = 0.0
    new_state = dict(model_new.named_parameters())
    old_state = dict(model_old.named_parameters())

    for name, param_new in new_state.items():
        if name in fisher_dict and name in old_state:
            fisher = fisher_dict[name]
            param_old = old_state[name].detach()
            # computer EWC loss: F_i * (θ_i - θ*_i)^2
            loss += (fisher * (param_new - param_old).pow(2)).sum()

    return loss

def compute_fisher(model, ewc_sampler, loss_fn, device, max_item):
    """
    computer Fisher information matrix (approximation of diagonal) given model and data.
    return: fisher_dict, such as {name: Tensor}
    """

    model.eval()
    fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    ewc_batch = ewc_sampler.batch_num()
    for step in range(ewc_batch):
        model.zero_grad()
        session, seq, pos, neg = ewc_sampler.sampler()
        seq, pos = np.array(seq), np.array(pos)
        feature, logits = model(seq, max_item)  # batch x itemnum
        pos = pos[:, -1]
        pos = torch.LongTensor(pos).to(device)
        loss = loss_fn(logits, pos)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2)

    for name in fisher_dict:
        fisher_dict[name] /= ewc_batch

    return fisher_dict

def global_prune_by_fisher(model, fisher_dict, prune_ratio=0.2):
    """
    Global pruning (parameter zeroing) using Fisher information.

    Args:
        model (nn.Module): Model to be pruned.
        fisher_dict (dict): Fisher information dictionary, such as {param_name: tensor with same shape as param}.
        prune_ratio (float): Pruning ratio, e.g. 0.2 means zeroing the bottom 20% parameters with smallest Fisher values.

    Returns:
        mask_dict (dict): Pruning mask dictionary, such as {param_name: tensor with same shape as param}.
    """

    # 1. Collect all Fisher values
    all_fisher = torch.cat([
        f.view(-1) for name, f in fisher_dict.items()
        if name in dict(model.named_parameters())
    ])
    
    # 2. Compute global pruning threshold (by sorting Fisher values)
    threshold = torch.quantile(all_fisher, prune_ratio)

    # 3. Generate mask and apply pruning
    mask_dict = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in fisher_dict:
                fisher = fisher_dict[name]
                # Generate mask: keep important parameters
                mask = (fisher > threshold).float()
                # Apply mask: directly set unimportant parameters to zero
                param.mul_(mask)
                # Save mask: keep important parameters as 1, unimportant parameters as 0
                mask_dict[name] = mask

    return mask_dict

def fisher_weighted_dropout(model, fisher_dict, max_p=0.6):
    """
    Weighted Dropout (element-wise probability + scaling) based on Fisher information.

    Args:
        model (nn.Module): Model to be pruned.
        fisher_dict (dict): Fisher information dictionary, such as {param_name: tensor with same shape as param}.
        max_p (float): Maximum dropout probability, e.g. 0.6 means 60% dropout.

    Returns:
        mask_dict (dict): Pruning mask dictionary, such as {param_name: tensor with same shape as param}.
    """
    mask_dict = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in fisher_dict:
                fisher = fisher_dict[name]

                # 1. Min-Max normalized to [0,1]
                fisher_norm = (fisher - fisher.min()) / (fisher.max() - fisher.min() + 1e-8)

                # 2. Element-wise probability assignment: range [0, max_p]
                p = fisher_norm * max_p
                p = torch.clamp(p, max=0.99)

                # 3. Sample dropout mask
                rand = torch.rand_like(param)
                mask = (rand > p).float()

                # 4. Scaling (inverted dropout)
                scale = 1.0 / (1.0 - p)
                mask = mask * scale

                mask_dict[name] = mask

    return mask_dict

def apply_mask(model, mask_dict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask_dict:
                param.mul_(mask_dict[name])