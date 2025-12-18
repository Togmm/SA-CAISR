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
    # if last_ckpt_path != None:
    #     os.remove(last_ckpt_path)
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
    logits_a: [B, item_num], 目标模型的预测分数（未 softmax）
    logits_b: [B, item_num], 参考模型的预测分数（未 softmax）
    """
    # 归一化预测分数（可选，如果 logits 本身不归一化，这很重要）
    z_a = F.normalize(logits_a, dim=1)
    z_b = F.normalize(logits_b, dim=1)

    # 相似度矩阵（dot-product similarity）
    sim_matrix = torch.matmul(z_a, z_b.T)  # [B, B]
    sim_matrix /= temperature

    # 构建标签（正样本是对角线）
    batch_size = logits_a.size(0)
    labels = torch.arange(batch_size).to(logits_a.device)

    # 对比损失（InfoNCE）
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def HardNegativeInfoNCE(logits_a, logits_b, temperature=0.07, top_k=128):
    """
    logits_a: [B, item_num]，目标模型的预测分数（未 softmax）
    logits_b: [B, item_num]，参考模型的预测分数（未 softmax）
    top_k: 每个样本使用多少个最相似的负样本
    """
    # [1] 归一化预测表示
    z_a = F.normalize(logits_a, dim=1)  # 当前模型
    z_b = F.normalize(logits_b, dim=1)  # 参考模型

    # [2] 相似度矩阵：sim(i,j) = z_a[i] @ z_b[j]
    sim_matrix = torch.matmul(z_a, z_b.T)  # [B, B]

    batch_size = sim_matrix.size(0)

    # [3] 排除正样本对角线（mask），避免被选为 hard negative
    mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix_masked = sim_matrix.masked_fill(mask, -1e9)

    # [4] 获取 top-k hardest negatives 的相似度和索引
    topk_vals, topk_indices = torch.topk(sim_matrix_masked, k=min(top_k, batch_size), dim=1)  # [B, k]


    # [5] 获取正样本相似度（对角线）
    sim_pos = torch.diag(sim_matrix).unsqueeze(1)  # [B, 1]

    # [6] 拼接正样本 + top-k hard negatives
    logits = torch.cat([sim_pos, topk_vals], dim=1)  # [B, k+1]

    # [7] 构建标签：正样本是每行的第0个位置
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    # [8] 计算 InfoNCE Loss（仅用正样本和 hard negatives）
    loss = F.cross_entropy(logits / temperature, labels)

    return loss

def contrastive_loss(logits, logits_dp, labels, margin=0.5):
    """
    logits, logits_dp: (batch_size, num_classes)
    labels: (batch_size,) LongTensor
    margin: 最小距离的边界（远离时）
    """
    logits = F.normalize(logits, dim=1)
    logits_dp = F.normalize(logits_dp, dim=1)

    # 计算 logits_dp 的前 20 个最大值的索引（即 top-20 预测）
    topk = 20
    topk_preds = logits_dp.topk(topk, dim=1).indices  # (batch_size, 20)

    # 判断每个样本的真实标签是否出现在对应的 top-20 中
    # labels: (batch_size,) -> (batch_size, 1) 以便广播比较
    labels_expanded = labels.unsqueeze(1)  # (batch_size, 1)

    # mask 表示是否在 top-20 内，即预测正确（在宽限范围内）
    mask = (topk_preds == labels_expanded).any(dim=1)  # (batch_size,)

    # 计算余弦相似度（也可以换成 L2）
    sim = F.cosine_similarity(logits, logits_dp, dim=1)  # (batch_size,)

    # 对于正确的样本，希望两者靠近 => 相似度高 => loss = 1 - sim
    positive_loss = (1 - sim)[mask].mean() if mask.any() else 0.0

    # 对于错误的样本，希望两者远离 => 相似度低 => loss = max(0, sim - margin)
    negative_loss = F.relu(sim[~mask] - margin).mean() if (~mask).any() else 0.0

    return positive_loss + negative_loss

def contrastive_loss_fe(logits, logits_dp, feature, feature_dp, labels, margin=0.5):
    """
    logits, logits_dp: (batch_size, num_classes)
    labels: (batch_size,) LongTensor
    margin: 最小距离的边界（远离时）
    """
    logits = F.normalize(logits, dim=1)
    logits_dp = F.normalize(logits_dp, dim=1)
    # 计算 logits_dp 的前 20 个最大值的索引（即 top-20 预测）
    topk = 20
    topk_preds = logits_dp.topk(topk, dim=1).indices  # (batch_size, 20)

    # 判断每个样本的真实标签是否出现在对应的 top-20 中
    # labels: (batch_size,) -> (batch_size, 1) 以便广播比较
    labels_expanded = labels.unsqueeze(1)  # (batch_size, 1)

    # mask 表示是否在 top-20 内，即预测正确（在宽限范围内）
    mask = (topk_preds == labels_expanded).any(dim=1)  # (batch_size,)

    # 计算余弦相似度（也可以换成 L2）
    sim = F.cosine_similarity(feature, feature_dp, dim=1)  # (batch_size,)

    # 对于正确的样本，希望两者靠近 => 相似度高 => loss = 1 - sim
    positive_loss = (1 - sim)[mask].mean() if mask.any() else 0.0

    # 对于错误的样本，希望两者远离 => 相似度低 => loss = max(0, sim - margin)
    negative_loss = F.relu(sim[~mask] - margin).mean() if (~mask).any() else 0.0

    return positive_loss + negative_loss

def distillation_loss(logits, logits_dp, temperature=1.0):
    """
    logits: 学生模型输出，shape=[B, N]
    logits_dp: 教师模型输出，shape=[B, N]
    temperature: 蒸馏温度，一般设为1.0~5.0之间
    """
    # 使用temperature softmax得到soft targets
    soft_target = F.softmax(logits_dp / temperature, dim=1)
    log_pred = F.log_softmax(logits / temperature, dim=1)
    
    # KL散度（student 模仿 teacher）
    loss_kl = F.kl_div(log_pred, soft_target, reduction='batchmean') * (temperature ** 2)
    return loss_kl

def ewc_loss(model_new, model_old, fisher_dict):
    """
    计算 Elastic Weight Consolidation (EWC) 损失。

    参数:
        model_new: 当前训练阶段的模型（需要更新的）
        model_old: 参考阶段的旧模型（已冻结）
        fisher_dict: dict，{参数名: Fisher 信息张量}
        lambda_ewc: float，EWC 损失权重
    返回:
        torch.Tensor: 总 EWC 损失
    """

    loss = 0.0
    new_state = dict(model_new.named_parameters())
    old_state = dict(model_old.named_parameters())

    for name, param_new in new_state.items():
        if name in fisher_dict and name in old_state:
            fisher = fisher_dict[name]
            param_old = old_state[name].detach()
            # 计算 EWC loss: F_i * (θ_i - θ*_i)^2
            loss += (fisher * (param_new - param_old).pow(2)).sum()

    return loss

def compute_fisher(model, ewc_sampler, loss_fn, device, max_item):
    """
    给定模型和数据，估计 Fisher 信息矩阵（近似对角）。
    返回: fisher_dict，形如 {name: Tensor}
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

    # 归一化
    for name in fisher_dict:
        fisher_dict[name] /= ewc_batch

    return fisher_dict

def global_prune_by_fisher(model, fisher_dict, prune_ratio=0.2):
    """
    使用 Fisher 信息进行全局剪枝（参数置零）。

    Args:
        model (nn.Module): 要剪枝的模型。
        fisher_dict (dict): Fisher 信息字典，形如 {param_name: tensor with same shape as param}。
        prune_ratio (float): 剪枝比例，比如 0.2 表示置零最小的 20% 参数。

    Returns:
        mask_dict (dict): 剪枝后的 mask 字典，可用于后续继续训练时保持稀疏性。
    """

    # 1. 收集所有 Fisher 值
    all_fisher = torch.cat([
        f.view(-1) for name, f in fisher_dict.items()
        if name in dict(model.named_parameters())
    ])
    
    # 2. 计算全局剪枝阈值（按Fisher值排序）
    threshold = torch.quantile(all_fisher, prune_ratio)

    # 3. 生成 mask 并应用剪枝
    mask_dict = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in fisher_dict:
                fisher = fisher_dict[name]
                # 生成 mask：保留重要参数
                mask = (fisher > threshold).float()
                # 应用 mask：直接将不重要参数置零
                param.mul_(mask)
                # 保存 mask
                mask_dict[name] = mask

    return mask_dict

def fisher_weighted_dropout(model, fisher_dict, max_p=0.6):
    """
    基于 Fisher 信息的加权 Dropout (逐元素概率 + 缩放)。
    """
    mask_dict = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in fisher_dict:
                fisher = fisher_dict[name]

                # 1. Min-Max 归一化到 [0,1]
                fisher_norm = (fisher - fisher.min()) / (fisher.max() - fisher.min() + 1e-8)

                # 2. 概率分配：范围 [0, max_p]
                p = fisher_norm * max_p

                # 3. 采样 dropout mask
                rand = torch.rand_like(param)
                mask = (rand > p).float()

                # 4. 缩放 (逐元素 inverted dropout)
                scale = 1.0 / (1.0 - p + 1e-8)
                # param.mul_(mask * scale)

                mask_dict[name] = mask

    return mask_dict

def apply_mask(model, mask_dict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask_dict:
                param.mul_(mask_dict[name])