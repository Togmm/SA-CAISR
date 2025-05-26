import os
import time
import torch
import argparse
from typing import Union, Iterable, Tuple
from model import SASRec, SASRec_continue_learning
from utils import *
import torch.nn.functional as F

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

def save_model(model, args, period, epoch):
    folder = './' + args.dataset + '_' + args.train_dir + f'/period{period}'
    os.makedirs(folder, exist_ok=True)
    fname = 'epoch=%d.ckpt' % (epoch)
    torch.save(model.state_dict(), os.path.join(folder, fname))

def init_model(model):
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

def evaluate_result(num_tmp_batch, tmp_sampler, model):
    model.eval()
    with torch.no_grad():
        MRR_10, RECALL_10, MRR_20, RECALL_20, tol_num = 0.0, 0.0, 0.0, 0.0, 0.0
        for step in range(num_tmp_batch):
            seq, pos, neg = tmp_sampler.sampler()
            seq, pos = np.array(seq), np.array(pos)
            result_logits = -model.predict(seq, max_item)
            result = evaluate_continue_learning(result_logits, max_item, pos)
            MRR_10 += result[0]; RECALL_10 += result[1]; MRR_20 += result[2]; RECALL_20 += result[3]; tol_num += result[4]
        print(f'Num of Test/Valid: {tol_num}')
    model.eval()
    return MRR_10 / tol_num, RECALL_10 / tol_num, MRR_20 / tol_num, RECALL_20 / tol_num

def dpo_loss(beta, win_logits, lose_logits, win_logits_ref, lose_logits_ref):
    diff = win_logits - lose_logits
    diff_ref = win_logits_ref - lose_logits_ref
    # diff = win_logits - win_logits_ref
    dpo_loss = - F.logsigmoid(beta * (diff - diff_ref))
    return dpo_loss.mean()


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

def contrastive_loss(logits, logits_dp, labels, margin=0.5):
    """
    logits, logits_dp: (batch_size, num_classes)
    labels: (batch_size,) LongTensor
    margin: 最小距离的边界（远离时）
    """

    # 计算 logits_dp 的预测类别
    pred_dp = logits_dp.argmax(dim=1)  # (batch_size,)

    # 判断哪些是“正例”（预测对了）
    mask = (pred_dp == labels)  # (batch_size,)

    # 计算余弦相似度（也可以换成 L2）
    sim = F.cosine_similarity(logits, logits_dp, dim=1)  # (batch_size,)

    # 对于正确的样本，希望两者靠近 => 相似度高 => loss = 1 - sim
    positive_loss = (1 - sim)[mask].mean() if mask.any() else 0.0

    # 对于错误的样本，希望两者远离 => 相似度低 => loss = max(0, sim - margin)
    negative_loss = F.relu(sim[~mask] - margin).mean() if (~mask).any() else 0.0

    return positive_loss + negative_loss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='DIGINETICA', type=str)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--test_batch', default=64, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=200, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.9, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--finetune', default=False, type=bool)  # use fine tuned hyper-parameter without dropout
parser.add_argument('--dropout', default=False, type=bool)  # use dropout
parser.add_argument('--joint', default=False, type=bool)  # use joint learning
parser.add_argument('--valid_portion', default=0.1, type=float)
parser.add_argument('--stop', default=5, type=int) 
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--beta', default=20, type=int)
parser.add_argument('--alpha', default=4, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    log = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'a')
    log.write('epoch (val_MRR, val_Recall) (test_MRR, test_Recall)\n')

    if args.dataset == 'DIGINETICA':
        itemnum = 43136    # number of items in DIGINETICA
    elif args.dataset == 'YOOCHOOSE':
        itemnum = 25958    # number of items in YOOCHOOSE
    elif args.dataset == 'TAOBAO':
        itemnum = 681413
    else:
        raise ValueError('Invalid dataset name')
    
    model_dp = SASRec_continue_learning(itemnum, args).to(args.device)
    args.dropout_rate = 0.3
    model = SASRec_continue_learning(itemnum, args).to(args.device)
    

    periods = get_periods(args.dataset)
    # dataloader = DataLoader(args.dataset)
    dataloader = DataLoader_TAOBAO(args.dataset, args.maxlen)
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    best_epoch, itemnum_prev = 0, 0

    MRR_20 = []
    Recall_20 = []
    MRR_10 = []
    Recall_10 = []
 
    for period in periods:
        print('Period %d:' % period)
        best_performance, performance = 0, 0
        new_data = dataloader.train_loader(period - 1)

        if args.joint and period > 1:
            for p in range(1, period):
                pre_train_sess = dataloader.train_loader(p-1)
                new_data.extend(pre_train_sess)

        max_item = dataloader.max_item()
        print(f'Max_item:{max_item}')
        train_sampler = Sampler_pos_neg_seq(new_data, args.maxlen, args.batch_size, max_item)
        # train_sampler = Sampler(new_data, args.maxlen, args.batch_size)
        user_valid, user_train = train_sampler.split_data(valid_portion=0.1, return_train=True)
        num_batch = train_sampler.batch_num()
        # load test data
        user_test, info = dataloader.evaluate_loader(period)
        print(info)
        test_sampler = Sampler(user_test, args.maxlen, args.test_batch, max_item, True)
        num_test_batch = test_sampler.batch_num()
        
        cc = 0.0
        for u in range(len(user_train)):
            cc += len(user_train[u])
        print('Train average sequence length: %.2f' % (cc / len(user_train)))

        cc = 0.0
        for u in range(len(user_valid)):
            cc += len(user_valid[u])
        print('Valid average sequence length: %.2f' % (cc / len(user_valid)))

        cc = 0.0
        for u in range(len(user_test)):
            cc += len(user_test[u])
        print('Test average sequence length: %.2f' % (cc / len(user_test)))
        
        # import pdb 
        # pdb.set_trace()
        if period == 1:
            continue
            
        if args.inference_only:
            args.state_dict_path = '/data/wangxinru-slurm/project/Rec/SASRec/SASRec/TAOBAO_test/period2/epoch=2.ckpt'
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print(f'load ckpt from {args.state_dict_path} successfully!')
            test_result = evaluate_result(num_test_batch, test_sampler, model)
            print('test (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)' % (test_result[0], test_result[1], test_result[2], test_result[3]))
            exit()
        
        if period > 2 and not args.joint: 
            args.state_dict_path = os.path.join('./' + args.dataset + '_' + args.train_dir, 'period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
            model_dp.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print(f'load ckpt from {args.state_dict_path} successfully!')
        elif period == 2:
            args.state_dict_path = '/data/wangxinru-slurm/project/Rec/SASRec/SASRec/TAOBAO_TAOBAO_FINTUNE/period1/epoch=6.ckpt'
            model_dp.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print(f'load ckpt from {args.state_dict_path} successfully!')
        else:
            init_model(model_dp)

        init_model(model)

        # model_ref = copy.deepcopy(model)
        # for param in model_ref.parameters():
        #     param.requires_grad = False
        # model_ref.eval()

        for param in model_dp.parameters():
            param.requires_grad = False
        model_dp.eval()

        t0 = time.time()
        best_epoch = 1
        stop_counter = 0
        valid_sampler = Sampler(user_valid, args.maxlen, args.test_batch, max_item, is_subseq=True)
        num_valid_batch = valid_sampler.batch_num()
        valid_result = evaluate_result(num_valid_batch, valid_sampler, model)
        print('Valid period: %d  (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)' % (period, valid_result[0], valid_result[1], valid_result[2], valid_result[3]))
        log.write('Valid period: %d (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)\n' % (period, valid_result[0], valid_result[1], valid_result[2], valid_result[3]))
        for epoch in range(1, args.num_epochs + 1):
            print(f'num_batch:{num_batch}')
            model.train()
            for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                seq, pos, neg = train_sampler.sampler()
                seq, pos, neg = np.array(seq), np.array(pos), np.array(neg)
                feature, logits = model(seq, max_item)  # batch x itemnum
                # 构建多分类
                # pos = torch.LongTensor(pos[:, -1]).to(args.device)
                # adam_optimizer.zero_grad()
                # loss = ce_criterion(logits, pos)
                
                # ce
                pos = pos[:, -1]
                pos = torch.LongTensor(pos).to(args.device)
                adam_optimizer.zero_grad()
                loss_ce = ce_criterion(logits, pos)

                # 构建正负样本对 bce
                # batch_indices = torch.arange(logits.size(0), device=logits.device)
                # pos = pos[:, -1]
                # neg = neg[:, -1]
                # pos_logits = logits[batch_indices, pos]
                # neg_logits = logits[batch_indices, neg]
                # pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                # adam_optimizer.zero_grad()
                # loss_bce = bce_criterion(pos_logits, pos_labels)
                # loss_bce += bce_criterion(neg_logits, neg_labels)
                
                # InFoNCE
                feature_dp, logits_dp = model_dp(seq, max_item)  # batch x itemnum
                # loss_nce = InFoNce(feature, feature_dp)
                # loss_nce = InFoNCE(logits, logits_dp)
                # loss = loss_ce + args.alpha * loss_nce
                loss = InFoNCE(logits, logits_dp)

                # mse
                # loss_mse = F.mse_loss(feature, feature_dp)
                # loss_mse = F.mse_loss(logits, logits_dp)
                # loss = loss_ce + args.alpha * loss_mse

                # contrastive_loss
                # loss_contrastive = contrastive_loss(feature, feature_dp, pos)
                # loss_contrastive = contrastive_loss(logits, logits_dp, pos)
                # loss = loss_ce + args.alpha * loss_contrastive

                # dpo
                # logits_ref = model_ref(seq, max_item)  # batch x itemnum
                # logits = F.logsigmoid(logits)
                # logits_ref = F.logsigmoid(logits_ref)
                # batch_indices = torch.arange(logits.size(0), device=logits.device)
                # pos = torch.tensor(pos[:, -1]).to(args.device)
                # # neg = torch.tensor(neg[:, -1]).to(args.device)
                # neg = logits_ref.argmax(dim=-1)
                # valid_mask = pos != neg
                # valid_batch_indices = batch_indices[valid_mask]
                # valid_pos = pos[valid_mask]
                # valid_neg = neg[valid_mask]
                # pos_logits = logits[valid_batch_indices, valid_pos]
                # neg_logits = logits[valid_batch_indices, valid_neg]
                # pos_logits_ref = logits_ref[valid_batch_indices, valid_pos]
                # neg_logits_ref = logits_ref[valid_batch_indices, valid_neg]
                # optimizer.zero_grad()
                # loss = dpo_loss(args.beta, pos_logits, neg_logits, pos_logits_ref, neg_logits_ref)

                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                if step % 50 == 0 or step == num_batch - 1:
                    print("loss in period {} epoch {} iteration {}: {}".format(period, epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

            valid_sampler = Sampler(user_valid, args.maxlen, args.test_batch, max_item, is_subseq=True)
            num_valid_batch = valid_sampler.batch_num()
            valid_result = evaluate_result(num_valid_batch, valid_sampler, model)
            print('Valid period: %d epoch: %d (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)' % (period, epoch, valid_result[0], valid_result[1], valid_result[2], valid_result[3]))
            log.write('Valid period: %d epoch: %d (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)\n' % (period, epoch, valid_result[0], valid_result[1], valid_result[2], valid_result[3]))
            performance = valid_result[3]
            # early stop
            if best_performance >= performance:
                stop_counter += 1
                if stop_counter >= args.stop:
                    break
            else:
                stop_counter = 0
                best_epoch = epoch
                best_performance = performance
                save_model(model, args, period, epoch)

        ckpt_path = './' + args.dataset + '_' + args.train_dir + f'/period{period}/epoch={best_epoch}.ckpt'
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
        test_result = evaluate_result(num_test_batch, test_sampler, model)
        print(num_test_batch)
        print('Test period: %d epoch: %d (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)' % (period, best_epoch, test_result[0], test_result[1], test_result[2], test_result[3]))
        log.write('Test period: %d epoch: %d (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)\n' % (period, best_epoch, test_result[0], test_result[1], test_result[2], test_result[3]))
        MRR_10.append(test_result[0])
        Recall_10.append(test_result[1])
        MRR_20.append(test_result[2])
        Recall_20.append(test_result[3])

    MRR_20, Recall_20, MRR_10, Recall_10 = np.array(MRR_20).mean(), \
                                           np.array(Recall_20).mean(), \
                                           np.array(MRR_10).mean(), \
                                           np.array(Recall_10).mean()  
    log.write('(MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)' % (MRR_10, Recall_10, MRR_20, Recall_20))
    log.close()
    print("Done")
