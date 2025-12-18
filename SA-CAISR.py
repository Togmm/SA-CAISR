import os
import time
import torch
import argparse
from typing import Union, Iterable, Tuple
from model import SASRec
from utils_0617 import *
import torch.nn.functional as F
from contextlib import contextmanager
import optuna
from optuna.trial import TrialState
from function import *
import psutil

@contextmanager
def apply_fisher_mask(model, mask_dict):
    """
    临时应用 Fisher-guided dropout mask 到 model 参数上，
    forward 完成后自动恢复。
    """
    backup = {}
    with torch.no_grad():
        # 先保存原始参数副本
        for name, param in model.named_parameters():
            if name in mask_dict:
                backup[name] = param.data.clone()
                # 临时替换为 masked 权重
                param.data = param.data * mask_dict[name]
    try:
        yield
    finally:
        with torch.no_grad():
            # 恢复原始权重
            for name, param in model.named_parameters():
                if name in backup:
                    param.data = backup[name]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='DIGINETICA', type=str)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--test_batch', default=64, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=150, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--dropout_rate_train', default=0.0838, type=float)
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
parser.add_argument('--beta', default=0, type=float)
parser.add_argument('--alpha', default=0.9915, type=float)
parser.add_argument('--kneg', default=1, type=int)
parser.add_argument('--probs_sampling', default=False, type=bool) 
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--prune_ratio', default=0.2, type=float)
parser.add_argument('--decay_factor', default=0.21739, type=float)
parser.add_argument('--ema_beta', default=0.821179, type=float)
parser.add_argument('--max_p', default=0.40017, type=float)
parser.add_argument('--topk', default=0, type=int)

args = parser.parse_args()
if not os.path.isdir(f'result/{args.dataset}_tol/' + args.dataset + '_' + args.train_dir):
        os.makedirs(f'result/{args.dataset}_tol/' + args.dataset + '_' + args.train_dir)
with open(os.path.join(f'result/{args.dataset}_tol/' + args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

if __name__ == '__main__':
    last_ckpt_path = None
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    log = open(os.path.join(f'result/{args.dataset}_tol/' + args.dataset + '_' + args.train_dir, 'log.txt'), 'a', buffering=1)
    log.write('epoch (val_MRR, val_Recall) (test_MRR, test_Recall)\n')


    if args.dataset == 'DIGINETICA':
        itemnum = 43136    # number of items in DIGINETICA
    elif args.dataset == 'YOOCHOOSE':
        itemnum = 25958    # number of items in YOOCHOOSE
    elif args.dataset == 'TAOBAO':
        itemnum = 681413
    elif args.dataset == 'Gowalla':
        itemnum = 70760
    elif args.dataset == 'yelp':
        itemnum = 60154
    elif args.dataset == 'ml-1m':
        itemnum = 2884
    elif args.dataset == 'Amazon_Sports_and_Outdoors':
        itemnum = 205843
    else:
        raise ValueError('Invalid dataset name')
    
    model_dp = SASRec(itemnum, args).to(args.device)
    args.dropout_rate = args.dropout_rate_train
    model = SASRec(itemnum, args).to(args.device)


    periods = get_periods(args.dataset)
    dataloader = DataLoader(args.dataset, args.maxlen)
    ce_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    best_epoch, itemnum_prev = 0, 0

    MRR_20 = []
    Recall_20 = []
    NDCG_20 = []
    MRR_10 = []
    Recall_10 = []
    NDCG_10 = []
    
    for period in periods:
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        print('Period %d:' % period)
        best_performance, performance = 0, 0

        new_data, info = dataloader.train_loader(period - 1)
        print(info)

        if args.joint and period > 1:
            for p in range(1, period):
                pre_train_sess = dataloader.train_loader(p-1)
                new_data.extend(pre_train_sess)

        max_item = dataloader.max_item()
        print(f'Max_item:{max_item}')
        train_sampler = Sampler_pos_neg_seq(new_data, args.maxlen, args.batch_size, max_item)
        user_valid, user_train = train_sampler.split_data(valid_portion=0.1, return_train=True)
        num_batch = train_sampler.batch_num()
        # load test data
        user_test, info = dataloader.evaluate_loader(period)
        print(info)
        test_sampler = Sampler(user_test, args.maxlen, args.test_batch, max_item)
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
        
        if period == 1:
            continue
            
        if args.inference_only:
            args.state_dict_path = '/data/wangxinru-slurm/project/Rec/SASRec/SASRec/result/DIGINETICA_fintune_ce_dp_0.3_lr_0.0005_baseline_68_0626/period1/epoch=12.ckpt'
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print(f'load ckpt from {args.state_dict_path} successfully!')
            test_result = evaluate_result(num_test_batch, test_sampler, model, max_item)
            print('test (MRR@10: %.4f, Recall@10: %.4f, NDCG@10: %.4f, MRR@20: %.4f, Recall@20: %.4f, NDCG@20: %.4f)' % (test_result[0], test_result[1], test_result[2], test_result[3], test_result[4], test_result[5]))
            exit()
        
        if period > 2 and not args.joint: 
            args.state_dict_path = os.path.join(f'result/{args.dataset}_tol/' + args.dataset + '_' + args.train_dir, 'period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
            model_dp.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print(f'load ckpt from {args.state_dict_path} successfully!')
        elif period == 2 and not args.joint:
            args.state_dict_path = f'/home/aizoo/data/usershare/wangxinru/SASRec_bp/vision_0.98/SASRec/result/{args.dataset}_fintune_ce_dp_0.3_lr_0.0005_baseline/period1/best.ckpt'
            model_dp.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print(f'load ckpt from {args.state_dict_path} successfully!')
        else:
            init_model(model_dp)

        # for param in model_dp.parameters():
        #     param.requires_grad = False

        t0 = time.time()
        best_epoch = 1
        stop_counter = 0
        valid_sampler = Sampler(user_valid, args.maxlen, args.test_batch, max_item, is_subseq=True)
        num_valid_batch = valid_sampler.batch_num()
        valid_result = evaluate_result(num_valid_batch, valid_sampler, model, max_item)
        print('Valid period: %d  (MRR@10: %.4f, Recall@10: %.4f, NDCG@10: %.4f, MRR@20: %.4f, Recall@20: %.4f, NDCG@20: %.4f)' % (period, valid_result[0], valid_result[1], valid_result[2], valid_result[3], valid_result[4], valid_result[5]))
        log.write('Valid period: %d (MRR@10: %.4f, Recall@10: %.4f, NDCG@10: %.4f, MRR@20: %.4f, Recall@20: %.4f, NDCG@20: %.4f)\n' % (period, valid_result[0], valid_result[1], valid_result[2], valid_result[3], valid_result[4], valid_result[5]))

        fisher_dict = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
        for epoch in range(1, args.num_epochs + 1):
            print(f'num_batch:{num_batch}')
            model.train()
            for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                session, seq, pos, neg = train_sampler.sampler() # single negative samples

                seq, pos, neg = np.array(seq), np.array(pos), np.array(neg)
                
                assert np.max(seq) < itemnum, f"Sequence contains item id {np.max(seq)} >= itemnum {itemnum}"

                feature, logits = model(seq, max_item)  # batch x itemnum
                pos = pos[:, -1]
                pos = torch.as_tensor(pos, dtype=torch.long, device=args.device)

                feature_dp, logits_dp = model_dp(seq, max_item)

                loss_ce_dp = ce_criterion(logits_dp, pos)

                # 注意：这里只取 model_dp 的参数
                named_params_dp = [(n, p) for n, p in model_dp.named_parameters() if p.requires_grad]
                params_dp = [p for _, p in named_params_dp]

                grads_dp = torch.autograd.grad(loss_ce_dp, params_dp, retain_graph=True)

                for (name, p), g in zip(named_params_dp, grads_dp):
                    if g is not None:
                        if name not in fisher_dict:
                            fisher_dict[name] = torch.zeros_like(p.data)
                        fisher_dict[name] = args.ema_beta * fisher_dict[name] + (1 - args.ema_beta) * g.detach().pow(2)

                mask_dict = fisher_weighted_dropout(model_dp, fisher_dict, args.max_p)

                # 在 model_dp 上应用 mask，forward 完成后会恢复
                with apply_fisher_mask(model_dp, mask_dict):
                    # InFoNCE
                    feature_dp, logits_dp = model_dp(seq, max_item)  # batch x itemnum
                    # loss_nce = InFoNCE(logits, logits_dp)
                    loss_nce = HardNegativeInfoNCE(logits, logits_dp, top_k = args.topk)
                    # loss_nce = distillation_loss(logits, logits_dp)

                adam_optimizer.zero_grad()
                loss_ce = ce_criterion(logits, pos)
                loss = loss_ce + args.alpha * loss_nce

                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                if step % 50 == 0 or step == num_batch - 1:
                    print("loss in period {} epoch {} iteration {}: {}".format(period, epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

            for name in fisher_dict:
                fisher_dict[name] *= args.decay_factor
                
            valid_sampler = Sampler(user_valid, args.maxlen, args.test_batch, max_item, is_subseq=True)
            num_valid_batch = valid_sampler.batch_num()
            valid_result = evaluate_result(num_valid_batch, valid_sampler, model, max_item)
            print('Valid period: %d epoch: %d (MRR@10: %.4f, Recall@10: %.4f, NDCG@10: %.4f, MRR@20: %.4f, Recall@20: %.4f, NDCG@20: %.4f)' % (period, epoch, valid_result[0], valid_result[1], valid_result[2], valid_result[3], valid_result[4], valid_result[5]))
            log.write('Valid period: %d epoch: %d (MRR@10: %.4f, Recall@10: %.4f, NDCG@10: %.4f, MRR@20: %.4f, Recall@20: %.4f, NDCG@20: %.4f)\n' % (period, epoch, valid_result[0], valid_result[1], valid_result[2], valid_result[3], valid_result[4], valid_result[5]))
            performance = valid_result[4]
            # early stop
            if best_performance >= performance:
                stop_counter += 1
                if stop_counter >= args.stop:
                    break
            else:
                stop_counter = 0
                best_epoch = epoch
                best_performance = performance
                last_ckpt_path = save_model(model, args, period, epoch, last_ckpt_path)
            
            
        ckpt_path = f'result/{args.dataset}_tol/' + args.dataset + '_' + args.train_dir + f'/period{period}/epoch={best_epoch}.ckpt'
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
        test_result = evaluate_result(num_test_batch, test_sampler, model, max_item)
        print(num_test_batch)
        print('Test period: %d epoch: %d (MRR@10: %.4f, Recall@10: %.4f, NDCG@10: %.4f, MRR@20: %.4f, Recall@20: %.4f, NDCG@20: %.4f)' % (period, best_epoch, test_result[0], test_result[1], test_result[2], test_result[3], test_result[4], test_result[5]))
        log.write('Test period: %d epoch: %d (MRR@10: %.4f, Recall@10: %.4f, NDCG@10: %.4f, MRR@20: %.4f, Recall@20: %.4f, NDCG@20: %.4f)\n' % (period, best_epoch, test_result[0], test_result[1], test_result[2], test_result[3], test_result[4], test_result[5]))
        MRR_10.append(test_result[0])
        Recall_10.append(test_result[1])
        NDCG_10.append(test_result[2])
        MRR_20.append(test_result[3])
        Recall_20.append(test_result[4])
        NDCG_20.append(test_result[5])

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        # 以 MB 为单位
        rss_mem = mem_info.rss / 1024**2   # 常用指标: Resident Set Size
        print(f"Current memory usage: {rss_mem:.2f} MB")
        log.write(f"Current memory usage: {rss_mem:.2f} MB\n")

    MRR_20, Recall_20, NDCG_20, MRR_10, Recall_10, NDCG_10 = np.array(MRR_20).mean(), \
                                           np.array(Recall_20).mean(), \
                                           np.array(NDCG_20).mean(), \
                                           np.array(MRR_10).mean(), \
                                           np.array(Recall_10).mean(), \
                                           np.array(NDCG_10).mean()  
    log.write('(MRR@10: %.4f, Recall@10: %.4f, NDCG@10: %.4f, MRR@20: %.4f, Recall@20: %.4f, NDCG@20: %.4f)' % (MRR_10, Recall_10, NDCG_10, MRR_20, Recall_20, NDCG_20))
    log.close()
    if last_ckpt_path != None:
        os.remove(last_ckpt_path)
    print("Done")
