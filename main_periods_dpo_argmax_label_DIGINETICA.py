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
    folder = './result/' + args.dataset + '_' + args.train_dir + f'/period{period}'
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

def evaluate_result(num_tmp_batch, tmp_sampler, model, max_item):
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
    model.train()
    return MRR_10 / tol_num, RECALL_10 / tol_num, MRR_20 / tol_num, RECALL_20 / tol_num

def dpo_loss(beta, win_logits, lose_logits, win_logits_ref, lose_logits_ref):
    diff = win_logits - lose_logits
    diff_ref = win_logits_ref - lose_logits_ref
    dpo_loss = - F.logsigmoid(beta * (diff - diff_ref))
    return dpo_loss.mean()

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
parser.add_argument('--alpha', default=0.5, type=float)

args = parser.parse_args()
if not os.path.isdir('result/' + args.dataset + '_' + args.train_dir):
    os.makedirs('result/' + args.dataset + '_' + args.train_dir)
with open(os.path.join('result/' + args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    log = open(os.path.join('result/' + args.dataset + '_' + args.train_dir, 'log.txt'), 'a')
    log.write('epoch (val_MRR, val_Recall) (test_MRR, test_Recall)\n')

    if args.dataset == 'DIGINETICA':
        itemnum = 43136    # number of items in DIGINETICA
    elif args.dataset == 'YOOCHOOSE':
        itemnum = 25958    # number of items in YOOCHOOSE
    elif args.dataset == 'TAOBAO':
        itemnum = 681413
    elif args.dataset == 'Amazon_cds':
        itemnum = 16172
    elif args.dataset == 'Amazon_cds_avg':
        itemnum = 16172
    elif args.dataset == 'Amazon_games':
        itemnum = 14622
    else:
        raise ValueError('Invalid dataset name')
    
    model = SASRec_continue_learning(itemnum, args).to(args.device)

    periods = get_periods(args.dataset)
    dataloader = DataLoader(args.dataset)
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
        new_data, info = dataloader.train_loader(period - 1)

        if args.joint and period > 1:
            for p in range(1, period):
                pre_train_sess, info = dataloader.train_loader(p-1)
                new_data.extend(pre_train_sess)

        max_item = dataloader.max_item()
        print(f'Max_item:{max_item}')
        train_sampler = Sampler_pos_neg_seq(new_data, args.maxlen, args.batch_size, max_item)
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
            args.state_dict_path = '/data/wangxinru-slurm/project/Rec/SASRec/SASRec/TAOBAO_continual_learning_dpo_nodropout/period2/epoch=13.ckpt'
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print(f'load ckpt from {args.state_dict_path} successfully!')
            test_result = evaluate_result(num_test_batch, test_sampler, model, max_item)
            print('test (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)' % (test_result[0], test_result[1], test_result[2], test_result[3]))
            exit()
        
        if period > 2 and not args.joint: 
            args.state_dict_path = os.path.join('./result/' + args.dataset + '_' + args.train_dir, 'period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        elif period == 2:
            args.state_dict_path = '/data/wangxinru-slurm/project/Rec/SASRec/SASRec/result/DIGINETICA_fintune_ce_dp_0.3_lr_0.005/period1/epoch=13.ckpt'
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        else:
            init_model(model)

        model_ref = copy.deepcopy(model)
        for param in model_ref.parameters():
            param.requires_grad = False

        valid_sampler = Sampler(user_valid, args.maxlen, args.test_batch, max_item, is_subseq=True)
        num_valid_batch = valid_sampler.batch_num()
        valid_result = evaluate_result(num_valid_batch, valid_sampler, model, max_item)
        print('Valid period: %d (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)' % (period, valid_result[0], valid_result[1], valid_result[2], valid_result[3]))
        log.write('Valid period: %d (MRR@10: %.4f, Recall@10: %.4f, MRR@20: %.4f, Recall@20: %.4f)\n' % (period, valid_result[0], valid_result[1], valid_result[2], valid_result[3]))
        t0 = time.time()
        best_epoch = 1
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            print(f'num_batch:{num_batch}')
            for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                seq, pos, neg = train_sampler.sampler()
                seq, pos, neg = np.array(seq), np.array(pos), np.array(neg)
                feature, logits = model(seq, max_item)  # batch x itemnum
                
                # 构建正负样本对
                # batch_indices = torch.arange(logits.size(0), device=logits.device)
                # pos = torch.tensor(pos[:, -1]).to(args.device)
                # neg = torch.tensor(neg[:, -1]).to(args.device)
                # # neg = logits.argmax(dim=-1)
                # valid_mask = pos != neg
                # valid_batch_indices = batch_indices[valid_mask]
                # valid_pos = pos[valid_mask]
                # valid_neg = neg[valid_mask]
                # pos_logits = logits[valid_batch_indices, valid_pos]
                # neg_logits = logits[valid_batch_indices, valid_neg]
                # pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                # adam_optimizer.zero_grad()
                # loss = dpo_loss(args.beta, pos_logits, neg_logits, pos_logits, neg_logits)
                # loss = bce_criterion(pos_logits, pos_labels)
                # loss += bce_criterion(neg_logits, neg_labels)
                
                feature_ref, logits_ref = model_ref(seq, max_item)  # batch x itemnum
                logits = F.log_softmax(logits, dim=-1)
                logits_ref = F.log_softmax(logits_ref, dim=-1)
                # logits = F.logsigmoid(logits)
                # logits_ref = F.logsigmoid(logits_ref)
                # logits = F.softmax(logits, dim=-1)
                # logits_ref = F.softmax(logits_ref, dim=-1)
                batch_indices = torch.arange(logits.size(0), device=logits.device)
                pos = torch.LongTensor(pos[:, -1]).to(args.device)

                # 单负样本
                neg = torch.tensor(neg[:, -1]).to(args.device)
                # neg = logits.argmax(dim=-1)
                valid_mask = pos != neg
                valid_batch_indices = batch_indices[valid_mask]
                valid_pos = pos[valid_mask]
                valid_neg = neg[valid_mask]
                pos_logits = logits[valid_batch_indices, valid_pos]
                neg_logits = logits[valid_batch_indices, valid_neg]
                pos_logits_ref = logits_ref[valid_batch_indices, valid_pos]
                neg_logits_ref = logits_ref[valid_batch_indices, valid_neg]
                adam_optimizer.zero_grad()
                loss = dpo_loss(args.beta, pos_logits, neg_logits, pos_logits_ref, neg_logits_ref)

                # 多负样本
                # loss_list = []
                # for i in range(-5, 0):  # 最后 5 个负样本
                #     neg_i = torch.tensor(neg[:, i]).to(args.device)

                #     # 合法样本筛选
                #     valid_mask = (pos != neg_i) & (neg_i != 0)
                #     valid_batch_indices = batch_indices[valid_mask]
                #     valid_pos = pos[valid_mask]
                #     valid_neg = neg_i[valid_mask]

                #     # 从 logits 中取出正负样本的得分
                #     pos_logits = logits[valid_batch_indices, valid_pos]
                #     neg_logits = logits[valid_batch_indices, valid_neg]
                #     pos_logits_ref = logits_ref[valid_batch_indices, valid_pos]
                #     neg_logits_ref = logits_ref[valid_batch_indices, valid_neg]

                #     # 计算 DPO Loss
                #     loss_list.append(dpo_loss(args.beta, pos_logits, neg_logits, pos_logits_ref, neg_logits_ref))

                # # 平均多个负样本的 Loss
                # loss = loss_ce + args.alpha * torch.stack(loss_list).mean()
                # loss = loss_ce + args.alpha * loss_dpo

                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                if step % 50 == 0 or step == num_batch - 1:
                    print("loss in period {} epoch {} iteration {}: {}".format(period, epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

            valid_sampler = Sampler(user_valid, args.maxlen, args.test_batch, max_item, is_subseq=True)
            num_valid_batch = valid_sampler.batch_num()
            valid_result = evaluate_result(num_valid_batch, valid_sampler, model, max_item)
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

        ckpt_path = './result/' + args.dataset + '_' + args.train_dir + f'/period{period}/epoch={best_epoch}.ckpt'
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
        test_result = evaluate_result(num_test_batch, test_sampler, model, max_item)
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
