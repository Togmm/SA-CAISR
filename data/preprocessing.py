#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ADER
# @File         : preprocessing.py

import argparse
import os
from util import *
import numpy as np
from collections import defaultdict
import json

def str2bool(v):
    """
    Convert string to boolean
    :param v: string
    :return: boolean True or False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_data(dataset_path):
    """
    Load data from raw dataset.
    :param dataset_path: the full name of dataset including extension name
    :return sess_map: map from raw data session name to session Id, a dictionary sess_map[sess_name]=sessId
    :return item_map: map from raw data item name to item Id, a dictionary item_map[item_name]=itemId
    :return reformed_data: a list: each element is a action, which is a list of [sessId, itemId, time]
    """
    # load data according to file extension name
    filename_extension = dataset_path.split('/')[-1].split('.')[-1]
    if filename_extension == 'dat':
        sess_map, item_map, reformed_data = read_dat(dataset_path)
    elif filename_extension == 'csv':
        sess_map, item_map, reformed_data, item_map_reverse = read_csv(dataset_path)
    elif filename_extension == 'txt':
        sess_map, item_map, reformed_data = read_txt(dataset_path)
    elif filename_extension == 'json':
        sess_map, item_map, reformed_data = read_json(dataset_path)
    else:
        print("Error: new data file type !!!")

    # print raw dataset information
    print('Total number of sessions in dataset:', len(sess_map.keys()))
    print('Total number of items in dataset:', len(item_map.keys()))
    print('Total number of actions in dataset:', len(reformed_data))
    print('Average number of actions per user:', len(reformed_data) / len(sess_map.keys()))
    print('Average number of actions per item:', len(reformed_data) / len(item_map.keys()))

    return sess_map, item_map, reformed_data, item_map_reverse


def short_remove(reformed_data, args):
    """
    Remove data according to threshold
    :param reformed_data: loaded data, a list: each element is a action, which is a list of [sessId, itemId, time]
    :param args: args.threshold_item: minimum number of appearance time of item -1
                 args.threshold_sess: minimum length of session -1
                 args.yoochoose_select: select a most recent fraction of entire dataset
    :return removed_data: result data after removing
    :return sess_end: a map recording session end time, a dictionary sess_end[sessId]=end_time
    """
    org_sess_end = dict()
    for [userId, _, time] in reformed_data:
        org_sess_end = generate_sess_end_map(org_sess_end, userId, time)

    # remove session whose length is 1
    sess_counter = defaultdict(lambda: 0)
    for [userId, _, _] in reformed_data:
        sess_counter[userId] += 1
    removed_data = list(filter(lambda x: sess_counter[x[0]] > 1, reformed_data))

    # remove item which appear less or equal to threshold_item
    item_counter = defaultdict(lambda: 0)
    for [_, itemId, _] in removed_data:
        item_counter[itemId] += 1
    removed_data = list(filter(lambda x: item_counter[x[1]] > args.threshold_item, removed_data))

    # remove session whose length less or equal to threshold_sess
    sess_counter = defaultdict(lambda: 0)
    for [userId, _, _] in removed_data:
        sess_counter[userId] += 1
    removed_data = list(filter(lambda x: sess_counter[x[0]] > args.threshold_sess, removed_data))

    # record session end time
    sess_end = dict()
    for [userId, _, time] in removed_data:
        sess_end = generate_sess_end_map(sess_end, userId, time)

    # if yoochoose dataset, choose a most recent fraction of entire dataset
    if args.yoochoose_select < 1.0 and args.dataset == 'yoochoose-clicks.dat':
        max_time = max(map(lambda x: x[2], removed_data))
        if args.test_fraction == 'day':
            test_threshold = 86400
        elif args.test_fraction == 'week':
            test_threshold = 86400 * 7

        train_session_times = []
        for userId in sess_end.keys():
            if sess_counter[userId] > 1 and sess_end[userId] <= max_time - test_threshold:
                for _ in range(sess_counter[userId]-1):
                    train_session_times.append(sess_end[userId])
        threshold = np.percentile(train_session_times, (1.0 - args.yoochoose_select) * 100.0, interpolation='lower')
        removed_data = list(filter(lambda x: sess_end[x[0]] >= threshold, removed_data))

    # print information of removed data
    print('Number of sessions after pre-processing:', len(set(map(lambda x: x[0], removed_data))))
    print('Number of items after pre-processing:', len(set(map(lambda x: x[1], removed_data))))
    print('Number of actions before pre-processing:', len(reformed_data))
    print('Number of actions after pre-processing:', len(removed_data))
    print('Average number of actions per session:', len(removed_data) / len(set(map(lambda x: x[0], removed_data))))
    print('Average number of actions per item:', len(removed_data) / len(set(map(lambda x: x[1], removed_data))))

    return removed_data, sess_end


def time_partition(removed_data, session_end, args):
    """
    Partition data according to time periods
    :param removed_data: input data, a list: each element is a action, which is a list of [sessId, itemId, time]
    :param session_end: a dictionary recording session end time, session_end[sessId]=end_time
    :param : args: args.test_fraction: time interval for each partition
    :return: time_fraction: a dictionary, the keys are different time periods, value is a list of actions in that
                            time period
    """
    if args.is_time_fraction:
        # split entire dataset by time interval
        time_fraction = dict()
        all_times = np.array(list(session_end.values()))
        max_time = max(all_times)
        min_time = min(all_times)

        if args.dataset == 'train-item-views.csv':
            # for DIGINETICA, choose the most recent 16 fraction and put left dataset in initial set
            if args.test_fraction == 'week':
                period_threshold = np.arange(max_time, min_time, -7 * 86400)
            elif args.test_fraction == 'day':
                period_threshold = np.arange(max_time, min_time, -86400)
            else:
                raise ValueError('invalid time fraction')
            period_threshold = np.sort(period_threshold)
            period_threshold = period_threshold[-17:]

        elif args.dataset == 'yoochoose-clicks.dat':
            # for YOOCHOOSE, choose the earliest 17 fraction
            if args.test_fraction == 'week':
                period_threshold = np.arange(min_time, max_time, 7 * 86400)
            elif args.test_fraction == 'day':
                period_threshold = np.arange(min_time, max_time, 86400)
            else:
                raise ValueError('invalid time fraction')
            period_threshold = np.sort(period_threshold)
            period_threshold = period_threshold[1:]
            period_threshold = period_threshold[:17]

        for [sessId, itemId, time] in removed_data:
            # find period of each action
            if args.dataset == 'yoochoose-clicks.dat' and time > period_threshold[-1]:
                continue
            period = period_threshold.searchsorted(time) + 1
            # generate time period for dictionary keys
            if period not in time_fraction:
                time_fraction[period] = []
            # partition data according to period
            time_fraction[period].append([sessId, itemId, time])
    else:
        # if not partition, put all actions in the last period
        time_fraction = removed_data

    return time_fraction


def generating_txt(time_fraction, sess_end, args):
    """
    Generate final txt file
    :param time_fraction: input data, a dictionary, the keys are different time periods,
                          value is a list of actions in that time period
    :param sess_end: session end time map, sess_map[sessId]=end_time
    :param : args: args.test_fraction: if not split, time interval for test partition
    """

    if args.is_time_fraction:
        # item map second time
        item_map = {}
        for period in sorted(time_fraction.keys()):
            time_fraction[period].sort(key=lambda x: sess_end[x[0]])
        for period in sorted(time_fraction.keys()):
            for i, [userId, itemId, time] in enumerate(time_fraction[period]):
                itemId = generate_name_Id_map(itemId, item_map)
                time_fraction[period][i] = [userId, itemId, time]

        # sort action according to time sequence
        for period in sorted(time_fraction.keys()):
            time_fraction[period].sort(key=lambda x: x[2])

        # generate text file
        for i, period in enumerate(sorted(time_fraction.keys())):
            with open('period_' + str(i) + '.txt', 'w') as file_train:
                for [userId, itemId, time] in time_fraction[period]:
                    file_train.write('%d %d\n' % (userId, itemId))
    else:
        # item map second time
        item_map = {}
        time_fraction.sort(key=lambda x: x[2])
        for i, [userId, itemId, time] in enumerate(time_fraction):
            itemId = generate_name_Id_map(itemId, item_map)
            time_fraction[i] = [userId, itemId, time]

        # sort action according to time sequence
        time_fraction.sort(key=lambda x: x[2])

        max_time = max(map(lambda x: x[2], time_fraction))
        if args.test_fraction == 'day':
            test_threshold = 86400
        elif args.test_fraction == 'week':
            test_threshold = 86400 * 7

        # generate text file
        item_set = set()
        with open('test.txt', 'w') as file_test, open('train.txt', 'w') as file_train:
            for [userId, itemId, time] in time_fraction:
                if sess_end[userId] < max_time - test_threshold:
                    file_train.write('%d %d\n' % (userId, itemId))
                    item_set.add(itemId)
                else:
                    file_test.write('%d %d\n' % (userId, itemId))

def generate_final(info, item_map, item_reverse_map_reverse, write_file):
    
    info.sort(key=lambda x: x[2])
    for i, [userId, itemId, time] in enumerate(info):
        itemId = generate_name_Id_map(itemId, item_map, item_reverse_map_reverse)
        info[i] = [userId, itemId, time]
    info.sort(key=lambda x: x[2])

    # with open(write_file, 'w', encoding='utf-8') as f:
    #     for userId, itemId, time in info:
    #         f.write(f"{int(userId)} {int(itemId)}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='train-item-views.csv', type=str)  # or 'yoochoose-clicks.dat'
    parser.add_argument('--is_time_fraction', default=True, type=str2bool)  # split into different time fraction or not
    parser.add_argument('--test_fraction', default='week', type=str)  # 'day' or 'week'
    parser.add_argument('--threshold_sess', default=2, type=int)  # minimum number of appearance time of item -1
    parser.add_argument('--threshold_item', default=2, type=int)  # minimum length of session -1
    parser.add_argument('--yoochoose_select', default=1.0, type=float)  # select most recent portion in yoochoose
    args = parser.parse_args()
    print('Start preprocess ' + args.dataset + ':')

    # for reproducibility
    SEED = 666
    np.random.seed(SEED)

    # load data and get the session and item lookup table
    os.chdir('data')
    sess_map, item_map, reformed_data, item_map_reverse = read_data(args.dataset)

    # create dictionary for processed data
    if args.dataset.split('.')[0] == 'yoochoose-clicks':
        dataset_name = 'YOOCHOOSE'
    elif args.dataset.split('.')[0] == 'train-item-views':
        dataset_name = 'DIGINETICA'
    elif args.dataset.split('.')[0] == 'Gowalla_totalCheckins':
        dataset_name = 'Gowalla'
    elif args.dataset.split('.')[0] == 'TAOBAO':
        dataset_name = 'TAOBAO2.0'
    elif args.dataset.split('.')[0] == 'yelp_process':
        dataset_name = 'yelp'
    elif args.dataset.split('.')[0] == 'Amazon_cds':
        dataset_name = 'Amazon_cds'
    elif args.dataset.split('.')[0] == 'Amazon_games':
        dataset_name = 'Amazon_games'
    elif args.dataset.split('.')[0] == 'ml-1m':
        dataset_name = 'ml-1m'
    elif args.dataset.split('.')[0] == 'Amazon_Movies_and_TV':
        dataset_name = 'Amazon_Movies_and_TV'
    elif args.dataset.split('.')[0] == 'Amazon_Automotive':
        dataset_name = 'Amazon_Automotive'
    elif args.dataset.split('.')[0] == 'Amazon_Toys_and_Games':
        dataset_name = 'Amazon_Toys_and_Games'
    elif args.dataset.split('.')[0] == 'Amazon_Sports_and_Outdoors':
        dataset_name = 'Amazon_Sports_and_Outdoors_test'
    elif args.dataset.split('.')[0] == 'Electronics':
        dataset_name = 'Amazon_Electronics'
    
    if args.is_time_fraction:
        dataset_name = dataset_name
    else:
        dataset_name = dataset_name + 'joint'
    import pdb; pdb.set_trace()
    if not os.path.isdir(os.path.join('.', dataset_name)):
        os.makedirs(os.path.join('.', dataset_name))
    os.chdir(os.path.join('.', dataset_name))

    reformed_data.sort(key=lambda x: x[2]) 
    total_len = len(reformed_data)
    d0_len = int(total_len * 0.6)
    remain_len = total_len - d0_len
    period_len = remain_len // 4  

    
    D = []
    D.append(reformed_data[:d0_len])
    for i in range(3):
        D.append(reformed_data[d0_len + i * period_len : d0_len + (i + 1) * period_len])
    D.append(reformed_data[d0_len + 3 * period_len :])

    item_map = {}
    item_reverse_map_reverse = {}
    for i in range(5):
        print(f'period:{i}')
        # remove data according to occurrences time
        removed_data, _ = short_remove(D[i], args)
        generate_final(removed_data, item_map, item_reverse_map_reverse, f'/data/{dataset_name}/period_{i}.txt')
    print(len(item_map))

    # idx2item = {v:item_map_reverse[k] for k,v in item_reverse_map_reverse.items()}
    idx2item = {k:item_map_reverse[v] for k,v in item_reverse_map_reverse.items()}
    with open('data/case_study/idx2item.json', 'w', encoding='utf-8') as f:
        json.dump(idx2item, f, ensure_ascii=False, indent=4)
    

    # # partition data according to time periods
    # time_fraction = time_partition(removed_data, sess_end, args)

    # # generate final txt file
    # generating_txt(time_fraction, sess_end, args)

    print(args.dataset + ' finish!')
