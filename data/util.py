#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ADER
# @File         : util.py

import csv
import tqdm
import datetime
import pandas as pd


def generate_name_Id_map(name, map, reverse_map):
    """
    Given a name and map, return corresponding Id. If name not in map, generate a new Id.
    :param name: session or item name in dataset
    :param map: existing map, a dictionary: map[name]=Id
    :return: Id: allocated new Id of the corresponding name
    """
    if name in map:
        Id = map[name]
    else:
        Id = len(map.keys()) + 1
        map[name] = Id
    reverse_map[Id] = name
    return Id


def generate_sess_end_map(sess_end, sessId, time):
    """
    Generate map recording the session end time.
    :param sess_end: the map recording session end time, a dictionary see_end[sessId]=end_time
    :param sessId:session Id of new action
    :param time:time of new action
    :return: sess_end: the map recording session end time, a dictionary see_end[sessId]=end_time
    """
    if sessId in sess_end:
        sess_end[sessId] = max(time, sess_end[sessId])
    else:
        sess_end[sessId] = time
    return sess_end


def read_dat(dataset_path):
    """
    Read .dat type dataset file including MovieLens 1M dataset and Yoochoose dataset
    :param dataset_path: dataset path
    :return: sess_map: map[session name in row dataset]=session Id in system
    :return: item_map: map[item name in row dataset]=item Id in system
    :return: reformed_data: a list: each element is a action, which is a list of [sessId, itemId, time]
    """
    sess_map = {}
    item_map = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    if 'YOOCHOOSE' in dataset_name:
        with open(dataset_path, 'r') as f:
            """ YOOCHOOSE
            """
            for sample in tqdm.tqdm(f, desc='Loading data'):
                sess = sample.split(',')[0]
                item = sample.split(',')[2]
                time = sample.split(',')[1]
                time = int(datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])
    elif 'ml-1m' in dataset_name:
        with open(dataset_path, 'r') as f:
            for sample in tqdm.tqdm(f, desc='Loading data'):
                
                sample = sample.strip()
                sess = sample.split('::')[0]
                item = sample.split('::')[1]
                rate = int(sample.split('::')[2])
                time = sample.split('::')[3]
                if rate >= 3:
                    sessId = generate_name_Id_map(sess, sess_map)
                    itemId = generate_name_Id_map(item, item_map)
                    reformed_data.append([sessId, itemId, time])

    return sess_map, item_map, reformed_data


def read_csv(dataset_path):
    """
    Read .csv type dataset file including MovieLens 20M dataset and DIGINETICA dataset
    :param dataset_path: dataset path
    :return: sess_map: map[session name in row dataset]=session Id in system
    :return: item_map: map[item name in row dataset]=item Id in system
    :return: reformed_data: a list: each element is a action, which is a list of [sessId, itemId, time]
    """
    sess_map = {}
    sess_map_reverse = {}
    item_map = {}
    item_map_reverse = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    with open(dataset_path) as f:

        if 'TAOBAO' in dataset_name:
            reader = csv.DictReader(f, delimiter=',')
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                sess = sample['user_id']
                item = sample['item_id']
                date = sample['time']
                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, date])
        elif 'ml' in dataset_name:
            reader = csv.DictReader(f, delimiter=',')
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                sess = sample['userId']
                item = sample['movieId']
                date = sample['timestamp']
                rate = float(sample['rating'])
                if rate >= 3:
                    sessId = generate_name_Id_map(sess, sess_map)
                    itemId = generate_name_Id_map(item, item_map)
                    reformed_data.append([sessId, itemId, date])
        elif 'Amazon_cds' in dataset_name:
            reader = csv.DictReader(f, delimiter='\t')
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                sess = sample['user_id']
                item = sample['item_id']
                date = sample['time']
                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, date])
        elif dataset_name.split('-')[0] == 'train':
            """ DIGINETICA
            """
            # with sequence information
            reader = csv.DictReader(f, delimiter=';')
            timeframes = []
            for sample in reader:
                timeframes.append(int(sample['timeframe']))
            converter = 86400.00 / max(timeframes)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=';')
            # load data
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                sess = sample['sessionId']
                item = sample['itemId']
                date = sample['eventdate']
                timeframe = int(sample['timeframe'])
                if date:
                    time = int(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()) + timeframe * converter
                else:
                    continue
                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])
        elif 'Amazon_Sports_and_Outdoors' or 'Electronics' in dataset_name:
            reader = csv.reader(f, delimiter=',')
            for i, sample in enumerate(tqdm.tqdm(reader, desc='Loading data')):
                if i == 0:  # 跳过表头
                    continue
                sess = sample[1]
                item = sample[0]
                rate = float(sample[2])
                date = sample[3]
                if rate >= 3:
                    sessId = generate_name_Id_map(sess, sess_map, sess_map_reverse)
                    itemId = generate_name_Id_map(item, item_map, item_map_reverse)
                    reformed_data.append([sessId, itemId, date])
        else:
            print("Error: new csv data file!")
    return sess_map, item_map, reformed_data, item_map_reverse

def read_txt(dataset_path):
    sess_map = {}
    item_map = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    with open(dataset_path) as f:
        for sample in tqdm.tqdm(f, desc='Loading data'):
            sess = sample.split()[0]
            item = sample.split()[4]
            time = sample.split()[1]

            sessId = generate_name_Id_map(sess, sess_map)
            itemId = generate_name_Id_map(item, item_map)
            reformed_data.append([sessId, itemId, time])
    return sess_map, item_map, reformed_data


def read_json(dataset_path):
    sess_map = {}
    item_map = {}
    reformed_data = []
    df = pd.read_json(dataset_path, lines=True)
    for row in df.itertuples():
        sess = row.reviewerID
        item = row.asin
        stars = row.overall
        time = row.reviewTime
        if stars >= 3:
            sessId = generate_name_Id_map(sess, sess_map)
            itemId = generate_name_Id_map(item, item_map)
            reformed_data.append([sessId, itemId, time])

    return sess_map, item_map, reformed_data