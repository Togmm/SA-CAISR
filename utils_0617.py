import sys
import copy
import torch
import random
import numpy as np
import os
import re
from collections import defaultdict, deque
from multiprocessing import Process, Queue
import math
from typing import Union, Iterable, Tuple
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.functional as F
import json
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

def create_optimizer_and_scheduler(
        args, model, batch_per_epoch
):

    total_steps = batch_per_epoch // args.gradient_accumulation_steps * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_prop)

    print("Batch per epoch: %d" % batch_per_epoch)
    print("Total steps: %d" % total_steps)
    print("Warmup proportion:", args.warmup_prop)
    print("Warm up steps: %d" % warmup_steps)

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return optimizer, scheduler

class DataLoader_ADER:
    """ DataLoader object to load train, valid and test data from dataset.
        Args:
            dataset (str): Name of the dataset.
    """

    def __init__(self,
                 dataset: str,
                 ) -> None:

        self.item_set = set()
        self.path = os.path.join('./data', dataset)
        # remove item in testing data that not appeared in training data
        self.is_remove_item = True

    def train_loader(self,
                     period: int
                     ) -> (list, str):
        """ Load train data of specific period.
            Args:
                period (int): The period which load training data from.
            Returns:
                sessions (list): Training item sequences (session) of selected periods.
                info (str): Information of training data.
        """
        Sessions = defaultdict(list)
        file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                sessId, itemId = line.rstrip().split(' ')
                sessId = int(sessId)
                itemId = int(itemId)
                self.item_set.add(itemId)
                Sessions[sessId].append(itemId)

        sessions = list(Sessions.values())
        del Sessions
        info = 'Train set information: total number of action: %d.' \
               % sum(list(map(lambda session: len(session), sessions)))
        print(info)

        return sessions, info

    def evaluate_loader(self,
                        period: int,
                        ) -> (list, str):
        """ This method loads test data of specific period.
            Args:
                period (int): The period which load testing data from.
            Returns:
                sessions (list): Testing item sequences (session) of selected periods.
                info (str): Information of testing data.
        """
        Sessions = defaultdict(list)
        removed_num = 0
        total_num = 0
        file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                total_num += 1
                sessId, itemId = line.rstrip().split(' ')
                sessId = int(sessId)
                itemId = int(itemId)
                # remove new items in test or validation set that not appear in train set
                if self.is_remove_item and (itemId not in self.item_set):
                    removed_num += 1
                    continue
                else:
                    self.item_set.add(itemId)
                Sessions[sessId].append(itemId)

        if self.is_remove_item:
            delete_keys = []
            for sessId in Sessions:
                if len(Sessions[sessId]) == 1:
                    removed_num += 1
                    delete_keys.append(sessId)
            for delete_key in delete_keys:
                del Sessions[delete_key]

        info = 'Test set information: original total number of action: %d, removed number of action: %d.' \
               % (total_num, removed_num)
        sessions = list(Sessions.values())
        del Sessions

        return sessions, info

    def max_item(self) -> int:
        """ This method returns the number of accumulative items until current cycle training data.
        """
        return max(self.item_set)

class DataLoader:
    """ DataLoader object to load train, valid and test data from dataset.
        Args:
            dataset (str): Name of the dataset.
    """

    def __init__(self,
                 dataset: str,
                 maxlen: int
                 ) -> None:
        
        self.maxlen = maxlen
        self.item_set = set()
        self.path = os.path.join('./data', dataset)
        # remove item in testing data that not appeared in training data
        self.is_remove_item = True

    def train_loader(self,
                     period: int
                     ) -> (list, str):
        """ Load train data of specific period.
            Args:
                period (int): The period which load training data from.
            Returns:
                sessions (list): Training item sequences (session) of selected periods.
                info (str): Information of training data.
        """
        Sessions = defaultdict(list)
        file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                try:
                    sessId, itemId = line.rstrip().split(' ')
                    sessId = int(sessId)
                    itemId = int(itemId)
                    self.item_set.add(itemId)
                    Sessions[sessId].append(itemId)
                except:
                    print(line)
                    import pdb
                    pdb.set_trace()

        delete_keys = []
        removed_num = 0
        for sessId in Sessions:
            if len(Sessions[sessId]) == 1:
                removed_num += 1
                delete_keys.append(sessId)
            elif len(Sessions[sessId]) > self.maxlen:
                Sessions[sessId] = Sessions[sessId][-self.maxlen:]

        for delete_key in delete_keys:
            del Sessions[delete_key]

        sessions = list(Sessions.values())
        del Sessions
        info = 'Train set information: total number of action: %d, removed number of action: %d.' \
               % (sum(list(map(lambda session: len(session), sessions))), removed_num)
        print(info)
        return sessions, info

    def evaluate_loader(self,
                        period: int,
                        ) -> (list, str):
        """ This method loads test data of specific period.
            Args:
                period (int): The period which load testing data from.
            Returns:
                sessions (list): Testing item sequences (session) of selected periods.
                info (str): Information of testing data.
        """
        Sessions = defaultdict(list)
        removed_num = 0
        total_num = 0
        file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                total_num += 1
                sessId, itemId = line.rstrip().split(' ')
                sessId = int(sessId)
                itemId = int(itemId)
                # remove new items in test or validation set that not appear in train set
                if self.is_remove_item and (itemId not in self.item_set):
                    removed_num += 1
                    continue
                else:
                    self.item_set.add(itemId)
                Sessions[sessId].append(itemId)

        if self.is_remove_item:
            delete_keys = []
            for sessId in Sessions:
                if len(Sessions[sessId]) == 1:
                    removed_num += 1
                    delete_keys.append(sessId)
                elif len(Sessions[sessId]) > self.maxlen:
                    Sessions[sessId] = Sessions[sessId][:self.maxlen]
            for delete_key in delete_keys:
                del Sessions[delete_key]

        info = 'Test set information: original total number of action: %d, removed number of action: %d.' \
               % (total_num, removed_num)
        sessions = list(Sessions.values())
        del Sessions

        return sessions, info

    def max_item(self) -> int:
        """ This method returns the number of accumulative items until current cycle training data.
        """
        return max(self.item_set)

class DataLoader_TAOBAO:
    """ DataLoader object to load train, valid and test data from dataset.
        Args:
            dataset (str): Name of the dataset.
    """

    def __init__(self,
                 dataset: str,
                 maxlen: int,
                 ) -> None:
    
        self.item_set = set()
        self.item_num = 0
        self.item_dict = {}
        self.path = os.path.join('./data', dataset)
        self.max_len = maxlen
        # remove item in testing data that not appeared in training data
        self.is_remove_item = True

    def train_loader(self,
                     period: int
                     ) -> (list, str):
        """ Load train data of specific period.
            Args:
                period (int): The period which load training data from.
            Returns:
                sessions (list): Training item sequences (session) of selected periods.
                info (str): Information of training data.
        """
        Sessions = defaultdict(list)
        file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                sessId, itemId = line.rstrip().split()

                if itemId not in self.item_dict:
                    self.item_num += 1
                    self.item_dict[itemId] = self.item_num

                itemId = self.item_dict[itemId]
                
                self.item_set.add(itemId)
                Sessions[sessId].append(itemId)

        sessions = list(Sessions.values())
        for idx, session in enumerate(sessions):
            if len(session) > self.max_len:
                sessions[idx] = session[-self.max_len:]
        del Sessions
        info = 'Train set information: total number of action: %d.' \
               % sum(list(map(lambda session: len(session), sessions)))
        print(info)

        return sessions

    def evaluate_loader(self,
                        period: int,
                        ) -> (list, str):
        """ This method loads test data of specific period.
            Args:
                period (int): The period which load testing data from.
            Returns:
                sessions (list): Testing item sequences (session) of selected periods.
                info (str): Information of testing data.
        """
        Sessions = defaultdict(list)
        removed_num = 0
        total_num = 0
        file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                total_num += 1
                sessId, itemId = line.rstrip().split()

                if itemId in self.item_dict:
                    itemId = self.item_dict[itemId]
                elif self.is_remove_item:
                    removed_num += 1
                    continue
                Sessions[sessId].append(itemId)

        if self.is_remove_item:
            delete_keys = []
            for sessId in Sessions:
                if len(Sessions[sessId]) == 1:
                    removed_num += 1
                    delete_keys.append(sessId)
            for delete_key in delete_keys:
                del Sessions[delete_key]

        info = 'Test set information: original total number of action: %d, removed number of action: %d.' \
               % (total_num, removed_num)
        sessions = list(Sessions.values())
        for idx, session in enumerate(sessions):
            if len(session) > self.max_len:
                sessions[idx] = session[-self.max_len:]
        del Sessions

        return sessions, info

    def max_item(self) -> int:
        """ This method returns the number of accumulative items until current cycle training data.
        """
        return self.item_num

class DataLoader_DPO:
    """ DataLoader object to load train, valid and test data from dataset.
        Args:
            dataset (str): Name of the dataset.
    """

    def __init__(self,
                 dataset: str,
                 maxlen: int,
                 ) -> None:
        self.item_num = 0
        self.item_dict = {}
        self.Sessions = defaultdict(lambda: {
            'deque': deque(maxlen=maxlen),
            'lastperiod': 0,
            'lastlength': 0,
            'is_newseq': 0,
        })
        
        self.item_set = set()
        self.path = os.path.join('./data', dataset)
        # remove item in testing data that not appeared in training data
        self.is_remove_item = True
        self.maxlen = maxlen

    def train_loader(self,
                     period: int
                     ):
        """ Load train data of specific period.
            Args:
                period (int): The period which load training data from.
            Returns:
                sessions (list): Training item sequences (session) of selected periods.
                info (str): Information of training data.
        """
        st = set()
        file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                sessId, itemId = line.rstrip().split()
                if itemId not in self.item_dict:
                    self.item_num += 1
                    self.item_dict[itemId] = self.item_num

                itemId = self.item_dict[itemId]
                self.item_set.add(itemId)
                if sessId in st:
                    # if self.Sessions[sessId]['lastlength'] >= self.maxlen // 2:
                    #     continue
                    self.Sessions[sessId]['lastlength'] += 1
                else:
                    if sessId in self.Sessions:
                        self.Sessions[sessId]['is_newseq'] = 0
                    else:
                        self.Sessions[sessId]['is_newseq'] = 1
                    st.add(sessId)
                    self.Sessions[sessId]['lastlength'] = 1
            
                self.Sessions[sessId]['deque'].append(itemId)
                self.Sessions[sessId]['lastperiod'] = period
   
        need_data = []
        for key in self.Sessions.keys():
            session = self.Sessions[key]
            if session['lastperiod'] == period:
                tmp_session = []
                tmp_session.append(session['is_newseq'])
                tmp_session.append(session['lastlength'])
                tmp_session.append(list(session['deque']))
                need_data.append(tmp_session)

        return need_data

    def evaluate_loader(self,
                        period: int,
                        ) -> (list, str):
        """ This method loads test data of specific period.
            Args:
                period (int): The period which load testing data from.
            Returns:
                sessions (list): Testing item sequences (session) of selected periods.
                info (str): Information of testing data.
        """
        Sessions = defaultdict(list)
        removed_num = 0
        total_num = 0
        file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                total_num += 1
                sessId, itemId = line.rstrip().split()

                if itemId in self.item_dict:
                    itemId = self.item_dict[itemId]
                elif self.is_remove_item:
                    removed_num += 1
                    continue
                Sessions[sessId].append(itemId)

        if self.is_remove_item:
            delete_keys = []
            for sessId in Sessions:
                if len(Sessions[sessId]) == 1:
                    removed_num += 1
                    delete_keys.append(sessId)
            for delete_key in delete_keys:
                del Sessions[delete_key]

        info = 'Test set information: original total number of action: %d, removed number of action: %d.' \
               % (total_num, removed_num)
        sessions = list(Sessions.values())
        for idx, session in enumerate(sessions):
            if len(session) > self.maxlen:
                sessions[idx] = session[-self.maxlen:]
        del Sessions

        return sessions, info

    def max_item(self) -> int:
        """ This method returns the number of accumulative items until current cycle training data.
        """
        return self.item_num

class Sampler:
    """ This object samples data and generates positive labels for train, valid and test data,
            as well as negative sample for training data.
        Args:
            data (list): Original data needs to be sampled.
            maxlen (int): The input length of each sequence for the model.
            batch_size (int): The number of data in one batch.
            is_subseq (bool): If True, the given data is sub-sequence. If False, the given data is full
                original data.
    """

    def __init__(self,
                 data: list,
                 maxlen: int,
                 batch_size: int,
                 max_item: int,
                 is_subseq: bool = False
                 ) -> None:

        self.maxlen = maxlen
        self.batch_size = batch_size

        self.dataset_size = 0
        self.batch_counter = 0
        self.data_indices = []
        self.logits = []

        self.prepared_data = []
        if not is_subseq:
            for session in data:
                self.prepared_data.append(session)
                length = len(session)
                if length > 2:
                    for t in range(1, length - 1):
                        self.prepared_data.append(session[:-t])
        else:
            for session in data:
                self.prepared_data.append(session)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def label_generator(self,
                        session: list,
                        ) -> (list, int):
        """ This method split sessions into input sequence and labels.
            Args:
                session (list): Original sub-sequence of different length.
            Return:
                seq (list): The input sequence with fixed length set by maxlen.
                pos (int): Label (item number).
        """
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.array(session[-1], dtype=np.int32)
        idx = self.maxlen - 1

        for itemId in reversed(session[:-1]):
            seq[idx] = itemId
            idx -= 1
            if idx == -1:
                break

        return seq, pos

    def add_exemplar(self,
                     exemplar: list
                     ) -> None:
        """ Add exemplar data and logits from previous cycle model
            Args:
                 exemplar (list): Exemplar data and corresponding logits.
        """
        self.logits = []
        for session, logits in exemplar:
            self.prepared_data.append(session)
            self.logits.append(logits)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def split_data(self,
                   valid_portion: float,
                   return_train: bool = False
                   ) -> Union[list, tuple]:
        """ Split data into valid and train dataset and remove validation data from original training data.
            Args:
                valid_portion (float): The portion of validation dataset w.r.t entire dataset.
                return_train: If True, return validation data and train data, else only return validation data.
            Returns:
                valid_data (list): Validation sub-sequence.
                train_data (list): Training sub-sequence.
        """

        data_size = len(self.prepared_data)
        sidx = np.arange(data_size, dtype='int32')
        np.random.shuffle(sidx)

        n_train = int(np.round(data_size * (1. - valid_portion)))
        valid_data = [self.prepared_data[s] for s in sidx[n_train:]]
        train_data = [self.prepared_data[s] for s in sidx[:n_train]]
        self.prepared_data = train_data

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

        if return_train:
            return valid_data, train_data
        else:
            return valid_data

    def sampler(self) -> Union[Iterable[Tuple[list, int]]]:
        """ This method returns a batch of sample: N * (sequence, label).
            Returns:
                one_batch (list): One batch of data in the size of N * (sequence length, 1).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                one_batch.append(self.label_generator(session))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def exemplar_sampler(self) -> Iterable[Tuple[list, int, list]]:
        """ This method returns a batch of exemplar data: N * (exemplar, logits).
            Return:
                one_batch (list): One batch of data in the size of N * (sequence length, previous item number).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                seq, pos = self.label_generator(session)
                one_batch.append((seq, pos, self.logits[index]))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def data_size(self) -> int:
        """ Returns the number of sub-sequences in the data set.
        """
        return len(self.prepared_data)

    def batch_num(self) -> int:
        """ Returns the number of batches according to dataset size and batch size.
        """
        return math.ceil(len(self.prepared_data) * 1.0 / self.batch_size)

class Sampler_pos_neg_item:
    """ This object samples data and generates positive labels for train, valid and test data,
            as well as negative sample for training data.
        Args:
            data (list): Original data needs to be sampled.
            maxlen (int): The input length of each sequence for the model.
            batch_size (int): The number of data in one batch.
            is_subseq (bool): If True, the given data is sub-sequence. If False, the given data is full
                original data.
    """
    def __init__(self,
                 data: list,
                 maxlen: int,
                 batch_size: int,
                 max_item: int,
                 is_subseq: bool = False
                 ) -> None:

        self.max_item = max_item
        self.maxlen = maxlen
        self.batch_size = batch_size

        self.dataset_size = 0
        self.batch_counter = 0
        self.data_indices = []
        self.logits = []
        self.prepared_data = []
        if not is_subseq:
            for session in data:
                length = len(session)
                self.prepared_data.append(session)
                if length > 2:
                    for t in range(1, length - 1):
                        self.prepared_data.append(session[:-t])
        else:
            for session in data:
                self.prepared_data.append(session)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def label_generator(self,
                        session: list,
                        ) -> (list, int):
        """ This method split sessions into input sequence and labels.
            Args:
                session (list): Original sub-sequence of different length.
            Return:
                seq (list): The input sequence with fixed length set by maxlen.
                pos (int): Label (item number).
        """
        ts = set(session)
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.array(session[-1], dtype=np.int32)
        neg = random_neq(1, self.max_item + 1, ts)
        idx = self.maxlen - 1

        for itemId in reversed(session[:-1]):
            seq[idx] = itemId
            idx -= 1
            if idx == -1:
                break

        return seq, pos, neg

    def add_exemplar(self,
                     exemplar: list
                     ) -> None:
        """ Add exemplar data and logits from previous cycle model
            Args:
                 exemplar (list): Exemplar data and corresponding logits.
        """
        self.logits = []
        for session, logits in exemplar:
            self.prepared_data.append(session)
            self.logits.append(logits)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def split_data(self,
                   valid_portion: float,
                   return_train: bool = False
                   ) -> Union[list, tuple]:
        """ Split data into valid and train dataset and remove validation data from original training data.
            Args:
                valid_portion (float): The portion of validation dataset w.r.t entire dataset.
                return_train: If True, return validation data and train data, else only return validation data.
            Returns:
                valid_data (list): Validation sub-sequence.
                train_data (list): Training sub-sequence.
        """
        
        data_size = len(self.prepared_data)
        sidx = np.arange(data_size, dtype='int32')
        np.random.shuffle(sidx)

        n_train = int(np.round(data_size * (1. - valid_portion)))
        valid_data = [self.prepared_data[s] for s in sidx[n_train:]]
        train_data = [self.prepared_data[s] for s in sidx[:n_train]]
        self.prepared_data = train_data

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

        if return_train:
            return valid_data, train_data
        else:
            return valid_data

    def sampler(self) -> Union[Iterable[Tuple[list, int]]]:
        """ This method returns a batch of sample: N * (sequence, label).
            Returns:
                one_batch (list): One batch of data in the size of N * (sequence length, 1).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                one_batch.append(self.label_generator(session))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def exemplar_sampler(self) -> Iterable[Tuple[list, int, list]]:
        """ This method returns a batch of exemplar data: N * (exemplar, logits).
            Return:
                one_batch (list): One batch of data in the size of N * (sequence length, previous item number).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                seq, pos = self.label_generator(session)
                one_batch.append((seq, pos, self.logits[index]))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def data_size(self) -> int:
        """ Returns the number of sub-sequences in the data set.
        """
        return len(self.prepared_data)

    def batch_num(self) -> int:
        """ Returns the number of batches according to dataset size and batch size.
        """
        return math.ceil(len(self.prepared_data) * 1.0 / self.batch_size)

def random_neq_p(values, probs, s):
    # t = np.random.randint(l, r)
    # print(sum(probs))
    t = np.random.choice(values, p=probs)
    while t in s:
        # t = np.random.randint(l, r)
        t = np.random.choice(values, p=probs)
    return t

def temperature_softmax(log_probs, temperature):
    scaled_log_probs = np.array(log_probs) / temperature
    exp_probs = np.exp(scaled_log_probs - np.max(scaled_log_probs)).astype('float64')  # Subtract max for numerical stability
    softmax_probs = exp_probs / np.sum(exp_probs)
    return softmax_probs


class Sampler_pos_neg_seq:
    """ This object samples data and generates positive labels for train, valid and test data,
            as well as negative sample for training data.
        Args:
            data (list): Original data needs to be sampled.
            maxlen (int): The input length of each sequence for the model.
            batch_size (int): The number of data in one batch.
            is_subseq (bool): If True, the given data is sub-sequence. If False, the given data is full
                original data.
    """
    def __init__(self,
                 data: list,
                 maxlen: int,
                 batch_size: int,
                 max_item: int,
                 is_subseq: bool = False
                 ) -> None:

        self.maxlen = maxlen
        self.batch_size = batch_size
        self.max_item = max_item
        self.dataset_size = 0
        self.batch_counter = 0
        self.data_indices = []
        self.logits = []
        self.prepared_data = []
        if not is_subseq:
            for session in data:
                length = len(session)
                self.prepared_data.append(session)
                if length > 2:
                    for t in range(1, length - 1):
                        self.prepared_data.append(session[:-t])
        else:
            for session in data:
                length = len(session)
                self.prepared_data.append(session)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def label_generator(self,
                        session: list,
                        ) -> (list, int):
        """ This method split sessions into input sequence and labels.
            Args:
                session (list): Original sub-sequence of different length.
            Return:
                seq (list): The input sequence with fixed length set by maxlen.
                pos (int): Label (item number).
        """
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = session[-1]
        idx = self.maxlen - 1

        ts = set(session)
        for i in reversed(session[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, self.max_item + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return session, seq, pos, neg

    def neg_generator_p(self,
                        session: list,
                        t: float,
                        probs: list,
                        ) -> (list, int):
        """ This method split sessions into input sequence and labels.
            Args:
                session (list): Original sub-sequence of different length.
            Return:
                seq (list): The input sequence with fixed length set by maxlen.
                pos (int): Label (item number).
        """

        negs = []

        # for ses, prob in zip(session, probs):
        #     prob = temperature_softmax(prob[1:], 1)
        #     neg = np.zeros([self.maxlen], dtype=np.int32)
        #     nxt = ses[-1]
        #     idx = self.maxlen - 1
        #     ts = set(ses)
        #     for i in reversed(ses[:-1]):
        #         if nxt != 0: neg[idx] = random_neq_p(np.arange(1, self.max_item + 1), prob.tolist(), ts)
        #         nxt = i
        #         idx -= 1
        #         if idx == -1: break
        #     negs.append(neg)

        for ses, prob in zip(session, probs):
            prob = temperature_softmax(prob, t)
            ts = set(ses)
            neg = random_neq_p(np.arange(1, self.max_item + 1), prob.tolist(), ts)
            negs.append(neg)
        return negs


    def add_exemplar(self,
                     exemplar: list
                     ) -> None:
        """ Add exemplar data and logits from previous cycle model
            Args:
                 exemplar (list): Exemplar data and corresponding logits.
        """
        self.logits = []
        for session, logits in exemplar:
            self.prepared_data.append(session)
            self.logits.append(logits)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def split_data(self,
                   valid_portion: float,
                   return_train: bool = False
                   ) -> Union[list, tuple]:
        """ Split data into valid and train dataset and remove validation data from original training data.
            Args:
                valid_portion (float): The portion of validation dataset w.r.t entire dataset.
                return_train: If True, return validation data and train data, else only return validation data.
            Returns:
                valid_data (list): Validation sub-sequence.
                train_data (list): Training sub-sequence.
        """
        
        data_size = len(self.prepared_data)
        sidx = np.arange(data_size, dtype='int32')
        np.random.shuffle(sidx)

        n_train = int(np.round(data_size * (1. - valid_portion)))
        valid_data = [self.prepared_data[s] for s in sidx[n_train:]]
        train_data = [self.prepared_data[s] for s in sidx[:n_train]]
        self.prepared_data = train_data
        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

        if return_train:
            return valid_data, train_data
        else:
            return valid_data

    def sampler_kneg(self, k) -> Union[Iterable[Tuple[list, int]]]:
        """ This method returns a batch of sample: N * (sequence, label).
            Returns:
                one_batch (list): One batch of data in the size of N * (sequence length, 1).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                for i in range(k):
                    one_batch.append(self.label_generator(session))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def sampler(self) -> Union[Iterable[Tuple[list, int]]]:
        """ This method returns a batch of sample: N * (sequence, label).
            Returns:
                one_batch (list): One batch of data in the size of N * (sequence length, 1).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                one_batch.append(self.label_generator(session))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def exemplar_sampler(self) -> Iterable[Tuple[list, int, list]]:
        """ This method returns a batch of exemplar data: N * (exemplar, logits).
            Return:
                one_batch (list): One batch of data in the size of N * (sequence length, previous item number).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                seq, pos = self.label_generator(session)
                one_batch.append((seq, pos, self.logits[index]))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def data_size(self) -> int:
        """ Returns the number of sub-sequences in the data set.
        """
        return len(self.prepared_data)

    def batch_num(self) -> int:
        """ Returns the number of batches according to dataset size and batch size.
        """
        return math.ceil(len(self.prepared_data) * 1.0 / self.batch_size)

class Sampler_DPO_sequence:
    """ This object samples data and generates positive labels for train, valid and test data,
            as well as negative sample for training data.
        Args:
            data (list): Original data needs to be sampled.
            maxlen (int): The input length of each sequence for the model.
            batch_size (int): The number of data in one batch.
            is_subseq (bool): If True, the given data is sub-sequence. If False, the given data is full
                original data.
    """
    def __init__(self,
                 data: list,
                 maxlen: int,
                 batch_size: int,
                 is_subseq: bool = False
                 ) -> None:

        self.maxlen = maxlen
        self.batch_size = batch_size

        self.dataset_size = 0
        self.batch_counter = 0
        self.data_indices = []
        self.logits = []

        self.prepared_data = []
        if not is_subseq:
            for session in data:
                self.prepared_data.append(session)
                length = len(session)
                if length > 2:
                    for t in range(1, length - 1):
                        self.prepared_data.append(session[:-t])
        else:
            for session in data:
                if session[1] < 2:  #如果新序列只有一个
                    continue
                self.prepared_data.append(session)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def label_generator(self,
                        session: list,
                        ) -> (list, int):
        """ This method split sessions into input sequence and labels.
            Args:
                session (list): Original sub-sequence of different length.
            Return:
                seq (list): The input sequence with fixed length set by maxlen.
                pos (int): Label (item number).
        """
        ts = set(session[2])
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        idx = self.maxlen - 1
        nxt = session[2][-1]
        for itemId in reversed(session[2][:-1]):
            seq[idx] = itemId
            pos[idx] = nxt
            nxt = itemId
            idx -= 1
            if idx == -1:
                break
        return session[0], session[1], seq, pos

    def add_exemplar(self,
                     exemplar: list
                     ) -> None:
        """ Add exemplar data and logits from previous cycle model
            Args:
                 exemplar (list): Exemplar data and corresponding logits.
        """
        self.logits = []
        for session, logits in exemplar:
            self.prepared_data.append(session)
            self.logits.append(logits)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def split_data(self,
                   valid_portion: float,
                   return_train: bool = False
                   ) -> Union[list, tuple]:
        """ Split data into valid and train dataset and remove validation data from original training data.
            Args:
                valid_portion (float): The portion of validation dataset w.r.t entire dataset.
                return_train: If True, return validation data and train data, else only return validation data.
            Returns:
                valid_data (list): Validation sub-sequence.
                train_data (list): Training sub-sequence.
        """

        data_size = len(self.prepared_data)
        sidx = np.arange(data_size, dtype='int32')
        np.random.shuffle(sidx)

        n_train = int(np.round(data_size * (1. - valid_portion)))
        valid_data = [self.prepared_data[s][2] for s in sidx[n_train:]]
        train_data = [self.prepared_data[s] for s in sidx[:n_train]]
        self.prepared_data = train_data

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

        if return_train:
            return valid_data, train_data
        else:
            return valid_data

    def sampler(self) -> Union[Iterable[Tuple[list, int]]]:
        """ This method returns a batch of sample: N * (sequence, label).
            Returns:
                one_batch (list): One batch of data in the size of N * (sequence length, 1).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                one_batch.append(self.label_generator(session))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def exemplar_sampler(self) -> Iterable[Tuple[list, int, list]]:
        """ This method returns a batch of exemplar data: N * (exemplar, logits).
            Return:
                one_batch (list): One batch of data in the size of N * (sequence length, previous item number).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                seq, pos = self.label_generator(session)
                one_batch.append((seq, pos, self.logits[index]))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def data_size(self) -> int:
        """ Returns the number of sub-sequences in the data set.
        """
        return len(self.prepared_data)

    def batch_num(self) -> int:
        """ Returns the number of batches according to dataset size and batch size.
        """
        return math.ceil(len(self.prepared_data) * 1.0 / self.batch_size)

class Sampler_DPO_add_old_sequence:
    """ This object samples data and generates positive labels for train, valid and test data,
            as well as negative sample for training data.
        Args:
            data (list): Original data needs to be sampled.
            maxlen (int): The input length of each sequence for the model.
            batch_size (int): The number of data in one batch.
            is_subseq (bool): If True, the given data is sub-sequence. If False, the given data is full
                original data.
    """
    def __init__(self,
                 data: list,
                 maxlen: int,
                 batch_size: int,
                 max_item: int,
                 equal: bool,
                 ) -> None:

        self.maxlen = maxlen
        self.batch_size = batch_size
        self.max_item = max_item
        self.dataset_size = 0
        self.batch_counter = 0
        self.data_indices = []
        self.logits = []

        self.prepared_data = []
        
        if not equal:
            for session in data:
                self.prepared_data.append(session[2])
                length = min(len(session[2]), session[1])
                if length > 2:
                    if length < len(session[2]):
                        for t in range(1, length):
                            self.prepared_data.append(session[2][:-t])
                    else:
                        for t in range(1, length - 1):
                            self.prepared_data.append(session[2][:-t])
        else:
            for session in data:
                length = min(len(session[2]), session[1])
                length = min(length, max(0, len(session[2]) - session[1]))
                if length >= 1:
                    left = len(session[2]) - session[1] - length
                    right= len(session[2]) - session[1] + length
                    self.prepared_data.append(session[2][left : right])
                    if length >= 2:
                        for x in range(1, length):
                            self.prepared_data.append(session[2][left : right - x])
                        
            

        
        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def label_generator(self,
                        session: list,
                        ) -> (list, int):
        """ This method split sessions into input sequence and labels.
            Args:
                session (list): Original sub-sequence of different length.
            Return:
                seq (list): The input sequence with fixed length set by maxlen.
                pos (int): Label (item number).
        """
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = session[-1]
        idx = self.maxlen - 1

        ts = set(session)
        for i in reversed(session[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, self.max_item + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return seq, pos, neg

    def add_exemplar(self,
                     exemplar: list
                     ) -> None:
        """ Add exemplar data and logits from previous cycle model
            Args:
                 exemplar (list): Exemplar data and corresponding logits.
        """
        self.logits = []
        for session, logits in exemplar:
            self.prepared_data.append(session)
            self.logits.append(logits)

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

    def split_data(self,
                   valid_portion: float,
                   return_train: bool = False
                   ) -> Union[list, tuple]:
        """ Split data into valid and train dataset and remove validation data from original training data.
            Args:
                valid_portion (float): The portion of validation dataset w.r.t entire dataset.
                return_train: If True, return validation data and train data, else only return validation data.
            Returns:
                valid_data (list): Validation sub-sequence.
                train_data (list): Training sub-sequence.
        """
        data_size = len(self.prepared_data)
        sidx = np.arange(data_size, dtype='int32')
        np.random.shuffle(sidx)

        n_train = int(np.round(data_size * (1. - valid_portion)))
        valid_data = [self.prepared_data[s] for s in sidx[n_train:]]
        train_data = [self.prepared_data[s] for s in sidx[:n_train]]
        self.prepared_data = train_data

        self.data_indices = list(range(len(self.prepared_data)))
        random.shuffle(self.data_indices)

        if return_train:
            return valid_data, train_data
        else:
            return valid_data

    def sampler(self) -> Union[Iterable[Tuple[list, int]]]:
        """ This method returns a batch of sample: N * (sequence, label).
            Returns:
                one_batch (list): One batch of data in the size of N * (sequence length, 1).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                one_batch.append(self.label_generator(session))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def exemplar_sampler(self) -> Iterable[Tuple[list, int, list]]:
        """ This method returns a batch of exemplar data: N * (exemplar, logits).
            Return:
                one_batch (list): One batch of data in the size of N * (sequence length, previous item number).
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < len(self.prepared_data):
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                seq, pos = self.label_generator(session)
                one_batch.append((seq, pos, self.logits[index]))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num():
            self.batch_counter = 0
            random.shuffle(self.data_indices)

        return zip(*one_batch)

    def data_size(self) -> int:
        """ Returns the number of sub-sequences in the data set.
        """
        return len(self.prepared_data)

    def batch_num(self) -> int:
        """ Returns the number of batches according to dataset size and batch size.
        """
        return math.ceil(len(self.prepared_data) * 1.0 / self.batch_size)


def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))

def sample_function_baseline(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):        
        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(0, usernum)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(0, usernum, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_baseline, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition_baseline_tol(fname):
    # 获取所有 period_X.txt 文件
    period_files = []
    for filename in os.listdir(fname):
        if re.match(r"period_\d+\.txt", filename):  # 匹配 period_X.txt 格式
            period_files.append(os.path.join(fname, filename))

    itemnum = 0
    valid_test_tol = []
    user_train = []
    user_valid = []
    user_test = []
    for file in period_files:
        data = []
        # assume user/item index starting from 1
        # f = open(file, 'r')
        with open(file, 'r') as f:
            lines = f.readlines()
        
        User = defaultdict(list)
        for line in lines:
            u, i = line.rstrip().split()
            u = int(u)
            i = int(i)
            # usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback >= 2:
                data.append(User[user])
        
        if '16' in file:
            split_idx = int(len(data) * 0.8)
        else:
            split_idx = len(data)

        train_data = data[:split_idx]
        valid_test_data = data[split_idx:]

        user_train.extend(train_data)
        valid_test_tol.extend(valid_test_data)

    for data in valid_test_tol:
        user_valid.append(data[:-1])
        user_test.append(data[-1])
    
    # return [user_train_tol, user_valid_tol, user_test_tol, usernum, itemnum]
    return [user_train, len(user_train), itemnum], [user_valid, user_test, len(user_valid), itemnum] 


# train/val/test data generation
def data_partition_baseline_pre15(fname):
    # 获取所有 period_X.txt 文件
    period_files = []
    for filename in os.listdir(fname):
        if re.match(r"period_\d+\.txt", filename):  # 匹配 period_X.txt 格式
            period_files.append(os.path.join(fname, filename))

    itemnum = 0
    user_train = []

    for file in period_files:
        data = []
        # assume user/item index starting from 1
        # f = open(file, 'r')
        if '16' in file:
            continue

        with open(file, 'r') as f:
            lines = f.readlines()
        
        User = defaultdict(list)
        for line in lines:
            u, i = line.rstrip().split()
            u = int(u)
            i = int(i)
            # usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback >= 2:
                data.append(User[user])

        train_data = data
        user_train.extend(train_data)

    # return [user_train_tol, user_valid_tol, user_test_tol, usernum, itemnum]
    return [user_train, len(user_train), 43136]

# train/val/test data generation
def data_partition_baseline_id16(fname):
    # 获取所有 period_X.txt 文件
    period_files = []
    for filename in os.listdir(fname):
        if re.match(r"period_0.txt", filename):  # 匹配 period_X.txt 格式
            period_files.append(os.path.join(fname, filename))

    itemnum = 0
    valid_test_tol = []
    user_train = []
    user_valid = []
    user_test = []
    for file in period_files:
        data = []
        # assume user/item index starting from 1
        # f = open(file, 'r')
        with open(file, 'r') as f:
            lines = f.readlines()
        
        User = defaultdict(list)
        for line in lines:
            u, i = line.rstrip().split()
            u = int(u)
            i = int(i)
            # usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback >= 2:
                data.append(User[user])
        
        if '16' in file:
            split_idx = int(len(data) * 0.8)
        else:
            split_idx = len(data)

        train_data = data[:split_idx]
        valid_test_data = data[split_idx:]

        user_train.extend(train_data)
        valid_test_tol.extend(valid_test_data)

    for data in valid_test_tol:
        user_valid.append(data[:-1])
        user_test.append(data[-1])
    
    # return [user_train_tol, user_valid_tol, user_test_tol, usernum, itemnum]
    return [user_train, len(user_train), itemnum], [user_valid, user_test, len(user_valid), itemnum] 


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split()
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate_continue_learning(seqs, predictions, max_item, pos):
    test_user = 0.0
    MRR_10, RECALL_10, MRR_20, RECALL_20 = 0.0, 0.0, 0.0, 0.0
    NDCG_10, NDCG_20 = 0.0,0.0
    for seq, prediction, label in zip(seqs, predictions, pos):
        # if label in seq:
        #     continue
        # for item in seq:
        #     prediction[item] = np.inf
        
        rank = prediction.argsort().argsort()[label].item()

        test_user += 1
        # print('label', label, rank)
        if rank < 10:
            RECALL_10 += 1
            MRR_10 += 1 / (rank + 1)
            NDCG_10 += 1 / np.log2(rank + 2)
        
        if rank < 20:
            RECALL_20 += 1
            MRR_20 += 1 / (rank + 1)
            NDCG_20 += 1 / np.log2(rank + 2)
            # print('********************')

    return MRR_10, RECALL_10, NDCG_10, MRR_20, RECALL_20, NDCG_20, test_user


def evaluate_baseline(model, dataset, args):
    [valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    test_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(0, usernum), 10000)
    else:
        users = range(0, usernum)
    for u in users:
        # if len(valid[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(valid[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(valid[u])
        rated.add(0)
        item_idx = [test[u]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        # item_idx = [i for i in range(0, itemnum + 1)]
        # item_idx[0] = test[u]

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        test_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if test_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / test_user, HT / test_user


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def create_optimizer_and_scheduler(
        args, model, batch_per_epoch
):

    total_steps = batch_per_epoch // args.gradient_accumulation_steps * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_prop)

    print("Batch per epoch: %d" % batch_per_epoch)
    print("Total steps: %d" % total_steps)
    print("Warmup proportion:", args.warmup_prop)
    print("Warm up steps: %d" % warmup_steps)

    no_decay = ["bias", "LayerNorm.weight", "lm_head"]#
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return optimizer, scheduler

class ExemplarGenerator:
    """ This object select exemplars from given data.
        Args:
            data (list): Training data and valid data at current cycle and exemplar data from previous cycle
                in the from of sub-sequence.
            exemplar size (int): The number of exemplars saved for each cycle.
            disable_m (bool): If true, save the same number of exemplars for each item.
            batch_size (int): Batch size to select exemplars.
            maxlen (int): The number of accumulative items until current cycle.
            dropout_rate (float): Dropout rate in trained model.
            max_item (int): The number of accumulative items until current cycle.
    """

    def __init__(self,
                 data: list,
                 exemplar_size: int,
                 disable_m: bool,
                 batch_size: int,
                 maxlen: int,
                 dropout_rate: float,
                 max_item: int,
                ) -> None:

        self.exemplars = defaultdict(list)
        self.m = exemplar_size
        self.max_item = max_item
        self.item_count = np.zeros(max_item)
        self.dropout_rate = dropout_rate

        self.sess_by_item = defaultdict(list)
        exemplar_sampler = Sampler(data, maxlen, batch_size, max_item, is_subseq=True)
        batch_num = exemplar_sampler.batch_num()

        for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                      desc='Sorting exemplars'):
            seq, pos = exemplar_sampler.sampler()
            pos = np.array(pos)
            for s, item in zip(seq, pos):
                session = np.append(s, item)
                self.sess_by_item[item].append(session)
                self.item_count[item - 1] += 1

        if disable_m:
            self.item_count = np.ones_like(self.item_count)
        item_prob = self.item_count / self.item_count.sum()
        item_count = np.random.multinomial(n=self.m, pvals=item_prob, size=1)[0]
        self.item_count = np.int32(item_count)

    def herding(self,
                rep: np.ndarray,
                logits: np.ndarray,
                seq: np.ndarray,
                item: int,
                m: int
                ) -> int:
        """ Herding algorithm for exemplar selection.
            Args:
                rep (numpy.ndarray): Calculated representations by trained model.
                logits (numpy.ndarray): Calculated logits by trained model.
                seq (numpy.ndarray): Input sessions.
                item (int): The index of item (lable) which the function selects exemplars for.
                m (int): The number of exemplar per label
            Returns:
                counter (int): The number of exemplars saved for the given item or label.
        """
        # Initialize mean and selected ids
        D = rep.T / np.linalg.norm(rep.T, axis=0)
        mu = D.mean(axis=1)
        w_t = mu
        step_t = 0
        selected_ids = []
        counter = 0
        while not (len(selected_ids) == m) and step_t < 1.1 * m:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            w_t = w_t + mu - D[:, ind_max]
            step_t += 1
            if ind_max not in selected_ids:
                selected_ids.append(ind_max)
                counter += 1
        self.exemplars[item] = [[seq[i][seq[i] != 0].tolist(), logits[i].tolist()] for i in selected_ids]
        return counter

    def herding_selection(self,
                          max_item,
                          model):
        """ This method selects exemplars using herding and selects exemplars.
            Args:
                sess (tf.Session): Tensorflow session.
                model (object): Trained model for evaluate.
            Returns:
                saved_num (int): Total number of exemplars saved for all items at current cycle.
        """
        saved_num = 0
        with torch.no_grad():
            for item in tqdm(self.sess_by_item, ncols=70, leave=False, unit='b', desc='Selecting exemplar'):
                m = self.item_count[item - 1]
                seq = self.sess_by_item[item]
                seq = np.array(seq)
                input_seq = seq[:, :-1]
                rep, logits = model(input_seq, max_item)
                rep = np.array(rep.cpu())
                logits = np.array(logits.cpu())
                saved = self.herding(rep, logits, seq, item, min(m, len(seq)))
                saved_num += saved

        return saved_num


    def randomly_selection(self,
                           max_item,
                           model
                           ) -> int:
        """ This method randomly selects exemplars.
            Args:
                sess (tf.Session): Tensorflow session.
                model (object): Trained model for evaluate.
            Returns:
                saved_num (int): Total number of exemplars saved for all items at current cycle.
        """
        saved_num = 0
        with torch.no_grad():
            for item in tqdm(self.sess_by_item, ncols=70, leave=False, unit='b', desc='Selecting exemplar'):
                seq = self.sess_by_item[item]
                seq = np.array(seq)
                seq_num = len(seq)
                m = self.item_count[item - 1]
                if m > 0:
                    selected_ids = np.random.choice(seq_num, min(m, seq_num), replace=False)
                    selected_seq = seq[selected_ids]
                    rep, logits = model(selected_seq[:, :-1], max_item)
                    logits = np.array(logits)
                    for s, l in zip(selected_seq, logits):
                        self.exemplars[item].append([s[s != 0].tolist(), l.tolist()])
                        saved_num += 1
        return saved_num
