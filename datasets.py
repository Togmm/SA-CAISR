import torch
from torch.utils.data import Dataset
import numpy as np
import ast
import swifter

class PretrainDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): The pandas dataframe containing the data.
        """
        feature_columns = ['hist_id', 'hist_rel_day', 'future_id']
        self.data = dataframe
        for col in feature_columns:
            self.data[col] = self.data[col].apply(ast.literal_eval)

        self.uid = dataframe['uid'].values
        self.cut_off_week = dataframe['cut_off_week'].values
        self.hist_id = self.data['hist_id'].values
        self.hist_rel_day_max = dataframe['hist_rel_day'].values
        self.future_id = dataframe['future_id'].values

        self.max_seq_train = 128
        self.max_seq_test = 32

    def __len__(self):
        # Return the number of rows in the dataframe
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the features and label for a given index
        uid = np.array([self.uid[idx]], dtype=np.int64)
        hist_id = self.hist_id[idx]
        hist_rel_day = self.hist_rel_day_max[idx]

        seq_lenth = np.array([len(hist_id) - 1], dtype=np.int64)

        future_id = self.future_id[idx]
        week_idx_cutoff = np.array([self.cut_off_week[idx]], dtype=np.int64)

        pad_len_train = self.max_seq_train - len(hist_id)
        hist_id = hist_id + [0] * pad_len_train
        hist_rel_day = hist_rel_day + [0] * pad_len_train

        pad_len_test = self.max_seq_test - len(future_id)
        future_id = future_id + [0] * pad_len_test

        hist_id = np.array(hist_id, dtype=np.int64)
        hist_rel_day = np.array(hist_rel_day, dtype=np.int64)
        future_id = np.array(future_id, dtype=np.int64)

        pos_idx = np.array(range(len(hist_id)), dtype=np.int64)

        # print('hist_id', hist_id, len(hist_id))
        # print('future_id', future_id, len(future_id))
        # print('hist_rel_day', hist_rel_day, len(hist_rel_day))
        # print('week_idx_cutoff', week_idx_cutoff)
        # print('seq_lenth', seq_lenth, len(seq_lenth))
        # print('hist_rel_day', hist_rel_day, len(hist_rel_day))

        # Return features and label as tensors
        return [torch.LongTensor(hist_id), torch.LongTensor(future_id), torch.LongTensor(hist_rel_day),
                torch.LongTensor(seq_lenth), torch.LongTensor(week_idx_cutoff), torch.LongTensor(pos_idx), torch.LongTensor(uid)]#

class FinetuneDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): The pandas dataframe containing the data.
        """
        feature_columns = ['hist_gti_idx', 'hist_day_rel_idx', 'win_gti_idx', 'lose_gti_idx']
        self.data = dataframe
        for col in feature_columns:
            self.data[col] = self.data[col].apply(ast.literal_eval)

        self.cut_off_week = dataframe['cut_off_week'].values
        self.hist_id = self.data['hist_gti_idx'].values
        self.hist_rel_day_max = dataframe['hist_day_rel_idx'].values
        self.win_gti_idx = dataframe['win_gti_idx'].values
        self.lose_gti_idx = dataframe['lose_gti_idx'].values
        self.goal_idx = dataframe['goal_idx'].values
        self.log_piref_win_minus_lose = dataframe['log_piref_win_minus_lose'].values
        self.goal_idx = dataframe['goal_idx'].values

        self.max_seq_train = 128
        self.max_seq_test = 32

    def __len__(self):
        # Return the number of rows in the dataframe
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the features and label for a given index
        hist_id = self.hist_id[idx]
        hist_rel_day = self.hist_rel_day_max[idx]
        seq_lenth = np.array([len(hist_id) - 1], dtype=np.int64)

        week_idx_cutoff = np.array([self.cut_off_week[idx]], dtype=np.int64)
        pad_len_train = self.max_seq_train - len(hist_id)
        hist_id = hist_id + [0] * pad_len_train
        hist_rel_day = hist_rel_day + [0] * pad_len_train

        win_gti_idx = self.win_gti_idx[idx]
        lose_gti_idx = self.lose_gti_idx[idx]

        pad_len_test = self.max_seq_test - len(win_gti_idx)
        win_gti_idx = win_gti_idx + [0] * pad_len_test
        lose_gti_idx = lose_gti_idx + [0] * pad_len_test

        hist_id = np.array(hist_id, dtype=np.int64)
        hist_rel_day = np.array(hist_rel_day, dtype=np.int64)
        win_gti_idx = np.array(win_gti_idx, dtype=np.int64)
        lose_gti_idx = np.array(lose_gti_idx, dtype=np.int64)

        pos_idx = np.array(range(len(hist_id)), dtype=np.int64)
        log_piref_win_minus_lose = np.array([self.log_piref_win_minus_lose[idx]], dtype=np.float64)
        goal_idx = np.array([self.goal_idx[idx]], dtype=np.int64)

        # print('hist_id', hist_id, len(hist_id))
        # print('hist_rel_day', hist_rel_day, len(hist_rel_day))
        # print('seq_lenth', seq_lenth, len(seq_lenth))
        # print('week_idx_cutoff', week_idx_cutoff)
        # print('win_gti_idx', win_gti_idx, len(win_gti_idx))
        # print('lose_gti_idx', lose_gti_idx, len(lose_gti_idx))
        # print('pos_idx', pos_idx, len(pos_idx))
        # print('log_piref_win_minus_lose', log_piref_win_minus_lose, len(log_piref_win_minus_lose))
        # print('goal_idx', goal_idx, len(goal_idx))

        # Return features and label as tensors
        return [torch.LongTensor(hist_id), torch.LongTensor(hist_rel_day), torch.LongTensor(seq_lenth),
                torch.LongTensor(win_gti_idx), torch.LongTensor(lose_gti_idx), torch.Tensor(log_piref_win_minus_lose),
                torch.LongTensor(week_idx_cutoff), torch.LongTensor(pos_idx), torch.LongTensor(goal_idx)]#

class ValidDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): The pandas dataframe containing the data.
        """
        feature_columns = ['hist_id', 'hist_rel_day', 'future_id']
        self.data = dataframe
        for col in feature_columns:
            self.data[col] = self.data[col].swifter.apply(ast.literal_eval)

        self.cut_off_week = dataframe['cut_off_week'].values
        self.hist_id = self.data['hist_id'].values
        self.hist_rel_day_max = dataframe['hist_rel_day'].values
        self.future_id = dataframe['future_id'].values

        self.max_seq_train = 128
        self.max_seq_test = 32

    def __len__(self):
        # Return the number of rows in the dataframe
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the features and label for a given index
        hist_id = self.hist_id[idx]
        hist_rel_day = self.hist_rel_day_max[idx]
        week_idx_cutoff = np.array([self.cut_off_week[idx] - 1], dtype=np.int64)
        future_id = self.future_id[idx]
        seq_lenth = np.array([len(hist_id) - 1], dtype=np.int64)

        pad_len_train = self.max_seq_train - len(hist_id)
        hist_id = hist_id + [0] * pad_len_train
        hist_rel_day = hist_rel_day + [0] * pad_len_train

        pad_len_test = self.max_seq_test - len(future_id)
        future_id = future_id + [0] * pad_len_test

        hist_id = np.array(hist_id, dtype=np.int64)
        hist_rel_day = np.array(hist_rel_day, dtype=np.int64)
        future_id = np.array(future_id, dtype=np.int64)

        pos_idx = np.array(range(len(hist_id)), dtype=np.int64)
        # print('hist_id', hist_id, len(hist_id))
        # print('future_id', future_id, len(future_id))
        # print('hist_rel_day', hist_rel_day, len(hist_rel_day))
        # print('week_idx_cutoff', week_idx_cutoff, len(week_idx_cutoff))
        # print('seq_lenth', seq_lenth, len(seq_lenth))
        # print('hist_rel_day', hist_rel_day, len(hist_rel_day))
        # Return features and label as tensors
        return [torch.LongTensor(hist_id), torch.LongTensor(future_id), torch.LongTensor(hist_rel_day),
                torch.LongTensor(seq_lenth), torch.LongTensor(week_idx_cutoff), torch.LongTensor(pos_idx)]
