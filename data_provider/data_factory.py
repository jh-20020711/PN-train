import torch
import torch.utils.data as Data

import numpy as np
import pandas as pd

from utils.timefeatures import time_features
import os
class Dataset:
    def __init__(self, args, root_path, size=None, data_path='', scale=True, timeenc=2, scaler=None, ho_index=-2):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        self.args = args
        self.dataset = args.data
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.finetune_sample_num = args.finetune_sample_num

        self.retrain_ho_num = args.finetune_sample_num
        self.retrain_ge_num = args.finetune_sample_num

        self.ind_ho_num = args.detect_sample_num
        self.ind_ge_num = args.detect_sample_num

        self.scale = scale
        self.timeenc = timeenc
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = scaler
        self.ho_index = ho_index
        self.__read_data__()

        self.train_loader, self.retrain_loader = self.get_dataset_train_retrain(self.border1s[0], self.border2s[0])
        self.val_loader, self.indication_loader = self.get_dataset_val_indication(self.border1s[1], self.border2s[1])
        self.test_loader, self.debug_loader = self.get_dataset_test_debug(self.border1s[2], self.border2s[2]-self.pred_len-self.seq_len+1) #self.get_dataset_test(self.border1s[2], self.border2s[2])

    def __read_data__(self):
        data_path = os.path.join(self.root_path, self.data_path)

        if '.h5' in data_path:
            df_raw = pd.read_hdf(os.path.join(self.root_path, self.data_path))
        elif '.npz' in data_path:
            df_raw = np.load(os.path.join(self.root_path, self.data_path), allow_pickle=True)
            df_raw = df_raw['data']
        elif '.txt' in data_path:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), delimiter=';')
            df_raw = df_raw.values

        num_train = int(len(df_raw) * self.args.train_ratio)
        num_test = int(len(df_raw) * (1-self.args.train_ratio)/2)
        num_vali = len(df_raw) - num_train - num_test
        self.border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        self.border2s = [num_train, num_train + num_vali, len(df_raw)]

        if '.h5' in data_path:
            df_data = df_raw.values.astype(float)
        elif '.npz' in data_path:
            df_data = df_raw[..., -1].astype(float)
        elif '.txt' in data_path:
            df_data = df_raw[:, 1:].astype(float)

        if self.scale:
            train_data = df_data[self.border1s[0]:self.border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        if '.h5' in data_path:
            df_stamp = df_raw.index.tolist()
            df_stamp = pd.DataFrame(df_stamp, columns=['date'])
        elif '.npz' in data_path:
            df_stamp = df_raw[:, 0, 0]
            df_stamp = pd.DataFrame(df_stamp, columns=['date'])
        data_stamp = time_features(df_stamp, timeenc=2, dataset=self.dataset)

        self.data = data[:]
        self.data_stamp = data_stamp[:]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def set_dataloader_value(self, seq_x, seq_y, seq_x_mark, seq_y_mark, batch_size, shuffle_flag, drop_last):
        loader = Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(np.array(seq_x)).float(),
                torch.from_numpy(np.array(seq_y)).float(),
                torch.from_numpy(np.array(seq_x_mark)).float(),
                torch.from_numpy(np.array(seq_y_mark)).float(),
            ),
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            drop_last=drop_last,
            persistent_workers=False
        )
        return loader

    def _get_datasets_ho_ge(self, start, end, type):
        seq_x, seq_y, seq_x_mark, seq_y_mark = [], [], [], []
        seq_x_ho, seq_y_ho, seq_x_mark_ho, seq_y_mark_ho = [], [], [], []
        seq_x_ge, seq_y_ge, seq_x_mark_ge, seq_y_mark_ge = [], [], [], []
        count_ho, count_ge, tot_count = 0, 0, 0

        if type == 'train':
            ho_all_index_lst = []
            ho_sep_index_lst = []

        for curr in range(start, end):
            s_end = curr + self.seq_len
            r_end = s_end + self.pred_len

            s_x = self.data[curr:s_end]
            s_y = self.data[s_end:r_end]
            s_x_mark = self.data_stamp[curr:s_end]
            s_y_mark = self.data_stamp[s_end:r_end]

            if any(s_x_mark[..., self.ho_index]) or any(s_y_mark[..., self.ho_index]):
                seq_x_ho.append(s_x)
                seq_y_ho.append(s_y)
                seq_x_mark_ho.append(s_x_mark)
                seq_y_mark_ho.append(s_y_mark)

                if type == 'train':
                    ho_all_index_lst.append(tot_count)
                    ho_sep_index_lst.append(count_ho)
                    count_ho += 1
            else:
                seq_x_ge.append(s_x)
                seq_y_ge.append(s_y)
                seq_x_mark_ge.append(s_x_mark)
                seq_y_mark_ge.append(s_y_mark)
                count_ge += 1

            tot_count += 1

            seq_x.append(s_x)
            seq_y.append(s_y)
            seq_x_mark.append(s_x_mark)
            seq_y_mark.append(s_y_mark)

        if type == 'train':
            if self.retrain_ho_num != 0:
                select_num = min(len(ho_sep_index_lst), self.retrain_ho_num)

                selected_indices = np.random.choice(len(ho_sep_index_lst), select_num, replace=False)

                ho_sep_index_array = np.array(ho_sep_index_lst)
                ho_all_index_array = np.array(ho_all_index_lst)

                ho_sep_location = ho_sep_index_array[selected_indices]
                ho_all_location = ho_all_index_array[selected_indices]

                ho_all_location_set = set(ho_all_location)

                seq_x = [x for i, x in enumerate(seq_x) if i not in ho_all_location_set]
                seq_y = [x for i, x in enumerate(seq_y) if i not in ho_all_location_set]
                seq_x_mark = [x for i, x in enumerate(seq_x_mark) if i not in ho_all_location_set]
                seq_y_mark = [x for i, x in enumerate(seq_y_mark) if i not in ho_all_location_set]

                seq_x_ho = np.array(seq_x_ho)
                seq_y_ho = np.array(seq_y_ho)
                seq_x_mark_ho = np.array(seq_x_mark_ho)
                seq_y_mark_ho = np.array(seq_y_mark_ho)

                seq_x_ho = seq_x_ho[ho_sep_location].tolist()
                seq_y_ho = seq_y_ho[ho_sep_location].tolist()
                seq_x_mark_ho = seq_x_mark_ho[ho_sep_location].tolist()
                seq_y_mark_ho = seq_y_mark_ho[ho_sep_location].tolist()
            else:
                seq_x_ho, seq_y_ho, seq_x_mark_ho, seq_y_mark_ho, seq_x_ge, seq_y_ge, seq_x_mark_ge, seq_y_mark_ge = seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x, seq_y, seq_x_mark, seq_y_mark

        return (seq_x, seq_y, seq_x_mark, seq_y_mark), (seq_x_ho, seq_y_ho, seq_x_mark_ho, seq_y_mark_ho, seq_x_ge, seq_y_ge, seq_x_mark_ge,
                             seq_y_mark_ge)

    def _get_datasets_all(self, start, end, type='train'):

        current_save_path = os.path.join(self.root_path, self.dataset, 'retrain'+str(self.args.finetune_sample_num), str(self.args.train_ratio))

        if not os.path.exists(current_save_path):
            os.makedirs(current_save_path)

        if type == 'train':
            file_path_all = os.path.join(current_save_path, 'train.npz')
            file_path_sep = os.path.join(current_save_path, 'retrain.npz')
            sep_batch_size = self.batch_size
            if self.retrain_ho_num != 0:
                select_num_ho = self.retrain_ho_num
                select_num_ge = self.retrain_ge_num
            else:
                select_num_ho = None
                select_num_ge = None
        elif type == 'val':
            file_path_all = os.path.join(current_save_path, 'val.npz')
            file_path_sep = os.path.join(current_save_path, 'detect.npz')
            sep_batch_size = 1
            if self.ind_ho_num != 0:
                select_num_ho = self.ind_ho_num
                select_num_ge = self.ind_ge_num
            else:
                select_num_ho = None
                select_num_ge = None
        else:
            file_path_all = os.path.join(current_save_path, 'test.npz')
            file_path_sep = os.path.join(current_save_path, 'debug.npz')
            sep_batch_size = self.batch_size

        if os.path.exists(file_path_all):
            current_data = np.load(file_path_all, allow_pickle=True)
            seq_x, seq_y, seq_x_mark, seq_y_mark = current_data['seq_x'], current_data['seq_y'], current_data[
                'seq_x_mark'], current_data['seq_y_mark']
        else:
            (seq_x, seq_y, seq_x_mark, seq_y_mark), (seq_x_ho, seq_y_ho, seq_x_mark_ho,
                                                     seq_y_mark_ho, seq_x_ge, seq_y_ge,
                                                     seq_x_mark_ge, seq_y_mark_ge) \
                = self._get_datasets_ho_ge(start, end, type)

            np.savez(file_path_all, seq_x=seq_x, seq_y=seq_y, seq_x_mark=seq_x_mark, seq_y_mark=seq_y_mark)

            np.savez(file_path_sep,
                     seq_x_ho=seq_x_ho, seq_y_ho=seq_y_ho,
                     seq_x_mark_ho=seq_x_mark_ho, seq_y_mark_ho=seq_y_mark_ho,
                     seq_x_ge=seq_x_ge, seq_y_ge=seq_y_ge,
                     seq_x_mark_ge=seq_x_mark_ge, seq_y_mark_ge=seq_y_mark_ge)

        data_loader = self.set_dataloader_value(seq_x, seq_y, seq_x_mark, seq_y_mark, self.batch_size, True, False)

        if os.path.exists(file_path_sep):
            data = np.load(file_path_sep, allow_pickle=True)
            seq_x_ho, seq_y_ho, seq_x_mark_ho, seq_y_mark_ho = data['seq_x_ho'], data['seq_y_ho'], data[
                'seq_x_mark_ho'], data['seq_y_mark_ho']
            seq_x_ge, seq_y_ge, seq_x_mark_ge, seq_y_mark_ge = data['seq_x_ge'], data['seq_y_ge'], data[
                'seq_x_mark_ge'], data['seq_y_mark_ge']

        if type == 'test':
            loader_ho = self.select_data(seq_x_ho, seq_y_ho, seq_x_mark_ho, seq_y_mark_ho, batch_size=sep_batch_size,
                                         select_num=None, shuffle_flag=False, type=type)


            loader_ge = self.select_data(seq_x_ge, seq_y_ge, seq_x_mark_ge, seq_y_mark_ge, batch_size=sep_batch_size,
                                         select_num=None, shuffle_flag=False, type=type)
        else:
            loader_ho = self.select_data(seq_x_ho, seq_y_ho, seq_x_mark_ho, seq_y_mark_ho, batch_size=sep_batch_size,
                                         select_num=select_num_ho, type=type)
            loader_ge = self.select_data(seq_x_ge, seq_y_ge, seq_x_mark_ge, seq_y_mark_ge,
                                         batch_size=sep_batch_size, select_num=select_num_ge, type=type)

        k_loader = (loader_ho, loader_ge)

        return data_loader, k_loader
    def get_dataset_train_retrain(self, start, end):

        data_loader, k_loader = self._get_datasets_all(start, end, type='train')

        return data_loader, k_loader

    def get_dataset_test_debug(self, start, end):
        data_loader, k_loader = self._get_datasets_all(start, end, type='test')
        return data_loader, k_loader

    def select_data(self, seq_x, seq_y, seq_x_mark, seq_y_mark, batch_size=32, shuffle_flag=True, select_num=None, is_random=True, type=None):

        seq_x, seq_y, seq_x_mark, seq_y_mark = np.array(seq_x), np.array(seq_y), np.array(seq_x_mark), np.array(
            seq_y_mark)

        if type != 'train':
            num_samples = len(seq_x)
            print(num_samples)
            seq_x_selected, seq_y_selected = [], []
            seq_x_mark_selected, seq_y_mark_selected = [], []

            if select_num is None:
                seq_x_selected = seq_x
                seq_y_selected = seq_y
                seq_x_mark_selected = seq_x_mark
                seq_y_mark_selected = seq_y_mark
            else:
                if is_random:
                    if num_samples < select_num:
                        select_num = num_samples

                    selected_indices = np.random.choice(num_samples, select_num, replace=False)
                    seq_x_selected = seq_x[selected_indices]
                    seq_y_selected = seq_y[selected_indices]
                    seq_x_mark_selected = seq_x_mark[selected_indices]
                    seq_y_mark_selected = seq_y_mark[selected_indices]
        else:
            seq_x_selected = seq_x
            seq_y_selected = seq_y
            seq_x_mark_selected = seq_x_mark
            seq_y_mark_selected = seq_y_mark

        if len(seq_x_selected) == 0:
            print('cannot use this dataset')
            print(seq_x_selected.shape)
        else:
            print(type, len(seq_x_selected))

        loader = self.set_dataloader_value(seq_x_selected, seq_y_selected, seq_x_mark_selected, seq_y_mark_selected,
                                              batch_size, shuffle_flag, False)
        return loader

    def get_dataset_val_indication(self, start, end):
        data_loader, k_loader = self._get_datasets_all(start, end, type='val')
        return data_loader, k_loader

    def get_dataset_test(self, start, end):
        file_path_test = os.path.join(self.root_path, self.dataset, 'test.npz')
        if os.path.exists(file_path_test):
            test_data = np.load(file_path_test, allow_pickle=True)
            seq_x, seq_y, seq_x_mark, seq_y_mark = test_data['seq_x'], test_data['seq_y'], test_data[
                'seq_x_mark'], test_data['seq_y_mark']
        else:
            seq_x, seq_y, seq_x_mark, seq_y_mark = [], [], [], []

            for curr in range(start, end-self.pred_len-self.seq_len+1):

                s_end = curr + self.seq_len
                r_end = s_end + self.pred_len

                s_x = self.data[curr:s_end]
                s_y = self.data[s_end:r_end]
                s_x_mark = self.data_stamp[curr:s_end]
                s_y_mark = self.data_stamp[s_end:r_end]

                seq_x.append(s_x)
                seq_y.append(s_y)
                seq_x_mark.append(s_x_mark)
                seq_y_mark.append(s_y_mark)

        test_loader = self.set_dataloader_value(seq_x, seq_y, seq_x_mark, seq_y_mark, self.args.test_bsz, False, False)

        return test_loader
