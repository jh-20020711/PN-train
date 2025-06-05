import os
import time
import warnings

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import re
import random
import copy
import pickle
from data_provider.data_factory import Dataset
from exp.exp_basic import Exp_Basic
from models.pn_train import STAEformer
from utils.criterion import select_criterion
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, get_intersections_importance

warnings.filterwarnings('ignore')

class Exp(Exp_Basic):
    def __init__(self, args, scaler):
        self.args = args
        self.scaler = scaler
        self.args.scaler = scaler

        self.ffn_t_index, self.ffn_s_index, self.attn_t_index, self.attn_s_index = None, None, None, None

        self.device = self._acquire_device()
        self.args.device = self.device

        self.n_inner = args.n_inner
        self.opt_str = args.opt
        args.indicate = False

        self.model = STAEformer(args).to(self.device)

        self.loss = self._select_criterion(args.loss_type)
        self.opt = self._select_optimizer()

        self.dataset = Dataset(args, root_path=args.root_path, data_path=args.data_path,
                               size=[args.seq_len, args.pred_len],
                               scaler=args.scaler)

        self.train_loader, (self.retrain_loader_ho, self.retrain_loader_ge) \
            = self.dataset.train_loader, self.dataset.retrain_loader
        self.vali_loader, (self.indication_loader_ho, self.indication_loader_ge) \
            = self.dataset.val_loader, self.dataset.indication_loader
        self.test_loader, (self.debug_loader_ho, self.debug_loader_ge) = self.dataset.test_loader, self.dataset.debug_loader

    def _select_optimizer(self):
        if self.args.opt == 'adamw':
            self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            self.opt = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def load_pickle(self, pickle_file):
        import pickle
        try:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', pickle_file, ':', e)
            raise
        return pickle_data

    def _select_criterion(self, loss_type):
        return select_criterion(loss_type)

    def train(self, setting, logger):
        # 模型的标准训练过程
        self.setting = setting  # 保存当前的实验设置
        path = os.path.join(self.args.checkpoints, setting)  # 创建模型保存路径
        if not os.path.exists(path):
            os.makedirs(path)  # 如果路径不存在，则创建它

        time_now = time.time()  # 记录当前时间，用于计算训练耗时

        train_steps = len(self.train_loader)  # 获取训练数据加载器的步数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 初始化早停机制

        if self.args.use_amp:
            amp_scaler = torch.cuda.amp.GradScaler()  # 如果启用了混合精度训练，则创建GradScaler对象用于缩放梯度

        for epoch in range(self.args.train_epochs):  # 遍历训练的每个周期
            iter_count = 0  # 初始化迭代计数器
            train_loss = []  # 用于存储每个批次的训练损失

            self.model.train()  # 将模型设置为训练模式
            epoch_time = time.time()  # 记录当前周期的起始时间
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):  # 遍历数据加载器中的每个批次
                iter_count += 1  # 增加迭代计数器

                self.opt.zero_grad()  # 清除之前计算的梯度

                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)  # 处理一个批次的数据，获取预测值和真实值
                pred = self.scaler.inverse_transform(pred)  # 对预测值进行逆变换，还原到原始数据的尺度
                true = self.scaler.inverse_transform(true)  # 对真实值进行逆变换，还原到原始数据的尺度

                loss = self.loss(pred, true)  # 计算当前批次的损失
                train_loss.append(loss.item())  # 将当前批次的损失添加到训练损失列表中

                if (i + 1) % 100 == 0:
                    # 每100个批次打印一次日志信息
                    log = "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    logger.info(log)
                    speed = (time.time() - time_now) / iter_count  # 计算训练速度
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)  # 计算剩余训练时间
                    log = '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time)
                    logger.info(log)
                    iter_count = 0  # 重置迭代计数器
                    time_now = time.time()  # 更新当前时间

                if self.args.use_amp:
                    # 如果启用了混合精度训练，则缩放损失并反向传播
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(self.opt)  # 更新优化器的参数
                    amp_scaler.update()  # 更新GradScaler对象的状态
                else:
                    loss.backward()  # 反向传播计算梯度
                    self.opt.step()  # 更新优化器的参数

            # 每个周期结束后打印日志信息
            log = "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
            logger.info(log)
            train_loss = np.average(train_loss)  # 计算平均训练损失
            vali_loss = self.vali(self.vali_loader)  # 在验证集上评估模型性能

            test_loss = 0.  # 初始化测试损失
            log = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss)
            logger.info(log)
        
            early_stopping(vali_loss, self.model, path)  # 检查是否需要早停
            if early_stopping.early_stop:
                print("Early stopping")
                logger.info("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)  # 调整学习率

        best_model_path = path + '/' + 'checkpoint.pth'  # 定义最佳模型的保存路径
        self.model.load_state_dict(torch.load(best_model_path))  # 加载最佳模型的权重
        self.args.checkpoint_path = best_model_path  # 更新检查点路径

        return self.model  # 返回训练后的模型

    def vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = self.loss(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        return total_loss

    def test_model(self, current_model, data_loader, log_info=None, log_name=None):
        #测试模型性能，记录预测结果和真实值。
        current_model.eval()
        preds, trues = [], []
        batch_x_marks = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(data_loader)):
                x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        pred = current_model(x, batch_x_mark)
                else:
                    pred = current_model(x, batch_x_mark)

                true = batch_y[:, -self.args.pred_len:]
                pred = pred.detach().cpu()
                true = true.detach().cpu()

                pred = self.scaler.inverse_transform(pred)
                pred[pred < 0] = 0
                true = self.scaler.inverse_transform(true)
                true[true < 0] = 0

                preds.append(pred)
                trues.append(true)
                batch_x_marks.append(batch_x_mark)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        batch_x_marks = torch.cat(batch_x_marks, dim=0).numpy()

        mae, mse, rmse, mape, wmape = metric(preds, trues)
        log = '{}, mae:{}, rmse:{}, wmape:{}, {}, {}'.format(log_info, mae, rmse, wmape, preds.shape, trues.shape)

        self.logger.info(log)
        torch.cuda.empty_cache()

        return preds, trues

    def test(self, setting, logger, loader=None):

        best_model_path = self.args.checkpoint_path
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.args.device))

        self.logger = logger

        self.model.eval()

        self.test_model(self.model, self.debug_loader_ho, log_info='holiday after test ', log_name='test_holiday')
        self.test_model(self.model, self.debug_loader_ge, log_info='general after test ', log_name='test_general')
        self.test_model(self.model, self.test_loader, log_info='test all ', log_name='test_all')

        return self.model

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):

        x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x, batch_x_mark)
        else:
            outputs = self.model(x, batch_x_mark)
        batch_y = batch_y[:, -self.args.pred_len:].to(self.device)

        return outputs, batch_y

    def _process_one_batch_model(self, current_model, batch_x, batch_y, batch_x_mark, batch_y_mark):
        x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = current_model(x, batch_x_mark)
        else:
            outputs = current_model(x, batch_x_mark)
        batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
        return outputs, batch_y

    def _get_union(self, curr_ho, curr_ge):

        curr_ho = set(curr_ho)
        curr_ge = set(curr_ge)

        common_any = curr_ho.intersection(curr_ge)

        return common_any

    def _remove_duplicate(self):
        self.k_t_index_ho_new_rm, self.k_t_index_ge_new_rm = {}, {}
        self.q_t_index_ho_new_rm, self.q_t_index_ge_new_rm = {}, {}
        self.v_t_index_ho_new_rm, self.v_t_index_ge_new_rm = {}, {}
        self.out_t_index_ho_new_rm,  self.out_t_index_ge_new_rm = {}, {}
        self.up_t_index_ho_new_rm,  self.up_t_index_ge_new_rm = {}, {}
        self.down_t_index_ho_new_rm, self.down_t_index_ge_new_rm = {}, {}

        self.k_s_index_ho_new_rm,  self.k_s_index_ge_new_rm = {}, {}
        self.q_s_index_ho_new_rm,  self.q_s_index_ge_new_rm = {}, {}
        self.v_s_index_ho_new_rm, self.v_s_index_ge_new_rm = {}, {}
        self.out_s_index_ho_new_rm, self.out_s_index_ge_new_rm = {}, {}
        self.up_s_index_ho_new_rm, self.up_s_index_ge_new_rm = {}, {}
        self.down_s_index_ho_new_rm, self.down_s_index_ge_new_rm = {}, {}

        self.k_t_index_common = {}
        self.q_t_index_common = {}
        self.v_t_index_common = {}
        self.out_t_index_common = {}
        self.up_t_index_common = {}
        self.down_t_index_common = {}

        self.k_s_index_common = {}
        self.q_s_index_common = {}
        self.v_s_index_common = {}
        self.out_s_index_common = {}
        self.up_s_index_common = {}
        self.down_s_index_common = {}

        for layer in range(self.model.num_layers):
            # kt
            curr_kt_ho = self.k_t_index_ho[layer]
            curr_kt_ge = self.k_t_index_ge[layer]

            common_any_kt = self._get_union(curr_kt_ho, curr_kt_ge)

            self.k_t_index_ho_new_rm[layer] = set(curr_kt_ho) - common_any_kt
            self.k_t_index_ge_new_rm[layer] = set(curr_kt_ge) - common_any_kt

            self.k_t_index_common[layer] = common_any_kt

            # qt
            curr_qt_ho = self.q_t_index_ho[layer]
            curr_qt_ge = self.q_t_index_ge[layer]

            common_any_qt = self._get_union(curr_qt_ho, curr_qt_ge)

            self.q_t_index_ho_new_rm[layer] = set(curr_qt_ho) - common_any_qt
            self.q_t_index_ge_new_rm[layer] = set(curr_qt_ge) - common_any_qt

            self.q_t_index_common[layer] = common_any_qt

            # vt
            curr_vt_ho = self.v_t_index_ho[layer]
            curr_vt_ge = self.v_t_index_ge[layer]

            common_any_vt = self._get_union(curr_vt_ho, curr_vt_ge)

            self.v_t_index_ho_new_rm[layer] = set(curr_vt_ho) - common_any_vt
            self.v_t_index_ge_new_rm[layer] = set(curr_vt_ge) - common_any_vt

            self.v_t_index_common[layer] = common_any_vt

            # vo
            curr_ot_ho = self.out_t_index_ho[layer]
            curr_ot_ge = self.out_t_index_ge[layer]

            common_any_ot = self._get_union(curr_ot_ho, curr_ot_ge)

            self.out_t_index_ho_new_rm[layer] = set(curr_ot_ho) - common_any_ot
            self.out_t_index_ge_new_rm[layer] = set(curr_ot_ge) - common_any_ot

            self.out_t_index_common[layer] = common_any_ot

            # up
            curr_ut_ho = self.up_t_index_ho[layer]
            curr_ut_ge = self.up_t_index_ge[layer]

            common_any_ut = self._get_union(curr_ut_ho, curr_ut_ge)

            self.up_t_index_ho_new_rm[layer] = set(curr_ut_ho) - common_any_ut
            self.up_t_index_ge_new_rm[layer] = set(curr_ut_ge) - common_any_ut
            self.up_t_index_common[layer] = common_any_ut

            # down
            curr_dt_ho = self.down_t_index_ho[layer]
            curr_dt_ge = self.down_t_index_ge[layer]

            common_any_dt = self._get_union(curr_dt_ho, curr_dt_ge)

            self.down_t_index_ho_new_rm[layer] = set(curr_dt_ho) - common_any_dt
            self.down_t_index_ge_new_rm[layer] = set(curr_dt_ge) - common_any_dt

            self.down_t_index_common[layer] = common_any_dt

            ## star s
            # ks
            curr_ks_ho = self.k_s_index_ho[layer]
            curr_ks_ge = self.k_s_index_ge[layer]

            common_any_ks = self._get_union(curr_ks_ho, curr_ks_ge)

            self.k_s_index_ho_new_rm[layer] = set(curr_ks_ho) - common_any_ks
            self.k_s_index_ge_new_rm[layer] = set(curr_ks_ge) - common_any_ks

            self.k_s_index_common[layer] = common_any_ks

            # qs
            curr_qs_ho = self.q_s_index_ho[layer]
            curr_qs_ge = self.q_s_index_ge[layer]

            common_any_qs = self._get_union(curr_qs_ho, curr_qs_ge)

            self.q_s_index_ho_new_rm[layer] = set(curr_qs_ho) - common_any_qs
            self.q_s_index_ge_new_rm[layer] = set(curr_qs_ge) - common_any_qs

            self.q_s_index_common[layer] = common_any_qs

            # vs
            curr_vs_ho = self.v_s_index_ho[layer]
            curr_vs_ge = self.v_s_index_ge[layer]

            common_any_vs = self._get_union(curr_vs_ho, curr_vs_ge)

            self.v_s_index_ho_new_rm[layer] = set(curr_vs_ho) - common_any_vs
            self.v_s_index_ge_new_rm[layer] = set(curr_vs_ge) - common_any_vs
            self.v_s_index_common[layer] = common_any_vs

            # vs
            curr_os_ho = self.out_s_index_ho[layer]
            curr_os_ge = self.out_s_index_ge[layer]

            common_any_os = self._get_union(curr_os_ho, curr_os_ge)

            self.out_s_index_ho_new_rm[layer] = set(curr_os_ho) - common_any_os
            self.out_s_index_ge_new_rm[layer] = set(curr_os_ge) - common_any_os

            self.out_s_index_common[layer] = common_any_os

            # us
            curr_us_ho = self.up_s_index_ho[layer]
            curr_us_ge = self.up_s_index_ge[layer]

            common_any_us = self._get_union(curr_us_ho, curr_us_ge)

            self.up_s_index_ho_new_rm[layer] = set(curr_us_ho) - common_any_us
            self.up_s_index_ge_new_rm[layer] = set(curr_us_ge) - common_any_us

            self.up_s_index_common[layer] = common_any_us

            # down
            curr_ds_ho = self.down_s_index_ho[layer]
            curr_ds_ge = self.down_s_index_ge[layer]

            common_any_ds = self._get_union(curr_ds_ho, curr_ds_ge)

            self.down_s_index_ho_new_rm[layer] = set(curr_ds_ho) - common_any_ds
            self.down_s_index_ge_new_rm[layer] = set(curr_ds_ge) - common_any_ds
            self.down_s_index_common[layer] = common_any_ds

    def _merge_param(self, model_holiday, model_general):
        model_combined = copy.deepcopy(model_holiday)
        state_dict_holiday = model_holiday.state_dict()
        state_dict_general = model_general.state_dict()
        new_state_dict = {}

        for name in state_dict_holiday:
            if (name in model_general) and (name in state_dict_holiday):
                new_state_dict[name] = (state_dict_holiday[name] +
                                        state_dict_general[name]) / 2
            else:
                raise KeyError(f"Parameter {name} not found in all models")

        model_combined.load_state_dict(new_state_dict)

        return model_combined

    def _loop_pattern(self, dataloader, current_type='holiday'):

        preds, trues, batch_x_marks = [], [], []

        indication_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(dataloader)):

            x = batch_x.float().to(self.device)

            batch_y = batch_y.float()
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    pred, t_activations, s_activations, t_activations_save, s_activations_save = self.model(x, batch_x_mark, detect=True)
            else:
                pred, t_activations, s_activations, t_activations_save, s_activations_save = self.model(x, batch_x_mark, detect=True)

            pred = self.scaler.inverse_transform(pred)

            true = batch_y[:, -self.args.pred_len:].to(self.device)
            true = self.scaler.inverse_transform(true)

            loss = self.loss(pred, true)

            self.opt.zero_grad()
            loss.backward()

            self.set_important_score(t_activations, s_activations, t_activations_save, s_activations_save, current_type, index=i)

            pred = pred.detach().cpu()
            true = true.detach().cpu()

            pred[pred < 0] = 0
            true[true < 0] = 0
            preds.append(pred)
            trues.append(true)
            batch_x_marks.append(batch_x_mark)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        batch_x_marks = torch.cat(batch_x_marks, dim=0).numpy()

        if current_type == 'holiday':
            self.k_t_index_ho = get_intersections_importance(self.k_imp_t_lst_ho, num_layers=self.model.num_layers,
                                                             type='k_t_ho')
            self.q_t_index_ho = get_intersections_importance(self.q_imp_t_lst_ho, num_layers=self.model.num_layers,
                                                             type='q_t_ho')
            self.v_t_index_ho = get_intersections_importance(self.v_imp_t_lst_ho, num_layers=self.model.num_layers,
                                                             type='v_t_ho')
            self.out_t_index_ho = get_intersections_importance(self.out_imp_t_lst_ho, num_layers=self.model.num_layers,
                                                               type='o_t_ho')
            self.up_t_index_ho = get_intersections_importance(self.up_imp_t_lst_ho, num_layers=self.model.num_layers,
                                                              type='u_t_ho')
            self.down_t_index_ho = get_intersections_importance(self.down_imp_t_lst_ho,
                                                                num_layers=self.model.num_layers,
                                                                type='d_t_ho')

            self.k_s_index_ho = get_intersections_importance(self.k_imp_s_lst_ho, num_layers=self.model.num_layers,
                                                             type='k_s_ho')
            self.q_s_index_ho = get_intersections_importance(self.q_imp_s_lst_ho, num_layers=self.model.num_layers,
                                                             type='q_s_ho')
            self.v_s_index_ho = get_intersections_importance(self.v_imp_s_lst_ho, num_layers=self.model.num_layers,
                                                             type='v_s_ho')
            self.out_s_index_ho = get_intersections_importance(self.out_imp_s_lst_ho, num_layers=self.model.num_layers,
                                                               type='o_s_ho')
            self.up_s_index_ho = get_intersections_importance(self.up_imp_s_lst_ho, num_layers=self.model.num_layers,
                                                              type='u_s_ho')
            self.down_s_index_ho = get_intersections_importance(self.down_imp_s_lst_ho,
                                                                num_layers=self.model.num_layers,
                                                                type='d_s_ho')
        elif current_type == 'general':
            self.k_t_index_ge = get_intersections_importance(self.k_imp_t_lst_ge, num_layers=self.model.num_layers,
                                                             type='ge_k_t')
            self.q_t_index_ge = get_intersections_importance(self.q_imp_t_lst_ge, num_layers=self.model.num_layers,
                                                             type='ge_q_t')
            self.v_t_index_ge = get_intersections_importance(self.v_imp_t_lst_ge, num_layers=self.model.num_layers,
                                                             type='ge_v_t')
            self.out_t_index_ge = get_intersections_importance(self.out_imp_t_lst_ge, num_layers=self.model.num_layers,
                                                               type='ge_o_t')
            self.up_t_index_ge = get_intersections_importance(self.up_imp_t_lst_ge, num_layers=self.model.num_layers,
                                                              type='ge_u_t')
            self.down_t_index_ge = get_intersections_importance(self.down_imp_t_lst_ge,
                                                                num_layers=self.model.num_layers,
                                                                type='ge_d_t')

            self.k_s_index_ge = get_intersections_importance(self.k_imp_s_lst_ge, num_layers=self.model.num_layers,
                                                             type='ge_k_s')
            self.q_s_index_ge = get_intersections_importance(self.q_imp_s_lst_ge, num_layers=self.model.num_layers,
                                                             type='ge_q_s')
            self.v_s_index_ge = get_intersections_importance(self.v_imp_s_lst_ge, num_layers=self.model.num_layers,
                                                             type='ge_v_s')
            self.out_s_index_ge = get_intersections_importance(self.out_imp_s_lst_ge, num_layers=self.model.num_layers,
                                                               type='ge_o_s')
            self.up_s_index_ge = get_intersections_importance(self.up_imp_s_lst_ge, num_layers=self.model.num_layers,
                                                              type='ge_u_s')
            self.down_s_index_ge = get_intersections_importance(self.down_imp_s_lst_ge,
                                                                num_layers=self.model.num_layers,
                                                                type='ge_d_s')

        return preds, trues, batch_x_marks

    def detect(self, setting, logger):

        best_model_path = self.args.checkpoint_path
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.args.device))
        self.setting = setting
        self.logger = logger
        self.model.eval()

        self.q_imp_t_lst_ho, self.k_imp_t_lst_ho, self.v_imp_t_lst_ho, self.out_imp_t_lst_ho = [], [], [], []
        self.up_imp_t_lst_ho, self.down_imp_t_lst_ho = [], []

        self.q_imp_s_lst_ho, self.k_imp_s_lst_ho, self.v_imp_s_lst_ho, self.out_imp_s_lst_ho = [], [], [], []
        self.up_imp_s_lst_ho, self.down_imp_s_lst_ho = [], []

        self.q_imp_t_lst_ge, self.k_imp_t_lst_ge, self.v_imp_t_lst_ge, self.out_imp_t_lst_ge = [], [], [], []
        self.up_imp_t_lst_ge, self.down_imp_t_lst_ge = [], []

        self.q_imp_s_lst_ge, self.k_imp_s_lst_ge, self.v_imp_s_lst_ge, self.out_imp_s_lst_ge = [], [], [], []
        self.up_imp_s_lst_ge, self.down_imp_s_lst_ge = [], []

        self._loop_pattern(self.indication_loader_ho, current_type='holiday')

        self._loop_pattern(self.indication_loader_ge, current_type='general')

        self.k_t_index_ho_new, self.k_t_index_ge_new = self.k_t_index_ho, self.k_t_index_ge
        self.q_t_index_ho_new, self.q_t_index_ge_new = self.q_t_index_ho, self.q_t_index_ge
        self.v_t_index_ho_new, self.v_t_index_ge_new = self.v_t_index_ho, self.v_t_index_ge
        self.up_t_index_ho_new, self.up_t_index_ge_new = self.up_t_index_ho,  self.up_t_index_ge
        self.down_t_index_ho_new, self.down_t_index_ge_new = self.down_t_index_ho,  self.down_t_index_ge
        self.out_t_index_ho_new,  self.out_t_index_ge_new = self.out_t_index_ho,  self.out_t_index_ge

        self.k_s_index_ho_new, self.k_s_index_ge_new = self.k_s_index_ho, self.k_s_index_ge
        self.q_s_index_ho_new,  self.q_s_index_ge_new = self.q_s_index_ho, self.q_s_index_ge
        self.v_s_index_ho_new, self.v_s_index_ge_new = self.v_s_index_ho, self.v_s_index_ge
        self.up_s_index_ho_new,  self.up_s_index_ge_new = self.up_s_index_ho, self.up_s_index_ge
        self.down_s_index_ho_new,  self.down_s_index_ge_new = self.down_s_index_ho, self.down_s_index_ge
        self.out_s_index_ho_new, self.out_s_index_ge_new = self.out_s_index_ho, self.out_s_index_ge

    def _get_important_score(self, q, k, v, out, up, down, layer, name, param, q_imp, k_imp, v_imp, out_imp, up_imp,
                             down_imp):

        attn_neurons_len = int(self.model.model_dim * self.args.select_ratio)
        up_len = int(self.model.feed_forward_dim * self.args.select_ratio)
        down_len = attn_neurons_len

        if 'attn.FC_Q.weight' in name:
            if self.args.detect_func == 'f':
                importance = q[layer].detach().abs()
            elif self.args.detect_func == 'r':
                importance = q[layer].detach()
            elif self.args.detect_func == 'g':
                importance = param.grad.abs().sum(dim=1)
            elif self.args.detect_func == 'o':
                importance = (q[layer].detach().abs() * param.grad.abs().sum(dim=1))
            elif self.args.detect_func == 'n':
                feature_importance = torch.softmax(q[layer].detach(), dim=0)
                grad_importance = torch.softmax(param.grad.abs().sum(dim=1), dim=0)
                importance = feature_importance * grad_importance
            elif self.args.detect_func == 'a':
                importance = (q[layer].detach().abs() + param.grad.abs().sum(dim=1))
            top_neurons = importance.argsort(descending=True)[:attn_neurons_len]
            q_imp[layer] = top_neurons
        elif 'attn.FC_K.weight' in name:
            if self.args.detect_func == 'f':
                importance = k[layer].detach().abs()
            elif self.args.detect_func == 'r':
                importance = k[layer].detach()
            elif self.args.detect_func == 'g':
                importance = param.grad.abs().sum(dim=1)
            elif self.args.detect_func == 'o':
                importance = (k[layer].detach().abs() * param.grad.abs().sum(dim=1))
            elif self.args.detect_func == 'n':
                feature_importance = torch.softmax(k[layer].detach(), dim=0)
                grad_importance = torch.softmax(param.grad.abs().sum(dim=1), dim=0)
                importance = feature_importance * grad_importance
            elif self.args.detect_func == 'a':
                importance = (k[layer].detach() + param.grad.abs().sum(dim=1))

            top_neurons = importance.argsort(descending=True)[:attn_neurons_len]
            k_imp[layer] = top_neurons
        elif 'attn.FC_V.weight' in name:
            if self.args.detect_func == 'f':
                importance = v[layer].detach().abs()
            elif self.args.detect_func == 'r':
                importance = v[layer].detach()
            elif self.args.detect_func == 'g':
                importance = param.grad.abs().sum(dim=1)
            elif self.args.detect_func == 'o':
                importance = (v[layer].detach().abs() * param.grad.abs().sum(dim=1))
            elif self.args.detect_func == 'n':
                feature_importance = torch.softmax(v[layer].detach(), dim=0)
                grad_importance = torch.softmax(param.grad.abs().sum(dim=1), dim=0)
                importance = feature_importance * grad_importance
            elif self.args.detect_func == 'a':
                importance = (v[layer].detach() + param.grad.abs().sum(dim=1))

            top_neurons = importance.argsort(descending=True)[:attn_neurons_len]
            v_imp[layer] = top_neurons
        elif 'attn.out_proj.weight' in name:
            if self.args.detect_func == 'f':
                importance = out[layer].detach().abs()
            elif self.args.detect_func == 'r':
                importance = out[layer].detach()
            elif self.args.detect_func == 'g':
                importance = param.grad.abs().sum(dim=1)
            elif self.args.detect_func == 'o':
                importance = (out[layer].detach().abs() * param.grad.abs().sum(dim=1))
            elif self.args.detect_func == 'n':
                feature_importance = torch.softmax(out[layer].detach(), dim=0)
                grad_importance = torch.softmax(param.grad.abs().sum(dim=1), dim=0)
                importance = feature_importance * grad_importance
            elif self.args.detect_func == 'a':
                importance = (out[layer].detach() + param.grad.abs().sum(dim=1))
            top_neurons = importance.argsort(descending=True)[:attn_neurons_len]
            out_imp[layer] = top_neurons
        elif 'feed_forward_up.0.weight' in name:
            if self.args.detect_func == 'f':
                importance = up[layer].detach().abs()
            elif self.args.detect_func == 'r':
                importance = up[layer].detach()
            elif self.args.detect_func == 'g':
                importance = param.grad.abs().sum(dim=1)
            elif self.args.detect_func == 'o':
                importance = (up[layer].detach().abs() * param.grad.abs().sum(dim=1))
            elif self.args.detect_func == 'n':
                feature_importance = torch.softmax(up[layer].detach(), dim=0)
                grad_importance = torch.softmax(param.grad.abs().sum(dim=1), dim=0)
                importance = feature_importance * grad_importance
            elif self.args.detect_func == 'a':
                importance = (up[layer].detach() + param.grad.abs().sum(dim=1))

            top_neurons = importance.argsort(descending=True)[:up_len]
            up_imp[layer] = top_neurons
        elif 'feed_forward_down.weight' in name:
            if self.args.detect_func == 'f':
                importance = down[layer].detach().abs()
            elif self.args.detect_func == 'r':
                importance = down[layer].detach()
            elif self.args.detect_func == 'g':
                importance = param.grad.abs().sum(dim=1)
            elif self.args.detect_func == 'o':
                importance = (down[layer].detach().abs() * param.grad.abs().sum(dim=1))
            elif self.args.detect_func == 'n':
                feature_importance = torch.softmax(down[layer].detach(), dim=0)
                grad_importance = torch.softmax(param.grad.abs().sum(dim=1), dim=0)
                importance = feature_importance * grad_importance
            elif self.args.detect_func == 'a':
                importance = (down[layer].detach() + param.grad.abs().sum(dim=1))
            top_neurons = importance.argsort(descending=True)[:down_len]
            down_imp[layer] = top_neurons
        return q_imp, k_imp, v_imp, out_imp, up_imp, down_imp

    def set_important_score(self, t_param, s_param, t_param_save, s_param_save, current_type, index=None):

        (t_q, t_k, t_v, t_out, t_up, t_down), (s_q, s_k, s_v, s_out, s_up, s_down) = t_param, s_param

        (t_q_save, t_k_save, t_v_save, t_out_save, t_up_save, t_down_save), (s_q_save, s_k_save, s_v_save, s_out_save, s_up_save, s_down_save) = t_param_save, s_param_save

        save_dict = {'t_q': t_q_save,
                     't_k': t_k_save,
                     't_v': t_v_save,
                     't_out': t_out_save,
                     't_up': t_up_save,
                     't_down': t_down_save,
                     's_q': s_q_save,
                     's_k': s_k_save,
                     's_v': s_v_save,
                     's_out': s_out_save,
                     's_up': s_up_save,
                     's_down': s_down_save}

        current_path = self.args.log_dir + '/' + current_type

        if not os.path.exists(current_path):
            os.makedirs(current_path)
        else:
            current_file_path = os.path.join(current_path, str(index) + '_detect_activations.pkl')
            if not os.path.exists(current_file_path):
                with open(os.path.join(current_path, str(index) + '_detect_activations.pkl'), 'wb') as f:
                    pickle.dump(save_dict, f)

        attn_neurons_len = int(self.model.model_dim * self.args.select_ratio)
        up_len = int(self.model.feed_forward_dim * self.args.select_ratio)
        down_len = attn_neurons_len

        q_imp_t, k_imp_t, v_imp_t, out_imp_t = torch.zeros((self.model.num_layers, attn_neurons_len),
                                                           dtype=torch.int), torch.zeros(
            (self.model.num_layers, attn_neurons_len), dtype=torch.int), torch.zeros(
            (self.model.num_layers, attn_neurons_len), dtype=torch.int), torch.zeros(
            (self.model.num_layers, attn_neurons_len), dtype=torch.int)

        up_imp_t, down_imp_t = torch.zeros((self.model.num_layers, up_len), dtype=torch.int), torch.zeros(
            (self.model.num_layers, down_len), dtype=torch.int)

        q_imp_s, k_imp_s, v_imp_s, out_imp_s = torch.zeros((self.model.num_layers, attn_neurons_len),
                                                           dtype=torch.int), torch.zeros(
            (self.model.num_layers, attn_neurons_len), dtype=torch.int), torch.zeros(
            (self.model.num_layers, attn_neurons_len), dtype=torch.int), torch.zeros(
            (self.model.num_layers, attn_neurons_len), dtype=torch.int)
        up_imp_s, down_imp_s = torch.zeros((self.model.num_layers, up_len), dtype=torch.int), torch.zeros(
            (self.model.num_layers, down_len), dtype=torch.int)

        for name, param in self.model.named_parameters():
            match = re.search(r'layers_(\w)\.(\d+)', name)
            if match:
                type = match.group(1)
                layer = int(match.group(2))
                if type == 't':
                    q_imp_t, k_imp_t, v_imp_t, out_imp_t, up_imp_t, down_imp_t = self._get_important_score(t_q, t_k,
                                                                                                           t_v, t_out,
                                                                                                           t_up, t_down,
                                                                                                           layer, name,
                                                                                                           param,
                                                                                                           q_imp_t,
                                                                                                           k_imp_t,
                                                                                                           v_imp_t,
                                                                                                           out_imp_t,
                                                                                                           up_imp_t,
                                                                                                           down_imp_t)
                elif type == 's':
                    q_imp_s, k_imp_s, v_imp_s, out_imp_s, up_imp_s, down_imp_s = self._get_important_score(s_q, s_k,
                                                                                                           s_v, s_out,
                                                                                                           s_up, s_down,
                                                                                                           layer, name,
                                                                                                           param,
                                                                                                           q_imp_s,
                                                                                                           k_imp_s,
                                                                                                           v_imp_s,
                                                                                                           out_imp_s,
                                                                                                           up_imp_s,
                                                                                                           down_imp_s)

        if current_type == 'holiday':
            self.q_imp_t_lst_ho.append(q_imp_t)
            self.k_imp_t_lst_ho.append(k_imp_t)
            self.v_imp_t_lst_ho.append(v_imp_t)
            self.out_imp_t_lst_ho.append(out_imp_t)
            self.up_imp_t_lst_ho.append(up_imp_t)
            self.down_imp_t_lst_ho.append(down_imp_t)

            self.q_imp_s_lst_ho.append(q_imp_s)
            self.k_imp_s_lst_ho.append(k_imp_s)
            self.v_imp_s_lst_ho.append(v_imp_s)
            self.out_imp_s_lst_ho.append(out_imp_s)
            self.up_imp_s_lst_ho.append(up_imp_s)
            self.down_imp_s_lst_ho.append(down_imp_s)
        elif current_type == 'general':
            self.q_imp_t_lst_ge.append(q_imp_t)
            self.k_imp_t_lst_ge.append(k_imp_t)
            self.v_imp_t_lst_ge.append(v_imp_t)
            self.out_imp_t_lst_ge.append(out_imp_t)
            self.up_imp_t_lst_ge.append(up_imp_t)
            self.down_imp_t_lst_ge.append(down_imp_t)

            self.q_imp_s_lst_ge.append(q_imp_s)
            self.k_imp_s_lst_ge.append(k_imp_s)
            self.v_imp_s_lst_ge.append(v_imp_s)
            self.out_imp_s_lst_ge.append(out_imp_s)
            self.up_imp_s_lst_ge.append(up_imp_s)
            self.down_imp_s_lst_ge.append(down_imp_s)

    def _update_weights_random(self, zero_ratio, is_rand=True):
        select_param = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                is_updated = True
                if not is_rand:
                    is_updated = 'attn.FC_Q.weight' in name or 'attn.FC_K.weight' in name or 'attn.FC_V.weight' in name or 'attn.out_proj.weight' in name or 'feed_forward_up.0.weight' in name or 'feed_forward_down.weight' in name
                if is_updated:
                    param_size = param.numel()
                    num_params_layer = int(param_size * zero_ratio)
                    select_param += num_params_layer
                    if num_params_layer > 0:
                        indices = random.sample(range(param_size), num_params_layer)
                        indices = torch.tensor(indices, dtype=torch.long)
                        mask = torch.ones(param_size, dtype=torch.bool)
                        mask[indices] = False
                        mask = mask.view(param.size()).to(self.device)
                        param.data.mul_(mask)
        return select_param

    def _select_update_weights(self, pattern_type='holiday', is_update=False, not_specific=True):
        select_param = 0
        tot_param = 0
        tot_key_param = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                tot_param += param.numel()
                match = re.search(r'layers_(\w)\.(\d+)', name)
                if match:
                    type = match.group(1)
                    layer = int(match.group(2))
                    if type == 't':
                        if 'attn.FC_Q.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.q_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.q_t_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.FC_K.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.k_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.k_t_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.FC_V.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.v_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.v_t_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.out_proj.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.out_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.out_t_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'feed_forward_up.0.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.up_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.up_t_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'feed_forward_down.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.down_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.down_t_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                    elif type == 's':
                        if 'attn.FC_Q.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.q_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.q_s_index_ge_new[layer]
                            param.data[list(tune_index)] = 0
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.FC_K.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.k_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.k_s_index_ge_new[layer]
                            param.data[list(tune_index)] = 0
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.FC_V.weight' in name:
                            tot_key_param += param.numel()

                            if pattern_type == 'holiday':
                                tune_index = self.v_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.v_s_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.out_proj.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.out_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.out_s_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'feed_forward_up.0.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.up_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.up_s_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'feed_forward_down.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.down_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.down_s_index_ge_new[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
        return select_param, tot_param, tot_key_param


    def _select_param(self, pattern_type='holiday', is_update=False, not_specific=True):
        select_param = 0
        tot_param = 0
        tot_key_param = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                tot_param += param.numel()
                match = re.search(r'layers_(\w)\.(\d+)', name)
                if match:
                    type = match.group(1)
                    layer = int(match.group(2))
                    if type == 't':
                        if 'attn.FC_Q.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.q_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.q_t_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.q_t_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.q_t_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.FC_K.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.k_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.k_t_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.k_t_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.k_t_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.FC_V.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.v_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.v_t_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.v_t_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.v_t_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.out_proj.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.out_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.out_t_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.out_t_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.out_t_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'feed_forward_up.0.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.up_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.up_t_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.up_t_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.up_t_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update:
                                    param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'feed_forward_down.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.down_t_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.down_t_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.down_t_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.down_t_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                    elif type == 's':
                        if 'attn.FC_Q.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.q_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.q_s_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.q_s_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.q_s_index_ho_new_rm[layer]
                            param.data[list(tune_index)] = 0
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.FC_K.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.k_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.k_s_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.k_s_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.k_s_index_ho_new_rm[layer]
                            param.data[list(tune_index)] = 0
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.FC_V.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.v_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.v_s_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.v_s_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.v_s_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'attn.out_proj.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.out_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.out_s_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.out_s_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.out_s_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'feed_forward_up.0.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.up_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.up_s_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.up_s_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.up_s_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
                        elif 'feed_forward_down.weight' in name:
                            tot_key_param += param.numel()
                            if pattern_type == 'holiday':
                                tune_index = self.down_s_index_ho_new[layer]
                            elif pattern_type == 'general':
                                tune_index = self.down_s_index_ge_new[layer]
                            elif pattern_type == 'common':
                                tune_index = self.down_s_index_common[layer]
                            elif pattern_type == 'holiday_specific':
                                tune_index = self.down_s_index_ho_new_rm[layer]
                            if len(list(tune_index)) != 0:
                                if not_specific:
                                    index_set = set(range(len(param)))
                                    result_set = list(index_set - tune_index)
                                    tune_index = random.sample(result_set, len(tune_index))
                                if is_update: param.data[list(tune_index)] = 0
                                select_param += param.data[list(tune_index)].numel()
        return select_param, tot_param, tot_key_param
    def get_index_map(self):
        if self.args.wo_t:
            index_mappings = {
                's': {
                    'attn.FC_Q.weight': {
                        'holiday': self.q_s_index_ho_new,
                        'general': self.q_s_index_ge_new,
                    },
                    'attn.FC_K.weight': {
                        'holiday': self.k_s_index_ho_new,
                        'general': self.k_s_index_ge_new,
                    },
                    'attn.FC_V.weight': {
                        'holiday': self.v_s_index_ho_new,
                        'general': self.v_s_index_ge_new,
                    },
                    'attn.out_proj.weight': {
                        'holiday': self.out_s_index_ho_new,
                        'general': self.out_s_index_ge_new,
                    },
                    'feed_forward_up.0.weight': {
                        'holiday': self.up_s_index_ho_new,
                        'general': self.up_s_index_ge_new,
                    },
                    'feed_forward_down.weight': {
                        'holiday': self.down_s_index_ho_new,
                        'general': self.down_s_index_ge_new,
                    },
                },
            }
        elif self.args.wo_s:
            index_mappings = {
                't': {
                    'attn.FC_Q.weight': {
                        'holiday': self.q_t_index_ho_new,
                        'general': self.q_t_index_ge_new,
                    },
                    'attn.FC_K.weight': {
                        'holiday': self.k_t_index_ho_new,
                        'general': self.k_t_index_ge_new,
                    },
                    'attn.FC_V.weight': {
                        'holiday': self.v_t_index_ho_new,
                        'general': self.v_t_index_ge_new,
                    },
                    'attn.out_proj.weight': {
                        'holiday': self.out_t_index_ho_new,
                        'general': self.out_t_index_ge_new,
                    },
                    'feed_forward_up.0.weight': {
                        'holiday': self.up_t_index_ho_new,
                        'general': self.up_t_index_ge_new,
                    },
                    'feed_forward_down.weight': {
                        'holiday': self.down_t_index_ho_new,
                        'general': self.down_t_index_ge_new,
                    },
                }
            }
        elif self.args.wo_attn:
            index_mappings = {
                't': {
                    'attn.out_proj.weight': {
                        'holiday': self.out_t_index_ho_new,
                        'general': self.out_t_index_ge_new,
                    },
                    'feed_forward_up.0.weight': {
                        'holiday': self.up_t_index_ho_new,
                        'general': self.up_t_index_ge_new,
                    },
                    'feed_forward_down.weight': {
                        'holiday': self.down_t_index_ho_new,
                        'general': self.down_t_index_ge_new,
                    },
                },
                's': {
                    'attn.out_proj.weight': {
                        'holiday': self.out_s_index_ho_new,
                        'general': self.out_s_index_ge_new,
                    },
                    'feed_forward_up.0.weight': {
                        'holiday': self.up_s_index_ho_new,
                        'general': self.up_s_index_ge_new,
                    },
                    'feed_forward_down.weight': {
                        'holiday': self.down_s_index_ho_new,
                        'general': self.down_s_index_ge_new,
                    },
                },
            }
        elif self.args.wo_ffn:
            index_mappings = {
                't': {
                    'attn.FC_Q.weight': {
                        'holiday': self.q_t_index_ho_new,
                        'general': self.q_t_index_ge_new,
                    },
                    'attn.FC_K.weight': {
                        'holiday': self.k_t_index_ho_new,
                        'general': self.k_t_index_ge_new,
                    },
                    'attn.FC_V.weight': {
                        'holiday': self.v_t_index_ho_new,
                        'general': self.v_t_index_ge_new,
                    }
                },
                's': {
                    'attn.FC_Q.weight': {
                        'holiday': self.q_s_index_ho_new,
                        'general': self.q_s_index_ge_new,
                    },
                    'attn.FC_K.weight': {
                        'holiday': self.k_s_index_ho_new,
                        'general': self.k_s_index_ge_new,
                    },
                    'attn.FC_V.weight': {
                        'holiday': self.v_s_index_ho_new,
                        'general': self.v_s_index_ge_new,
                    }
                },
            }
        else:
            index_mappings = {
                't': {
                    'attn.FC_Q.weight': {
                        'holiday': self.q_t_index_ho_new,
                        'general': self.q_t_index_ge_new,
                    },
                    'attn.FC_K.weight': {
                        'holiday': self.k_t_index_ho_new,
                        'general': self.k_t_index_ge_new,
                    },
                    'attn.FC_V.weight': {
                        'holiday': self.v_t_index_ho_new,
                        'general': self.v_t_index_ge_new,
                    },
                    'attn.out_proj.weight': {
                        'holiday': self.out_t_index_ho_new,
                        'general': self.out_t_index_ge_new,
                    },
                    'feed_forward_up.0.weight': {
                        'holiday': self.up_t_index_ho_new,
                        'general': self.up_t_index_ge_new,
                    },
                    'feed_forward_down.weight': {
                        'holiday': self.down_t_index_ho_new,
                        'general': self.down_t_index_ge_new,
                    },
                },
                's': {
                    'attn.FC_Q.weight': {
                        'holiday': self.q_s_index_ho_new,
                        'general': self.q_s_index_ge_new,
                    },
                    'attn.FC_K.weight': {
                        'holiday': self.k_s_index_ho_new,
                        'general': self.k_s_index_ge_new,
                    },
                    'attn.FC_V.weight': {
                        'holiday': self.v_s_index_ho_new,
                        'general': self.v_s_index_ge_new,
                    },
                    'attn.out_proj.weight': {
                        'holiday': self.out_s_index_ho_new,
                        'general': self.out_s_index_ge_new,
                    },
                    'feed_forward_up.0.weight': {
                        'holiday': self.up_s_index_ho_new,
                        'general': self.up_s_index_ge_new,
                    },
                    'feed_forward_down.weight': {
                        'holiday': self.down_s_index_ho_new,
                        'general': self.down_s_index_ge_new,
                    },
                },
            }
        return index_mappings
        
    def reverse_finetune(self, current_model, setting, logger, data_loader, finetune_type='general'):
        path = os.path.join(self.args.checkpoints, setting)  # 创建模型保存路径
        if not os.path.exists(path):
            os.makedirs(path)  # 如果路径不存在，则创建它
            
        # 使用常规数据集对模型进行训练
        opt = optim.AdamW(current_model.parameters(), lr=self.args.learning_rate)  # 定义优化器

        if self.args.use_amp:
            amp_scaler = torch.cuda.amp.GradScaler()  # 如果启用了混合精度训练，则创建GradScaler对象用于缩放梯度

        # 初始化早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 定义训练周期数
        #num_epochs = self.args.unfreeze_train_epochs
        num_epochs = self.args.reverse_epochs

        for epoch in range(num_epochs):  # 遍历训练的每个周期
            self.model.train()  # 将模型设置为训练模式
            train_loss = []  # 用于存储每个批次的训练损失

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):  # 遍历数据加载器中的每个批次
                opt.zero_grad()  # 清除之前计算的梯度

                pred, true = self._process_one_batch_model(current_model, batch_x, batch_y, batch_x_mark, batch_y_mark)  # 处理一个批次的数据，获取预测值和真实值
                pred = self.scaler.inverse_transform(pred)  # 对预测值进行逆变换，还原到原始数据的尺度
                true = self.scaler.inverse_transform(true)  # 对真实值进行逆变换，还原到原始数据的尺度

                curr_loss = self.loss(pred, true)  # 计算当前批次的损失
                train_loss.append(curr_loss.item())  # 将当前批次的损失添加到训练损失列表中

                if self.args.use_amp:
                    # 如果启用了混合精度训练，则缩放损失并反向传播
                    amp_scaler.scale(curr_loss).backward()
                    amp_scaler.unscale_(opt)  # 反缩放梯度
                    current_model = self.update_model_grad_reverse(current_model, finetune_type)  # 更新模型的梯度
                    amp_scaler.step(opt)  # 更新优化器的参数
                    amp_scaler.update()  # 更新GradScaler对象的状态
                else:
                    curr_loss.backward()  # 反向传播计算梯度
                    current_model = self.update_model_grad_reverse(current_model, finetune_type)  # 更新模型的梯度
                    opt.step()  # 更新优化器的参数

            # 计算当前周期的平均训练损失
            avg_train_loss = np.average(train_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.7f}")

            # 在验证集上评估模型性能
            vali_loss = self.vali(self.vali_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Vali Loss: {vali_loss:.7f}")

            # 检查是否需要早停
            early_stopping(vali_loss, current_model, path)  # 检查是否需要早停
            if early_stopping.early_stop:
                print("Early stopping")
                logger.info("Early stopping")
                break

            # 调整学习率（如果有需要）
            adjust_learning_rate(opt, epoch + 1, self.args)

        # 微调完成后，对模型进行测试并记录日志信息
        self.test_model(self.model, self.debug_loader_ho, log_info='holiday data after reverse train', log_name='after_reverse_holiday')
        self.test_model(self.model, self.debug_loader_ge, log_info='general after reverse train', log_name='after_reverse_general')
        self.test_model(self.model, self.test_loader, log_info='all test after reverse train', log_name='after_reverse_all')

    def update_model_grad_unfreeze(self, model, finetune_type):
        # 更新模型梯度，根据选择的参数进行梯度更新。
        index_mappings = self.get_index_map()  # 获取索引映射，用于确定哪些神经元与特定模式相关
        for name, param in model.named_parameters():  # 遍历模型的所有参数
            match = re.search(r'layers_(\w)\.(\d+)', name)  # 使用正则表达式匹配参数名中的层类型和层号
            if match:
                layer_type = match.group(1)  # 获取层类型（例如 't' 表示时间层，'s' 表示空间层）
                layer_num = int(match.group(2))  # 获取层的编号
                layer_indices = index_mappings.get(layer_type, {})  # 获取当前层类型的索引映射
                param_matched = False  # 标志表示当前参数是否匹配到索引映射中的参数名
                for param_name, finetune_indices in layer_indices.items():  # 遍历当前层类型的参数名和微调索引
                    if param_name in name:  # 如果当前参数名包含在映射的参数名中
                        tune_index = finetune_indices[finetune_type][layer_num]  # 获取与微调类型和层号对应的神经元索引
                        mask = torch.ones(param.size(0), dtype=torch.bool, device=param.device)  # 创建一个与参数大小相同的掩码，默认值为 True
                        mask[list(tune_index)] = False  # 将与微调相关的神经元索引位置设置为 False
                        param.grad[mask] = 0  # 将不需要更新的神经元梯度置零
                        param_matched = True  # 标志设置为 True，表示找到了匹配的参数
                        break
                if not param_matched:  # 如果当前参数没有匹配到任何索引映射
                    param.grad.zero_()  # 将参数的梯度全部置零
                if  param_matched:  # 如果当前参数匹配到任何索引映射
                    param.grad.zero_()  # 将参数的梯度全部置零
            else:
                param.grad.zero_()  # 如果参数名不匹配层类型和层号的正则表达式，则将梯度置零
        return model  # 返回更新梯度后的模型
        
    def update_model_grad_reverse(self, model, finetune_type):
        # 更新模型梯度，根据选择的参数进行梯度更新。
        index_mappings = self.get_index_map()  # 获取索引映射，用于确定哪些神经元与特定模式相关
        for name, param in model.named_parameters():  # 遍历模型的所有参数
            match = re.search(r'layers_(\w)\.(\d+)', name)  # 使用正则表达式匹配参数名中的层类型和层号
            if match:
                layer_type = match.group(1)  # 获取层类型（例如 't' 表示时间层，'s' 表示空间层）
                layer_num = int(match.group(2))  # 获取层的编号
                layer_indices = index_mappings.get(layer_type, {})  # 获取当前层类型的索引映射
                param_matched = False  # 标志表示当前参数是否匹配到索引映射中的参数名
                for param_name, finetune_indices in layer_indices.items():  # 遍历当前层类型的参数名和微调索引
                    if param_name in name:  # 如果当前参数名包含在映射的参数名中
                        tune_index = finetune_indices[finetune_type][layer_num]  # 获取与微调类型和层号对应的神经元索引
                        mask = torch.ones(param.size(0), dtype=torch.bool, device=param.device)  # 创建一个与参数大小相同的掩码，默认值为 True
                        mask[list(tune_index)] = False  # 将与微调相关的神经元索引位置设置为 False
                        param.grad[mask] = 0  # 将不需要更新的神经元梯度置零
                        param_matched = True  # 标志设置为 True，表示找到了匹配的参数
                        break
                # if not param_matched:  # 如果当前参数没有匹配到任何索引映射
                #     param.grad.zero_()  # 将参数的梯度全部置零
                if  param_matched:  # 如果当前参数匹配到任何索引映射
                    param.grad.zero_()  # 将参数的梯度全部置零
            # else:
            #     param.grad.zero_()  # 如果参数名不匹配层类型和层号的正则表达式，则将梯度置零
        return model  # 返回更新梯度后的模型

    def unfreeze_train(self, setting, logger):
        # 对检测到的模式神经元进行微调，以提高模型在特定模式上的性能。
        self.setting = setting  # 保存当前的实验设置
        self.logger = logger  # 保存日志记录器

        # 创建模型保存路径
        path = os.path.join(self.args.checkpoints, setting)
        # 在路径后添加 'finetune_type' 子目录
        path = path + '/' + 'finetune_type'
        # 如果路径不存在，则创建它
        if not os.path.exists(path):
            os.makedirs(path)

        # 加载预训练模型
        self.model.load_state_dict(torch.load(self.args.checkpoint_path, map_location=self.args.device))

        # 创建节假日模型和一般模型的副本
        holiday_model = copy.deepcopy(self.model).to(self.device)
        general_model = copy.deepcopy(self.model).to(self.device)

        # 如果有微调样本，则对节假日模型进行微调
        if self.args.finetune_sample_num != 0:
            self.process_finetune(holiday_model, self.retrain_loader_ho, finetune_type='holiday')

        # 重新初始化优化器
        self.opt = self._select_optimizer()

        # 调用 process_unfreeze 方法进行全局训练
        self.process_unfreeze(setting,logger,self.train_loader,self.model,finetune_type='holiday')


        return self.model  # 返回模型实例以便在调用该方法后可以继续使用或保存模型
    
    def process_unfreeze(self,setting,logger,data_loader,current_model,finetune_type):
        path = os.path.join(self.args.checkpoints, setting)  # 创建模型保存路径
        if not os.path.exists(path):
            os.makedirs(path)  # 如果路径不存在，则创建它
            
        # 使用完整数据集对模型进行训练
        opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)  # 定义优化器

        if self.args.use_amp:
            amp_scaler = torch.cuda.amp.GradScaler()  # 如果启用了混合精度训练，则创建GradScaler对象用于缩放梯度

        # 初始化早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 定义训练周期数
        num_epochs = self.args.unfreeze_train_epochs
        #num_epochs = self.args.reverse_epochs

        for epoch in range(num_epochs):  # 遍历训练的每个周期
            self.model.train()  # 将模型设置为训练模式
            train_loss = []  # 用于存储每个批次的训练损失

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):  # 遍历数据加载器中的每个批次
                opt.zero_grad()  # 清除之前计算的梯度

                pred, true = self._process_one_batch_model(current_model, batch_x, batch_y, batch_x_mark, batch_y_mark)  # 处理一个批次的数据，获取预测值和真实值
                pred = self.scaler.inverse_transform(pred)  # 对预测值进行逆变换，还原到原始数据的尺度
                true = self.scaler.inverse_transform(true)  # 对真实值进行逆变换，还原到原始数据的尺度

                curr_loss = self.loss(pred, true)  # 计算当前批次的损失
                train_loss.append(curr_loss.item())  # 将当前批次的损失添加到训练损失列表中

                if self.args.use_amp:
                    # 如果启用了混合精度训练，则缩放损失并反向传播
                    amp_scaler.scale(curr_loss).backward()
                    amp_scaler.unscale_(opt)  # 反缩放梯度
                    #current_model = self.update_model_grad_unfreeze(current_model, finetune_type)  # 更新模型的梯度
                    amp_scaler.step(opt)  # 更新优化器的参数
                    amp_scaler.update()  # 更新GradScaler对象的状态
                else:
                    curr_loss.backward()  # 反向传播计算梯度
                    #current_model = self.update_model_grad_unfreeze(current_model, finetune_type)  # 更新模型的梯度
                    opt.step()  # 更新优化器的参数

                # if self.args.use_amp:
                #     # 如果启用了混合精度训练，则缩放损失并反向传播
                #     amp_scaler.scale(curr_loss).backward()
                #     amp_scaler.unscale_(opt)  # 反缩放梯度
                #     amp_scaler.step(opt)  # 更新优化器的参数
                #     amp_scaler.update()  # 更新GradScaler对象的状态
                # else:
                #     curr_loss.backward()  # 反向传播计算梯度
                #     self.opt.step()  # 更新优化器的参数

            # 计算当前周期的平均训练损失
            avg_train_loss = np.average(train_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.7f}")

            # 在验证集上评估模型性能
            vali_loss = self.vali(self.vali_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Vali Loss: {vali_loss:.7f}")

            # 检查是否需要早停
            early_stopping(vali_loss, self.model, path)  # 检查是否需要早停
            if early_stopping.early_stop:
                print("Early stopping")
                logger.info("Early stopping")
                break

            # 调整学习率（如果有需要）
            adjust_learning_rate(opt, epoch + 1, self.args)

        # 微调完成后，对模型进行测试并记录日志信息
        self.test_model(self.model, self.debug_loader_ho, log_info='holiday data after unfreeze train', log_name='after_unfreeze_holiday')
        self.test_model(self.model, self.debug_loader_ge, log_info='general after unfreeze train', log_name='after_unfreeze_general')
        self.test_model(self.model, self.test_loader, log_info='all test after unfreeze train', log_name='after_unfreeze_all')

    def update_model_grad(self, model, finetune_type):
    # 更新模型梯度，根据选择的参数进行梯度更新。
        index_mappings = self.get_index_map()  # 获取索引映射，用于确定哪些神经元与特定模式相关
        for name, param in model.named_parameters():  # 遍历模型的所有参数
            match = re.search(r'layers_(\w)\.(\d+)', name)  # 使用正则表达式匹配参数名中的层类型和层号
            if match:
                layer_type = match.group(1)  # 获取层类型（例如 't' 表示时间层，'s' 表示空间层）
                layer_num = int(match.group(2))  # 获取层的编号
                layer_indices = index_mappings.get(layer_type, {})  # 获取当前层类型的索引映射
                param_matched = False  # 标志表示当前参数是否匹配到索引映射中的参数名
                for param_name, finetune_indices in layer_indices.items():  # 遍历当前层类型的参数名和微调索引
                    if param_name in name:  # 如果当前参数名包含在映射的参数名中
                        tune_index = finetune_indices[finetune_type][layer_num]  # 获取与微调类型和层号对应的神经元索引
                        mask = torch.ones(param.size(0), dtype=torch.bool, device=param.device)  # 创建一个与参数大小相同的掩码，默认值为 True
                        mask[list(tune_index)] = False  # 将与微调相关的神经元索引位置设置为 False
                        param.grad[mask] = 0  # 将不需要更新的神经元梯度置零
                        param_matched = True  # 标志设置为 True，表示找到了匹配的参数
                        break
                if not param_matched:  # 如果当前参数没有匹配到任何索引映射
                    param.grad.zero_()  # 将参数的梯度全部置零
                # if  param_matched:  # 如果当前参数匹配到任何索引映射
                #     param.grad.zero_()  # 将参数的梯度全部置零
            else:
                param.grad.zero_()  # 如果参数名不匹配层类型和层号的正则表达式，则将梯度置零
        return model  # 返回更新梯度后的模型

    def process_finetune(self, current_model, data_loader, finetune_type):
        # 对模型进行微调，使用特定数据集更新模型参数。
        opt = optim.AdamW(current_model.parameters(), lr=self.args.finetune_learning_rate)  # 定义优化器，使用AdamW算法和指定的学习率

        if self.args.use_amp:
            amp_scaler = torch.cuda.amp.GradScaler()  # 如果启用了混合精度训练，则创建GradScaler对象用于缩放梯度

        for epoch in range(self.args.finetune_epochs):  # 遍历微调的每个周期
            current_model.train()  # 将模型设置为训练模式

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):  # 遍历数据加载器中的每个批次
                opt.zero_grad()  # 清除之前计算的梯度

                pred, true = self._process_one_batch_model(current_model, batch_x, batch_y, batch_x_mark, batch_y_mark)  # 处理一个批次的数据，获取预测值和真实值
                pred = self.scaler.inverse_transform(pred)  # 对预测值进行逆变换，还原到原始数据的尺度
                true = self.scaler.inverse_transform(true)  # 对真实值进行逆变换，还原到原始数据的尺度

                curr_loss = self.loss(pred, true)  # 计算当前批次的损失

                if self.args.use_amp:
                    # 如果启用了混合精度训练，则缩放损失并反向传播
                    amp_scaler.scale(curr_loss).backward()
                    amp_scaler.unscale_(opt)  # 反缩放梯度
                    current_model = self.update_model_grad(current_model, finetune_type)  # 更新模型的梯度
                    amp_scaler.step(opt)  # 更新优化器的参数
                    amp_scaler.update()  # 更新GradScaler对象的状态
                else:
                    curr_loss.backward()  # 反向传播计算梯度
                    current_model = self.update_model_grad(current_model, finetune_type)  # 更新模型的梯度
                    opt.step()  # 更新优化器的参数

        # 微调完成后，对模型进行测试并记录日志信息
        self.test_model(current_model, self.debug_loader_ho, log_info='holiday data after finetune with type ' + finetune_type, log_name='after_finetune_holiday_'+finetune_type)
        self.test_model(current_model, self.debug_loader_ge, log_info='general after finetune with type ' + finetune_type, log_name='after_finetune_general_'+finetune_type)
        self.test_model(current_model, self.test_loader, log_info='all test after finetune with type ' + finetune_type, log_name='after_finetune_all_'+finetune_type)

        return current_model  # 返回微调后的模型
    def reverse_train(self, setting, logger):
        # 对检测到的模式神经元进行微调，以提高模型在特定模式上的性能。
        self.setting = setting  # 保存当前的实验设置
        self.logger = logger  # 保存日志记录器

        # 创建模型保存路径
        path = os.path.join(self.args.checkpoints, setting)
        # 在路径后添加 'finetune_type' 子目录
        path = path + '/' + 'finetune_type'
        # 如果路径不存在，则创建它
        if not os.path.exists(path):
            os.makedirs(path)

        # 加载预训练模型
        self.model.load_state_dict(torch.load(self.args.checkpoint_path, map_location=self.args.device))

        # 创建节假日模型和一般模型的副本
        holiday_model = copy.deepcopy(self.model).to(self.device)
        general_model = copy.deepcopy(self.model).to(self.device)

        # 如果有微调样本，则对节假日模型进行微调
        if self.args.finetune_sample_num != 0:
            self.process_finetune(holiday_model, self.retrain_loader_ho, finetune_type='holiday')

         # 打印数据加载器的样本数量
        print("Train Loader Sample Count:", len(self.train_loader.dataset))
        print("Retrain Loader GE Sample Count:", len(self.retrain_loader_ge.dataset))
        print("Retrain Loader Ho Sample Count:", len(self.retrain_loader_ho.dataset))

        # 重新初始化优化器
        self.opt = self._select_optimizer()

        # 调用 process_unfreeze 方法进行全局训练
        self.reverse_finetune(self.model, setting, logger, self.train_loader, finetune_type='general')

        return self.model  # 返回模型实例以便在调用该方法后可以继续使用或保存模型

    def finetune(self, setting, logger):
        # 对检测到的模式神经元进行微调，以提高模型在特定模式上的性能。
        self.setting = setting  # 保存当前的实验设置
        self.logger = logger  # 保存日志记录器

        # 创建模型保存路径
        path = os.path.join(self.args.checkpoints, setting)
        # 在路径后添加 'finetune_type' 子目录
        path = path + '/' + 'finetune_type'
        # 如果路径不存在，则创建它
        if not os.path.exists(path):
            os.makedirs(path)

        # 加载预训练模型
        self.model.load_state_dict(torch.load(self.args.checkpoint_path, map_location=self.args.device))

        # 创建节假日模型和一般模型的副本
        holiday_model = copy.deepcopy(self.model).to(self.device)
        general_model = copy.deepcopy(self.model).to(self.device)

        # 如果有微调样本，则对节假日模型进行微调
        if self.args.finetune_sample_num != 0:
            self.process_finetune(holiday_model, self.retrain_loader_ho, finetune_type='holiday')

        # 返回原始模型，假设微调只针对节假日模型
        return self.model  # 返回模型实例以便在调用该方法后可以继续使用或保存模型

    def verify(self, setting, logger, pattern_type='holiday'):

        path = os.path.join(self.args.checkpoints, setting)

        path = path + '/' + pattern_type

        if not os.path.exists(path):
            os.makedirs(path)

        best_model_path = self.args.checkpoint_path
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.args.device))

        self.setting = setting
        self.logger = logger

        self.test_model(self.model, self.test_loader, log_info='all test before verify ', log_name='before_verify_all')
        self.test_model(self.model, self.debug_loader_ho, log_info='holiday before verify ', log_name='before_verify_holiday')
        self.test_model(self.model, self.debug_loader_ge, log_info='general before verify ', log_name='before_verify_general')

        self.model.eval()

        if self.args.deactivate_type == 'none':
            select_param, tot_param, tot_key_param = self._select_update_weights(pattern_type=pattern_type, is_update=True,
                                                                  not_specific=True)
            log = 'not select param: {}, tot_param:{}, percentage: {}'.format(select_param, tot_param,
                                                                              select_param / tot_param)
            self.logger.info(log)
        elif self.args.deactivate_type == 'random':
            select_param, tot_param, tot_key_param = self._select_update_weights(pattern_type=pattern_type, is_update=False,
                                                                  not_specific=False)
            zero_ratio = select_param / tot_key_param
            select_param_random = self._update_weights_random(zero_ratio, is_rand=False)
            log = 'random select param: {}, tot_param:{}, percentage: {}, random_select: {}, percentage: {}'.format(
                select_param, tot_param,
                select_param / tot_param, select_param_random, select_param_random / tot_param)
            self.logger.info(log)
        else:
            select_param, tot_param, tot_key_param = self._select_update_weights(pattern_type=pattern_type, is_update=True, not_specific=False)
            log = 'update select param: {}, tot_param:{}, percentage: {}'.format(select_param, tot_param,
                                                                                 select_param / tot_param)
            self.logger.info(log)

        self.test_model(self.model, self.test_loader, log_info='all test after verify ', log_name='after_verify_all')
        self.test_model(self.model, self.debug_loader_ho, log_info='holiday after verify ', log_name='after_verify_holiday')
        self.test_model(self.model, self.debug_loader_ge, log_info='general after verify ', log_name='after_verify_general')

        return self.model
