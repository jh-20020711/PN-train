import argparse
import os
import torch
import yaml
import uuid
import datetime
import importlib
from utils.logging import get_logger
from utils.tools import init_dl_program, StandardScaler
import time

parser = argparse.ArgumentParser(description='[PN-Train] Pattern Neuron guided Training')

parser.add_argument('--data', type=str, default='metro-traffic', help='data')
parser.add_argument('--method', type=str, default='pn')
parser.add_argument('--mode', type=str, default='train', help='train, detect, finetune or test')
parser.add_argument('--finetune_sample_num', type=int, default=10)
parser.add_argument('--detect_sample_num', type=int, default=30)
parser.add_argument('--select_ratio', type=float, default=0.5, help='learning rate during test')

parser.add_argument('--deactivate_type', type=str, default='none', choices=['random', 'none', 'pn-train'])
parser.add_argument('--detect_func', type=str, default='r', choices=['o', 'a', 'f', 'g', 'n', 'r'])
parser.add_argument('--detect_type', type=str, default='holiday', choices=['holiday', 'general'])

parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--finetune_learning_rate', type=float, default=0.002, help='pattern neuron optimizer learning rate')
parser.add_argument('--finetune_epochs', type=int, default=1, help='train epochs')

parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--save_path', type=str, default='./results/', help='the path to save the output')
parser.add_argument('--adj_filename', type=str, default='', help='the adj file path')
parser.add_argument('--checkpoint_path', type=str, default='', help='pretrain checkpoint path')

parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_bsz', type=int, default=32)

parser.add_argument('--wo_t', action='store_true', help='whether not to finetune temporal transformer')
parser.add_argument('--wo_s', action='store_true', help='whether not to finetune spatio transformer')
parser.add_argument('--wo_attn', action='store_true', help='whether not to finetune attn in transformer')
parser.add_argument('--wo_ffn', action='store_true', help='whether not to finetune fnn in transformer')

parser.add_argument('--exp_name', type=str, default='')

args = parser.parse_args()

with open(f"models/PN-Train.yaml", "r") as f:
    cfg = yaml.safe_load(f)
cfg = cfg[args.data]

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.test_bsz = cfg['batch_size'] if args.test_bsz == -1 else args.test_bsz
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

for key, value in cfg.items():
    setattr(args, key, value)

Exp = getattr(importlib.import_module('exp.exp_{}'.format(args.method)), 'Exp')

def get_test_results(setting, logger):
    test_time = time.time()

    exp.test(setting, logger)

    torch.cuda.empty_cache()

for ii in range(args.itr):
    print('\n ====== Run {} ====='.format(ii))
    # setting record of experiments
    method_name = args.method
    uid = uuid.uuid4().hex[:4]
    suffix = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + uid
    setting = '{}-{}-rt{}-id{}'.format(args.seed, args.mode, args.finetune_sample_num, args.detect_sample_num)

    seed = args.seed + ii

    if args.use_gpu:
        init_dl_program(args.gpu, seed=seed)

    scaler = StandardScaler()

    args.log_dir = (args.save_path + '/' +
                    args.data + '/' +
                    method_name + '/' +
                    args.exp_name + '/' +
                    str(args.pred_len) + '/' +
                    setting)

    args.checkpoints = (args.save_path + '/' +
                    args.data + '/' +
                    method_name + '/' +
                    args.exp_name + '/' +
                    str(args.pred_len) + '/')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = get_logger(
        args.log_dir, __name__, 'info.log')

    logger.info(args)

    exp = Exp(args, scaler)  # set experiments

    logger.info(
        '{},{}'.format('Total parameters ', str(sum(p.numel() for p in exp.model.parameters() if p.requires_grad))))

    start_time = time.time()
    if args.mode == 'detect':
        print('>>>>>>>detect : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.detect(setting, logger)

    elif args.mode == 'verify':
        exp.detect(setting, logger)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print('>>>>>>>verify : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.verify(setting, logger, args.detect_type)

    elif args.mode == 'finetune':

        exp.detect(setting, logger)

        print('>>>>>>>finetune : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.finetune(setting, logger)

    elif args.mode == 'test':
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        get_test_results(setting, logger)

    elif args.mode == 'train':
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting, logger)
        logger.info(("total train time: {}".format(time.time() - start_time)))

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        get_test_results(setting, logger)
