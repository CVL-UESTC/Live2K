import argparse
import datetime
import logging
import math
import random
import time
import os
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ.setdefault("PYTHONHASHSEED", "0")
# os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torchinfo import summary

import numpy as np
from os import path as osp
from data import create_dataloader, create_dataset
from data.data_sampler import EnlargedSampler
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models import create_model
from utils import (MessageLogger, check_resume,
                         get_root_logger, get_time_str, init_tb_logger,
                         init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                         set_random_seed)
from utils.dist_util import get_dist_info, init_dist
from utils.options import dict2str, parse
torch.set_num_threads(10)
torch.autograd.set_detect_anomaly(True)

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, default='options/train/LPTN/train_FiveK.yml', help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='codes', log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['path']['root'], 'tb_logger', opt['name']))
    return logger, tb_logger

def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['path']['root'], 'tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    start_time1 = time.time()
    result = create_train_val_dataloader(opt, logger)
    elapsed_time = time.time() - start_time1
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result
    logger.info(f"Creating train and validation dataloaders took {elapsed_time:.2f} seconds.")

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")
    device = torch.device("cuda")  # 获取模型所在的设备
    # lq = torch.randn(1, 15, 1296, 1728).to(device)
    # lq_up = torch.randn(1, 15, 3072, 4096).to(device)
    # ref = torch.randn(1, 3, 3072, 4096).to(device)
    # ref_full = torch.randn(1, 3, 3072, 4096).to(device)
    # summary(model.net_g, input_data=(lq, lq_up, ref, ref_full), col_names=["input_size", "output_size", "num_params", "mult_adds"])
    # training

    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    
    
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)     #在每个 epoch 开始时，设置采样器的 epoch。这通常用于分布式训练，确保每个 epoch 的数据顺序不同，以避免每次训练都使用相同的数据顺序，从而提高模型的泛化能力。
        prefetcher.reset()                 #重置数据预取器，准备开始新的 epoch 的数据预取。
        train_data = prefetcher.next()     #预取器并不会一次性将所有数据加载到 CUDA 中，而是每次获取一个batch size的样本。

        #预取器工作原理：
        #在训练过程中，预取器（prefetcher）会在GPU进行前向传播和反向传播计算时，提前将下一批次的数据从磁盘加载到内存中，甚至直接加载到GPU内存中。
        #这样，当GPU完成当前批次的计算后，下一批次的数据已经准备好，可以立即进行计算，而不需要等待数据加载。
        #prefetcher.next()返回当前批次的数据，同时在后台加载下一批次的数据。

  
        while train_data is not None:      #进入批次数据的训练循环
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break

            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time

            # forwa = model.forwa

            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                # log_vars.update({'fuse_inp': time0,'LLFormer': time1, 'guidenet': time2, 'slicing': time3, 'apply': time4, 'total': all_time, 'forward': forward_time, 'backward': backward_time, 'forwa': forwa})
                
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'])
            
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
