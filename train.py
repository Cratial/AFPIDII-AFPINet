import os
import argparse
import collections
import torch
from logger import get_logger
import models.loss as module_loss
import models.metric as module_metric
from parse_config import ConfigParser
from utils import ensure_dir, prepare_device, get_by_path, msg_box, wandb_save_code
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(config):
    # do full cross validation in single thread
    verbosity = 2

    logger = get_logger('train', verbosity=verbosity)
    train_msg = msg_box("TRAIN")
    logger.debug(train_msg)

    # setup GPU device if available, move model into configured device
    device, device_ids = prepare_device(config['n_gpu'])

    # datasets
    train_datasets = dict()
    valid_datasets = dict()

    # train dataset
    keys = ['datasets', 'train']
    for name in get_by_path(config, keys):
        train_datasets[name] = config.init_obj([*keys, name], 'data_loaders')

    # valid dataset  <==  test dataset is used in this project
    valid_exist = False
    # with random splitting from the training dataset
    # keys = ['datasets', 'valid']
    # for name in get_by_path(config, keys):
    #     valid_exist = True
    #     valid_datasets[name] = config.init_obj([*keys, name], 'data_loaders')

    # data_loaders
    train_data_loaders = dict()
    valid_data_loaders = dict()
    # train dataloader
    keys = ['data_loaders', 'train']
    for name in get_by_path(config, keys):
        ### Concat dataset
        if get_by_path(config, keys)[name]['type'] == "MultiDatasetDataLoader":
            train_data_loaders[name] = config.init_obj([*keys, name], 'data_loaders', train_datasets)
        else:
            dataset = train_datasets[name]
            train_data_loaders[name] = config.init_obj([*keys, name], 'data_loaders', dataset)

        if not valid_exist:
            valid_data_loaders[name] = train_data_loaders[name].valid_loader

    # with random splitting from the training dataset
    # # valid dataloader
    # keys = ['data_loaders', 'valid']
    # for name in get_by_path(config, keys):
    #     dataset = valid_datasets[name]
    #     valid_data_loaders[name] = config.init_obj([*keys, name], 'data_loaders', dataset)

    # models
    models = dict()
    logger_model = get_logger('model', verbosity=1)
    for name in config['models']:
        model = config.init_obj(['models', name], 'models')
        logger_model.info(model)
        logger.info(model)
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        models[name] = model

    # optimizers
    optimizers = dict()
    for name in config['optimizers']:
        trainable_params = filter(lambda p: p.requires_grad, models[name].parameters())
        optimizers[name] = config.init_obj(['optimizers', name], torch.optim, trainable_params)

    # losses
    losses = dict()
    for name in config['losses']:
        kwargs = {}
        losses[name] = config.init_obj(['losses', name], module_loss, **kwargs)

    # metrics
    metrics_iter = [getattr(module_metric, met) for met in config['metrics']['per_iteration']]  # []
    metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['per_epoch']]  # func of accuracy computing

    # unchanged objects in each fold
    torch_args = {'datasets': {'train': train_datasets, 'valid': valid_datasets},
                  'losses': losses, 'metrics': {'iter': metrics_iter, 'epoch': metrics_epoch}}

    # learning rate schedulers
    lr_schedulers = dict()
    for name in config['lr_schedulers']:
        lr_schedulers[name] = config.init_obj(['lr_schedulers', name],
                                              torch.optim.lr_scheduler, optimizers[name])

    # update objects for each fold
    update_args = {'data_loaders': {'train': train_data_loaders, 'valid': valid_data_loaders},
                   'models': models, 'optimizers': optimizers, 'lr_schedulers': lr_schedulers}
    torch_args.update(update_args)

    # ====>>>>>>> train entrance <<<<<<<<========
    trainer = config.init_obj(['trainer'], 'trainers', torch_args, config.save_dir, config.resume, device)

    log_best = trainer.train()

    msg = msg_box("result")
    logger.info(f"{msg}\n{log_best}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="configs/wave_hcraft_spec_fusion.json", type=str)
    run_args.add_argument('-d', '--device', default=None, type=str)
    run_args.add_argument('-r', '--resume', default=None, type=str)
    # run_args.add_argument('-r', '--resume', default='saved/AFPID_FE2/1222_113504/model/model_best.pth', type=str)
    run_args.add_argument('--mode', default='train', type=str)
    # run_args.add_argument('--run_id', default=None, type=str)
    run_args.add_argument('--run_id', default='Fusion Model', type=str)
    run_args.add_argument('--log_name', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', "flags type target")
    options = [
        CustomArgs(['--fold_idx'], type=int, target="trainer;fold_idx"),
        # fold_idx > 0 means multiprocessing is enabled
        CustomArgs(['--num_workers'], type=int, target="data_loaders;train;data;kwargs;DataLoader_kwargs;num_workers"),
        CustomArgs(['--lr', '--learning_rate'], type=float, target="optimizers;model;args;lr"),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target="data_loaders;train;data;args;DataLoader_kwargs;batch_size"),
        CustomArgs(['--tp', '--transform_p'], type=float, target="datasets;train;data;kwargs;transform_p"),
        CustomArgs(['--epochs'], type=int, target=["trainer;kwargs;epochs", "trainer;kwargs;save_period"])
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, default=None, type=opt.type)

    cfg = ConfigParser.from_args(args, options)
    main(cfg)
