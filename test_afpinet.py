import os
import argparse
import collections
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from logger import get_logger
import models.loss as module_loss
import models.metric as module_metric
from models.metric import MetricTracker
from parse_config import ConfigParser
from utils import ensure_dir, prepare_device, get_by_path, msg_box
from sklearn.metrics import log_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

# fix random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    logger = get_logger('test')
    test_msg = msg_box("TEST")
    logger.debug(test_msg)

    # datasets
    test_datasets = dict()
    keys = ['datasets', 'test']
    for name in get_by_path(config, keys):
        test_datasets[name] = config.init_obj([*keys, name], 'data_loaders')

    # data_loaders
    test_data_loaders = dict()
    keys = ['data_loaders', 'test']
    for name in get_by_path(config, keys):
        dataset = test_datasets[name]
        test_data_loaders[name] = config.init_obj([*keys, name], 'data_loaders', dataset)

    # prepare model for testing
    device, device_ids = prepare_device(config['n_gpu'])

    # models
    resume = config.resume
    logger.info(f"Loading model: {resume} ...")
    checkpoint = torch.load(resume)
    models = dict()
    logger_model = get_logger('model', verbosity=0)
    for name in config['models']:
        model = config.init_obj(['models', name], 'models')
        logger_model.info(model)
        state_dict = checkpoint['models'][name]
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models[name] = model

    # losses
    kwargs = {}
    # TODO
    if config['losses']['loss'].get('balanced', False):  # false
        target = test_datasets['data'].y_test
        weight = compute_class_weight(class_weight='balanced',
                                      classes=target.unique(),
                                      y=target)
        weight = torch.FloatTensor(weight).to(device)
        kwargs.update(pos_weight=weight[1])
    # CrossEntropyLoss for multiple class classification
    loss_fn = config.init_obj(['losses', 'loss'], module_loss, **kwargs)

    # metrics
    metrics_iter = [getattr(module_metric, met) for met in config['metrics']['per_iteration']]
    metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['per_epoch']]
    keys_loss = ['loss']
    keys_iter = [m.__name__ for m in metrics_iter]
    keys_epoch = [m.__name__ for m in metrics_epoch]
    test_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch)

    with torch.no_grad():
        print("testing...")
        model = models['model']
        testloader = test_data_loaders['data']
        if len(metrics_epoch) > 0:  # true
            outputs = torch.FloatTensor().to(device)
            targets = torch.FloatTensor().to(device)
        for batch_idx, (data, target) in tqdm(enumerate(testloader), total=len(testloader)):
            if isinstance(data, dict):
                data = {k: v.to(device) for k, v in data.items()}
            else:
                data = data.to(device)

            target = target.to(device)
            output = model(data)
            if len(metrics_epoch) > 0:  # true
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            test_metrics.iter_update('loss', loss.item())
            for met in metrics_iter:
                test_metrics.iter_update(met.__name__, met(output, target))

        for met in metrics_epoch:
            test_metrics.epoch_update(met.__name__, met(outputs, targets))

    print(outputs.cpu().numpy().shape)

    outputs_log = nn.LogSoftmax(dim=1)(outputs)
    outputs = nn.Softmax(dim=1)(outputs)

    preds = torch.argmax(outputs, dim=1)

    cm = confusion_matrix(targets.cpu().numpy(), preds.cpu().numpy())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm.diagonal()
    values = [np.round(i,decimals=4) for i in list(cm.diagonal())]
    print(values)

    # label_list = config['datasets']['test']['data']['kwargs']['label_list']
    # print(label_list)

    # saving prediction and targets
    save_dir = config.save_dir["metric"]
    np.savez(os.path.join(save_dir, 'pred_target'), pred=preds.cpu().numpy(), target=targets.cpu().numpy())

    print("outputs shape", outputs.shape)
    print("sklearn log loss: ", log_loss(targets.cpu().numpy(), outputs.cpu().numpy()))
    print("pytorch log loss: ", torch.mean(nn.NLLLoss(reduction='none')(outputs_log.cpu(), targets.cpu().long())))

    test_log = test_metrics.result()
    logger.info(test_log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="configs/wave_hcraft_spec_fusion.json", type=str)
    # trained model pth
    run_args.add_argument('-r', '--resume', default='saved/acoustic_footstep_ide/Fusion Model/model/model_best.pth', type=str)
    run_args.add_argument('-d', '--device', default=None, type=str)
    run_args.add_argument('--mode', default='test', type=str)
    # run_args.add_argument('--run_id', default=None, type=str)
    run_args.add_argument('--run_id', default='Fusion Model', type=str)
    run_args.add_argument('--log_name', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', "flags default type target")
    options = [
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, default=opt.default, type=opt.type)

    # additional arguments for testing
    test_args = args.add_argument_group('test_args')
    test_args.add_argument('--output_path', default=None, type=str)

    cfg = ConfigParser.from_args(args, options)
    main(cfg)
