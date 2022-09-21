import os
import torch
import pprint
import random
import numpy as np
from termcolor import colored


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


def compute_accuracy(logits, labels):

    pred = torch.argmax(logits, dim=1)

    return (torch.sum((pred == labels).type(torch.float))/labels.size(0)).item() * 100.


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)

def load_model(model, dir):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(dir)['model']

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:
        print('loading model from :', dir)
        if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
            if 'module' in list(pretrained_dict.keys())[0]:
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
        model.load_state_dict(model_dict)
    return model

def restart_from_checkpoint(ckpt_path,run_variables=None,**kwargs):
    if not os.path.isfile(ckpt_path):
        return
    print('Found checkpoint at {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    for key,value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key],strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            print('=>loaded {} from checkpoint {}'.format(key,ckpt_path))
        else:
            print(
                "=> failed to load {} from checkpoint '{}'".format(key, ckpt_path)
            )
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
                print('{}:{})'.format(var_name,run_variables[var_name]))

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def detect_grad_nan(model):
    for param in model.parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()

def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow
