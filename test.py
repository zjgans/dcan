import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, by,set_gpu,set_seed
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.dcan import DCANet

def parse_option():
    parser = argparse.ArgumentParser('test')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'cub', 'tieredImageNet', 'CIFAR-FS'])
    parser.add_argument('-data_root', type=str, default='/data/~/dataset', help='dir of datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--model_path',type=str,default='./checkpoints/miniImageNet/1shot-5way/~/max_acc.pth')


    parser.add_argument('-temperature', type=float, default=0.2, metavar='tau',
                        help='temperature for metric-based loss')

    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('--use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='epoch_35.pth')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1, metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=200, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')
    parser.add_argument('-proto_size', type=int, default=100, help='the number of dynamic prototype')
    parser.add_argument('--meta_class', type=int, default=5, help='class of 5-way setting')
    parser.add_argument('--crop_size', type=int, default=84)

    parser.add_argument('--hidden_dim', type=int, default=320, help='hidden size for cross attention layer')
    parser.add_argument('-temperature_attn', type=float, default=5, metavar='gamma',
                        help='temperature for softmax in computing cross-attention')

    ''' about env '''
    parser.add_argument('-gpu', default='3', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-test_tag', type=str, default='1',
                        help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    if args.dataset == 'miniImageNet':
        args.num_class = 64
    elif args.dataset == 'cub':
        args.num_class = 100
    elif args.dataset == 'fc100':
        args.num_class = 60
    elif args.dataset == 'tieredImageNet':
        args.num_class = 351
    elif args.dataset == 'CIFAR-FS':
        args.num_class = 64
        args.crop_size = 42
    elif args.dataset == 'cars':
        args.num_class = 130
    elif args.dataset == 'dogs':
        args.num_class = 70

    return args

def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()
    test_accuracies = []

    query_label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...
    support_label = torch.arange(args.way).repeat(args.shot).cuda()

    k = args.way * args.shot
    num_test_examples = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'coda'
            logits = model((data_shot.unsqueeze(0).repeat(1, 1, 1, 1, 1), data_query))
            loss = F.cross_entropy(logits,query_label)
            acc = compute_accuracy(logits, query_label)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(model, args):

    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))
    # model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci

if __name__ == '__main__':

    args = parse_option()
    ''' define model '''
    set_seed(args.seed)
    model = DCANet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args)
