from __future__ import print_function

import argparse
import socket
import time
import os
import mkl
import seaborn as sns
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import DCANet
from can.can_heatmap import can_heatmap
from common.utils import compute_accuracy, load_model, setup_run, by
from test import evaluate
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["CUDA_LAUNCH_BLOCKING"]='0'
def parse_option():
    parser = argparse.ArgumentParser('heatmap')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'cub', 'tieredImageNet', 'CIFAR-FS'])
    parser.add_argument('-data_root', type=str, default='/data/lxj/odata/dataset', help='dir of datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/miniImageNet/1shot-5way/test/max_acc.pth')

    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=64, help='auxiliary batch size')
    parser.add_argument('-temperature', type=float, default=0.2, metavar='tau',
                        help='temperature for metric-based loss')
    parser.add_argument('-lamb', type=float, default=0.25, metavar='lambda', help='loss balancing term')

    ''' about training schedules '''
    parser.add_argument('-max_epoch', type=int, default=80, help='max epoch to run')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')

    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('--use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='epoch_35.pth')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1, metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=30, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=1000, help='number of testing episodes after training')
    parser.add_argument('-proto_size', type=int, default=20, help='the number of dynamic prototype')
    parser.add_argument('--meta_class', type=int, default=5, help='class of 5-way setting')
    parser.add_argument('--crop_size', type=int, default=84)
    parser.add_argument('--trans', type=int, default=4, help='number of transformations')
    parser.add_argument('--w_d', type=float, default=0.01, help='weight of distance loss')
    parser.add_argument('--w_p', type=float, default=1)
    parser.add_argument('--hidden_dim', type=int, default=320, help='hidden size for cross attention layer')
    parser.add_argument('-temperature_attn', type=float, default=5, metavar='gamma',
                        help='temperature for softmax in computing cross-attention')

    ''' about env '''
    parser.add_argument('-gpu', default='0', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-test_tag', type=str, default='test',
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

def main():

    args = parse_option()
    model = DCANet(args).cuda()
    # model = load_model(model, args.model_path)
    ckpt = torch.load(args.model_path)["params"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    # valset = Dataset('val', args)
    # val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    # val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    # val_loader = [x for x in val_loader]
    test_loader = [x for x in test_loader]
    pre_num = args.way * (1 + args.query)
    title = '5_epoch'

    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    # all_features,all_ys = get_feature_and_label(model,test_loader,args)
    #
    # for batch in range(args.test_episode):
    #     sub_feature, sub_ys = all_features[batch*pre_num:(batch+1)*pre_num],all_ys[batch*pre_num:(batch+1)*pre_num]
    #
    #     save_path = './visualization/{}'.format(title)
    #
    #     tsne_plot(sub_feature,sub_ys,save_path,batch)

def get_feature_and_label(net,test_loader,args):

    net= net.eval()
    all_features,all_ys = [],[]

    # data = test_loader[13]
    pre_num = args.way * (args.shot + args.query)
    pre_query = args.way * args.query
    pre_spt = args.way * args.shot
    length = pre_query * args.way
    k = args.way * args.shot

    query_label = torch.arange(args.way).repeat(args.query)
    support_label = torch.arange(args.way)
    query_index = torch.arange(0, length, step=5)
    query_index = torch.LongTensor(query_label + query_index)
    assert len(query_index) == len(query_label)

    with torch.no_grad():
        for batch_id, (images, label) in enumerate(test_loader):
        #     images,label = data
            images = images.cuda()
            net.mode = 'encoder'
            base_feat = net(images)

            net.mode='cca'
            data_shot, data_query = base_feat[:k], base_feat[k:]
            spt_feat, qry_feat = net((data_shot.unsqueeze(0).repeat(1, 1, 1, 1, 1), data_query))
            spt_feat = spt_feat.mean(dim=0)
            qry_feat = qry_feat.mean(dim=1)
            support_features = spt_feat.detach().cpu().numpy()
            query_features = qry_feat.detach().cpu().numpy()

            support_ys = support_label.view(-1).numpy()
            query_ys =query_label.view(-1).numpy()

            features = np.concatenate([support_features,query_features])
            labels = np.concatenate([support_ys,query_ys])
            all_features.append(features)

            all_ys.append(labels)

    all_features = np.concatenate(all_features)
    all_ys = np.concatenate(all_ys)

    return all_features,all_ys


def tsne_plot(all_features,all_ys,save_path,batch):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    colors = ['g','orangered','slategray','darkgoldenrod','darkcyan']     # lightseagreen orangered
    name = os.path.join(save_path,f'{batch}.jpg'.format(batch))
    all_transformed = TSNE(n_jobs=20,metric='cosine',square_distances='legacy').fit_transform(all_features)
    num_features = len(all_features)
    plt.figure()
    plt.xticks()
    plt.yticks()
    # Accent viridis_r
    camp = plt.get_cmap('viridis_r')

    # plt.scatter(all_transformed[:5,0],all_transformed[:5,1],c=colors,marker='s',cmap=camp,s=50)

    plt.scatter(all_transformed[5:num_features,0],all_transformed[5:num_features,1],
                c=np.array(colors)[all_ys[5:num_features].astype(int)],
                marker='*',cmap=camp,s=5)  #cmap="tab10"
    plt.scatter(all_transformed[:5, 0], all_transformed[:5, 1], c=colors, marker='s', cmap=camp, s=50)

    # plt.title(title,y=-2)
    plt.savefig(name,dpi=1500)
    plt.show()

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

if __name__ == '__main__':
    main()

