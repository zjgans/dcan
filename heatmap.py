from __future__ import print_function

import argparse
import cv2
import os
import mkl

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
import numpy as np
import tqdm
import numpy
import matplotlib.pyplot as plt
from common.meter import Meter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.dcan import DCANet
from common.utils import compute_accuracy, load_model,  by

mkl.set_num_threads(2)
def parse_option():
    parser = argparse.ArgumentParser('heatmap')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'cub', 'tieredImageNet', 'CIFAR-FS'])
    parser.add_argument('-data_root', type=str, default='/data/zhouj/odata/dataset', help='dir of datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--model_path',type=str,default='./checkpoints/miniImageNet/1shot-5way/ddf_eq_64_re0.5_nonlocal_0.25_C_320/max_acc.pth')

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
    parser.add_argument('-test_episode', type=int, default=30, help='number of testing episodes after training')
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
    parser.add_argument('-gpu', default='3', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-test_tag', type=str, default='ddf_eq_64_re0.5_nonlocal_0.25_C_320',
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

class Preprocessor(Dataset):
    def __init__(self, dataset, transforms=None):
        super(Preprocessor, self).__init__()
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label, _ = self.dataset[item]

        img = self.transforms(img)

        return img, label

def renet_heatmap():
    args = parse_option()
    model = DCANet(args).cuda()
    ckpt = torch.load(args.model_path)["params"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict,strict=False)
    activate_fun = nn.Softmax(dim=1)
    if torch.cuda.is_available():
        model = model.cuda()
        activate_fun = activate_fun.cuda()
        cudnn.benchmark = True


    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    val_loader = [x for x in val_loader]
    test_loader = [x for x in test_loader]

    # _, test_acc, test_ci = evaluate("best", model, val_loader, args, set='test')
    # print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    pre_num = args.way * (args.shot + args.query)
    pre_query = args.way * args.query
    pre_spt = args.way * args.shot
    length = pre_query*args.way
    k = args.way * args.shot

    query_label = torch.arange(args.way).repeat(args.query)
    query_index = torch.arange(0,length,step=5)
    query_index = torch.LongTensor(query_label+query_index).cuda()
    assert len(query_index)==len(query_label)

    model.eval()
    for batch_id, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.cuda()

        def attention_forward(encoder, imgs):
            # hard-coded forward because we need the feature-map and not the finalized feature
            encoder.mode = 'encoder'
            scr_feat, base_feat = model(imgs)
            # feats = encoder.avgpool2(x)
            scr_feat_batch = scr_feat.permute((0, 2, 3, 1)).contiguous().view((-1, scr_feat.shape[1]))
            # reminder: "fc" layer outputs: (feature, class logits)
            # scr_feat_batch = encoder.classifier(scr_feat_batch)
            scr_feat_batch = activate_fun(scr_feat_batch)
            # scr_feat_batch = F.normalize(scr_feat_batch,dim=1)

            scr_feat_batch = scr_feat_batch.view(
                (scr_feat.shape[0], scr_feat.shape[2], scr_feat.shape[3], scr_feat_batch.shape[1]))
            scr_feat_batch = scr_feat_batch.permute((0, 3, 1, 2))
            print(scr_feat_batch.size())

            # base heatmap
            base_feat_batch = base_feat.permute((0, 2, 3, 1)).contiguous().view((-1, base_feat.shape[1]))
            # reminder: "fc" layer outputs: (feature, class logits)
            # scr_feat_batch = encoder.classifier(scr_feat_batch)
            base_feat_batch = activate_fun(base_feat_batch)
            # scr_feat_batch = F.normalize(scr_feat_batch,dim=1)

            base_feat_batch = base_feat_batch.view(
                (base_feat.shape[0], base_feat.shape[2], base_feat.shape[3], base_feat_batch.shape[1]))
            base_feat_batch = base_feat_batch.permute((0, 3, 1, 2))

            encoder.mode= 'cca'
            data_shot, data_query = scr_feat[:k], scr_feat[k:]
            spt_feat,qry_feat = model((data_shot.unsqueeze(0).repeat(1,1,1,1,1),data_query))
            spt_feat = spt_feat.mean(dim=0)
            spt_feat = spt_feat.unsqueeze(0).repeat(args.shot,1,1,1,1)
            spt_feat = spt_feat.contiguous().view(-1,spt_feat.shape[2],spt_feat.shape[3],spt_feat.shape[4])
            # qry_feat = qry_feat.mean(dim=1)
            qry_feat = qry_feat.contiguous().view(-1,qry_feat.shape[2],qry_feat.shape[3],qry_feat.shape[4])
            qry_feat = qry_feat[query_index]

            spt_feat_batch = spt_feat.permute((0,2,3,1)).contiguous().view((-1,spt_feat.shape[1]))
            qry_feat_batch = qry_feat.permute((0,2,3,1)).contiguous().view((-1,qry_feat.shape[1]))

            spt_feat_batch = activate_fun(spt_feat_batch)
            qry_feat_batch = activate_fun(qry_feat_batch)

            spt_feat_batch = spt_feat_batch.view((spt_feat.shape[0],spt_feat.shape[2],spt_feat.shape[3],spt_feat_batch.shape[1]))
            qry_feat_batch = qry_feat_batch.view((qry_feat.shape[0],qry_feat.shape[2],qry_feat.shape[3],qry_feat_batch.shape[1]))

            spt_feat_batch = spt_feat_batch.permute((0, 3, 1, 2))
            qry_feat_batch = qry_feat_batch.permute((0, 3, 1, 2))

            return base_feat_batch,scr_feat_batch, spt_feat_batch,qry_feat_batch

        base_f,f_q,spt_feat,qry_feat = attention_forward(model, images)
        spt_imgs = images[:pre_spt]
        qry_imgs = images[pre_spt:]
        heatmap(images,base_f, pre_num, batch_id, img_size=84, split='base')
        heatmap(images, f_q, pre_num, batch_id, img_size=84, split='scr')
        heatmap_final(spt_imgs, spt_feat, pre_spt, batch_id, img_size=84, split='support')
        heatmap_final(qry_imgs, qry_feat, pre_query, batch_id, img_size=84, split='query')
     

def heatmap(im_q, f_q, batch_size,batch_id, img_size,split):
    os.makedirs('./imgs/test/batch_{}/_{}/'.format(batch_id,split), exist_ok=True)

    for idd in range(batch_size):
        aa = torch.norm(f_q, dim=1)

        imgg = im_q[idd] * torch.Tensor([[[0.229, 0.224, 0.225]]]).view(
            (1, 3, 1, 1)).cuda() + torch.Tensor(
            [[[0.485, 0.456, 0.406]]]).view((1, 3, 1, 1)).cuda()
        imgg = imgg.squeeze(0).cpu().numpy()
        maxValue = imgg.max()
        imgg = imgg * 255 / maxValue

        # imgg = imgg.squeeze(0).cpu().numpy()
        imgg = np.uint8(imgg)
        imgg = imgg.transpose(1,2,0)
        origin = cv2.cvtColor(imgg,cv2.COLOR_RGB2BGR)
        origin = cv2.resize(origin,(224,224),interpolation=cv2.INTER_AREA)

        heatmap = F.interpolate(((aa[idd]-aa[idd].min() )/ (aa[idd].max()-aa[idd].min())).detach().unsqueeze(0).unsqueeze(0).repeat(( 1,3, 1, 1)),
                                [img_size, img_size],mode='bilinear').squeeze(0).cpu().numpy()

        # thresh = 0.1
        # heatmap[heatmap<thresh] = 0
        max_heatmap = heatmap.max()
        heatmap = heatmap * 255 / max_heatmap
        heatmap = np.uint8(heatmap)
        # heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
        heatmap = heatmap.transpose(1,2,0)
        # heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatmap = cv2.cvtColor(heatmap,cv2.COLOR_RGB2BGR)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # heatmap_color = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
        # # print('heatmap',heatmap.shape)

        overlay = origin.copy()
        heatmap_img = heatmap_color.copy()
        beta = 0.1
        alpha = 0.5

        overlay = cv2.rectangle(overlay,(0,0),(origin.shape[1],origin.shape[0]),(255,0,0),-1)
        frame = cv2.addWeighted(overlay,beta,origin,1-beta,0)
        img_add = cv2.addWeighted(heatmap_img,alpha,frame,1-alpha,0)

        cv2.imwrite('./imgs/test/batch_{}/_{}/heat_{}_add0.5.jpg'.format(batch_id,split,idd),img_add)
        cv2.imwrite('./imgs/test/batch_{}/_{}/oring_{}.png'.format(batch_id,split,idd),origin)


def heatmap_final(im_q, f_q, batch_size,batch_id, img_size,split='support'):
    os.makedirs('./imgs/test/batch_{}/_{}'.format(batch_id,split), exist_ok=True)

    for idd in range(batch_size):
        aa = torch.norm(f_q, dim=1)

        imgg = im_q[idd] * torch.Tensor([[[0.229, 0.224, 0.225]]]).view(
            (1, 3, 1, 1)).cuda() + torch.Tensor(
            [[[0.485, 0.456, 0.406]]]).view((1, 3, 1, 1)).cuda()
        imgg = imgg.squeeze(0).cpu().numpy()
        maxValue = imgg.max()
        imgg = imgg * 255 / maxValue
        imgg = np.uint8(imgg)
        imgg = imgg.transpose(1,2,0)
        origin = cv2.cvtColor(imgg,cv2.COLOR_RGB2BGR)
        origin = cv2.resize(origin, (224, 224), interpolation=cv2.INTER_AREA)

        heatmap = F.interpolate(((aa[idd]-aa[idd].min() )/ (aa[idd].max()-aa[idd].min())).detach().unsqueeze(0).unsqueeze(0).repeat(( 1,3, 1, 1)),
                                [img_size, img_size],mode='bilinear').squeeze(0).cpu().numpy()

        max_heatmap = heatmap.max()
        heatmap = heatmap * 255 / max_heatmap
        heatmap = np.uint8(heatmap)
        heatmap = heatmap.transpose(1,2,0)
        # heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatmap = cv2.cvtColor(heatmap,cv2.COLOR_RGB2BGR)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = origin.copy()
        heatmap_img = heatmap_color.copy()
        beta = 0.1
        alpha = 0.5

        overlay = cv2.rectangle(overlay,(0,0),(origin.shape[1],origin.shape[0]),(255,0,0),-1)
        frame = cv2.addWeighted(overlay,beta,origin,1-beta,0)
        img_add = cv2.addWeighted(heatmap_img,alpha,frame,1-alpha,0)

        cv2.imwrite('./imgs/test/batch_{}/_{}/heat_{}_add0.jpg'.format(batch_id,split,idd),img_add)
        # cv2.imwrite('./imgs/oring_{}.png'.format(idd),origin)


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
            model.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.mode = 'cca'

            logits = model(data)
            loss = F.cross_entropy(logits,query_label)
            acc = compute_accuracy(logits, query_label)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


if __name__ == '__main__':
    renet_heatmap()

