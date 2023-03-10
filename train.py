import os
import tqdm
import time
import wandb
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, set_seed, restart_from_checkpoint
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.dataloader.aux_dataloader import get_aux_dataloader
from models.renet import DCANet

from test import test_main, evaluate
from utils import rotrate_concat,record_data
from common.utils import pprint,ensure_path,set_gpu
os.environ["CUDA_VISIBLE_DEVICES"]='2'
os.environ["CUDA_LAUNCH_BLOCKING"]='2'

def parse_args():
    parser = argparse.ArgumentParser(description='train')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'cub', 'tieredImageNet', 'CIFAR-FS','FC100'])
    parser.add_argument('-data_root', type=str, default='/home/lxj/new_main/dataset', help='dir of datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=64, help='auxiliary batch size')
    parser.add_argument('-temperature', type=float, default=0.2, metavar='tau', help='temperature for metric-based loss')
    parser.add_argument('-lamb', type=float, default=0.25, metavar='lambda', help='loss balancing term')
    parser.add_argument('--w_d', type=float, default=0.01, help='weight of distance loss')
    parser.add_argument('--w_p', type=float, default=0.5)

    ''' about training schedules '''
    parser.add_argument('-max_epoch', type=int, default=80, help='max epoch to run')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70], help='milestones for MultiStepLR')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('-use_resume',action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='epoch_10.pth')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1
                        , metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=200, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')
    parser.add_argument('-proto_size',type=int,default=100,help='the number of dynamic prototype')
    parser.add_argument('--crop_size', type=int, default=84)
    parser.add_argument('--trans', type=int, default=4, help='number of transformations')
    parser.add_argument('--hidden_size', type=int, default=320, help='hidden size for cross attention layer')
    parser.add_argument('--feat_dim', type=int, default=640)
    parser.add_argument('--sup_t',type=float,default=0.2)

    ''' about CoDA '''
    parser.add_argument('-temperature_attn', type=float, default=2, metavar='gamma', help='temperature for softmax in computing cross-attention')

    ''' about env '''
    parser.add_argument('-gpu', default='2', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-test_tag', type=str, default='test_wp0.5_wd0.01_lam0.25_t2', help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-wandb', action='store_true', help='not plotting learning curve on wandb',
                        )  # train: enable logging / test: disable logging
    args = parser.parse_args()
    pprint(vars(args))

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    args.save_path = os.path.join(f'checkpoints/{args.dataset}/{args.shot}shot-{args.way}way/', args.test_tag)
    ensure_path(args.save_path)
    if not args.wandb:
        wandb.init(project=f'renet-{args.dataset}-{args.way}w{args.shot}s',
                   config=args,
                   save_code=True,
                   name=args.test_tag)

    if args.dataset == 'miniImageNet':
        args.num_class = 64
    elif args.dataset == 'cub':
        args.num_class = 100
    elif args.dataset == 'cub':
        args.num_class = 64
    elif args.dataset == 'FC100':
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

def train(epoch, model, loader,optimizer, args=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    query_label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):

        data, train_labels = data.cuda(), train_labels.cuda()
        # data_aux, train_labels_aux = data_aux.cuda(),train_labels_aux.cuda()

        data_aux = data_aux.cuda()
        batch_size = data_aux.size(0)
        data_aux = rotrate_concat([data_aux])
        train_labels_aux = train_labels_aux.repeat(args.trans).cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        # model.mode = 'encoder'
        data, fea_loss, cst_loss, dis_loss = model(data)
        # data = model(data)
        data_aux = model(data_aux, aux=True)  # I prefer to separate feed-forwarding data and data_aux due to BN

        # loss for batch
        model.module.mode = 'coda'
        data_shot, data_query = data[:k], data[k:]
        logits, absolute_logits = model((data_shot.unsqueeze(0).repeat(1, 1, 1, 1, 1), data_query))

        # epi_loss = criterion(logits,query_label)
        epi_loss = F.cross_entropy(logits, query_label)
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_global,logits_eq = model(data_aux)
        loss_aux = F.cross_entropy(logits_global, train_labels_aux)

        ##=============================#
        proxy_labels = torch.zeros(args.trans * batch_size).cuda().long()
        for ii in range(args.trans):
            proxy_labels[ii * batch_size:(ii + 1) * batch_size] = ii
        loss_eq = F.cross_entropy(logits_eq, proxy_labels)
        ## =============================#

        l_re = fea_loss + dis_loss * args.w_d
        loss_aux = absolute_loss + loss_aux
        loss = args.lamb * (epi_loss)  + loss_aux + loss_eq + l_re

        # mean_logits = logits.flatten(start_dim=2).sum(dim=-1)
        acc = compute_accuracy(logits, query_label)
        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        # detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()

def train_main(args):
    Dataset = dataset_builder(args)

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = get_aux_dataloader(args,trainset_aux.img_label)
    # train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=4, pin_memory=True)

    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]
    test_loader = [x for x in test_loader]

    set_seed(args.seed)
    model = DCANet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    if args.wandb:
        wandb.watch(model)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    to_restore = {'epoch': 0}
    if args.use_resume:
        print('------load the parameters from  checkpoint--------')
        restart_from_checkpoint(
            os.path.join(args.save_path, args.resume_file),
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
        )
    start_epoch = to_restore['epoch']

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    for epoch in range(start_epoch, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, train_ci = train(epoch, model, train_loaders,optimizer, args)
        # _, val_acc, val_ci = evaluate(epoch, model, val_loader, args, set='test')

        if args.wandb:
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc}, step=epoch)
       ##################################################################
        if epoch % args.save_freq == 0 or epoch == args.max_epoch:

            test_loss, test_acc, _ = evaluate(epoch, model, test_loader, args, set='test')
            wandb.log({'test/loss': test_loss, 'test/acc': test_acc}, step=epoch)

            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }
            save_file = os.path.join(args.save_path, 'epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            if test_acc > max_acc:
                print(f'[ log ] *********A better model is found ({test_acc:.3f}) *********')
                max_acc, max_epoch = test_acc, epoch

                torch.save(dict(model=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
                torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))
        ################################################################################
            # torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            # torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        # if args.save_all:
        #     torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
        #     torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))
        # record_data(epoch,train_acc.cpu().numpy(),train_ci.cpu().numpy(),val_acc.cpu().numpy(),val_ci.cpu().numpy(),save_path='./record_data/gap.csv')
        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        lr_scheduler.step()
    return model

if __name__ == '__main__':

    args = parse_args()
    model = train_main(args)
    test_acc, test_ci = test_main(model, args)

    if not args.wandb:
        wandb.log({'test/acc': test_acc, 'test/confidence_interval': test_ci})
