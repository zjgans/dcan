import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, set_seed, setup_run,restart_from_checkpoint
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.dataloader.aux_dataloader import get_aux_dataloader
from models.renet import DCANet

from test import test_main, evaluate
from utils import rotrate_concat,record_data

os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["CUDA_LAUNCH_BLOCKING"]='0'

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

        data_aux = data_aux[0].cuda()
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
        logits_global, logits_eq = model(data_aux)

        loss_aux = F.cross_entropy(logits_global, train_labels_aux)
        # loss_aux = supcriterion(logits_sup,train_labels_aux)

        proxy_labels = torch.zeros(args.trans * batch_size).cuda().long()
        for ii in range(args.trans):
            proxy_labels[ii * batch_size:(ii + 1) * batch_size] = ii
        loss_eq = F.cross_entropy(logits_eq, proxy_labels)

        l_re = fea_loss + dis_loss * args.w_d
        loss_aux = loss_aux + absolute_loss
        loss = args.lamb * (epi_loss) + loss_aux + loss_eq + args.w_p * l_re

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

    if not args.no_wandb:
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
        _, val_acc, val_ci = evaluate(epoch, model, val_loader, args, set='test')

        if not args.no_wandb:
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

                torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
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
    args = setup_run(arg_mode='train')

    model = train_main(args)
    test_acc, test_ci = test_main(model, args)

    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/confidence_interval': test_ci})
