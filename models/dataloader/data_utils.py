from common.utils import set_seed


def dataset_builder(args):
    set_seed(args.seed)  # fix random seed for reproducibility

    if args.dataset == 'miniImageNet':
        from models.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'cub':
        from models.dataloader.cub import CUB as Dataset
    elif args.dataset == 'tieredImageNet':
        from models.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'CIFAR-FS':
        from models.dataloader.cifar_fs import DatasetLoader as Dataset
    else:
        raise ValueError('Unkown Dataset')
    return Dataset
