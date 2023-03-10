from __future__ import print_function

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GeomTransform:

    def __init__(self, normalize, opt):

        self.normalize = normalize
        self.color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        self.opt = opt
        self.size = opt.crop_size

    def transform_sample(self, img, indx=None):
     
        out = self.color_transform(img)
        # out = transforms.RandomGrayscale(p=0.2)(out)
        out = transforms.RandomHorizontalFlip()(out)
        out = transforms.functional.to_tensor(out)
        out = self.normalize(out)
        return out

    def __call__(self, x):
        # img = np.asarray(x).astype('uint8')
        if self.opt.dataset == 'CIFAR-FS' or self.opt.dataset == 'FC100':
            img = transforms.RandomResizedCrop(84)(Image.fromarray(x))
            img = self.transform_sample(img)

        if self.opt.dataset == 'miniImageNet' or self.opt.dataset == 'tieredImageNet':
            img = transforms.RandomCrop(84, padding=8)(Image.fromarray(x)
            img = self.transform_sample(img)
      
        if self.opt.dataset == 'cub':
            img = transforms.RandomResizedCrop(84)(Image.open(x).convert('RGB'))
            img = self.transform_sample(img)

        return img


class Preprocessor(Dataset):
    def __init__(self, dataset, geo_transforms=None):
        super(Preprocessor, self).__init__()
        self.geo_transforms = geo_transforms

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]

        img = self.geo_transforms(img)

        return img, label

def get_aux_dataloader(opt, dataset):
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean=mean, std=std)

        train_loader = DataLoader(Preprocessor(dataset, GeomTransform(normalize, opt)),
                                  batch_size=opt.batch, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers, pin_memory=True)

    if opt.dataset == 'miniImageNet':
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)

        train_loader = DataLoader(Preprocessor(dataset, GeomTransform(normalize, opt)),
                                  batch_size=opt.batch, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers,
                                  )

    if opt.dataset == 'tieredImageNet':
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)

        train_loader = DataLoader(Preprocessor(dataset,GeomTransform(normalize, opt)),
                                  batch_size=opt.batch, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers, pin_memory=True
                                  )
    if opt.dataset == 'cub':
        normalize = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))

        train_loader = DataLoader(Preprocessor(dataset, GeomTransform(normalize, opt)),
                                  batch_size=opt.batch, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers, pin_memory=True
                                  )

    return train_loader

