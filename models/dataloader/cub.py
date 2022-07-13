import os
import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CUB(Dataset):

    def __init__(self, split,args):
        self.dataset = args.dataset
        self.split = split
        self.data_root = os.path.join(args.data_root,args.dataset)

        IMAGE_PATH = os.path.join(self.data_root, 'images')
        SPLIT_PATH = os.path.join(self.data_root, 'split/')
        txt_path = osp.join(SPLIT_PATH, split + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        # IMAGE_PATH = os.path.join(args.data_dir, 'cub/')
        # SPLIT_PATH = os.path.join(args.data_dir, 'cub/split/')
        # txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        # lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []
        self.args = args
        if split == 'train':
            lines.pop(5864)  #this image file is broken

        # for l in lines:
        #     context = l.split(',')
        #     name = context[0]
        #     wnid = context[1]
        #     path = osp.join(IMAGE_PATH, name)
        #     if wnid not in self.wnids:
        #         self.wnids.append(wnid)
        #         lb += 1
        #
        #     data.append(path)
        #     label.append(lb)

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(IMAGE_PATH, wnid + '/' + name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = np.unique(np.array(label)).shape[0]
        # self.return_path = return_path
        img_label = []
        for id, (img, label) in enumerate(zip(self.data, self.label)):
            img_label.append((img, label))
        self.img_label = img_label

        if split == 'train':

            image_size = 84

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        # if self.return_path:
        #     return image, label, path

        return image, label

if __name__ == '__main__':
    pass
