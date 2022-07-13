import os
import os.path as osp

import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class tieredImageNet(Dataset):

    def __init__(self, split, args):
        self.dataset = args.dataset
        self.split = split
        self.root_path = args.data_root
        print("Current train dataset: ", self.dataset, self.split)
        if self.root_path is None:
            self.root_path = "/home/lxj/new_main/dataset"

        self.image_file_pattern = '%s_images.npz'
        self.label_file_pattern = '%s_labels.pkl'

        image_file = os.path.join(self.root_path, self.dataset,self.image_file_pattern % split)
        self.data = np.load(image_file)['images']
        label_file = os.path.join(self.root_path,self.dataset, self.label_file_pattern % split)
        self.origin_labels = self._load_labels(label_file)['labels']

        self.Label2idx = dict(zip(list(set(self.origin_labels)),
                                  range(len(list(set(self.origin_labels))))))
        self.label = []
        for l in self.origin_labels:
            self.label.append(self.Label2idx[l])

        img_label = []
        for id, (img, label) in enumerate(zip(self.data, self.label)):
            img_label.append((img, int(label)))
        self.img_label = img_label

        # Transformation
        if split == 'val' or split == 'test':

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif split == 'train':

            image_size = 84

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.label[i]
        image = self.transform(Image.fromarray(img))

        return image, label

    @staticmethod
    def _load_labels(file):
        try:
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
            return data
        except:
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data


if __name__ == '__main__':
    pass
