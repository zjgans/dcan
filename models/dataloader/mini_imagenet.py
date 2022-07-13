import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pickle
import numpy as np


class MiniImageNet(Dataset):

    def __init__(self, split, args):
        self.dataset = args.dataset
        self.split = split
        self.root_path = args.data_root
        print("Current train dataset: ", self.dataset, self.split)
        if self.root_path is None:
            self.root_path = "/media/a504/D/zhouj/dataset"

        if self.split == "test":
            split_name = "{}_category_split_test.pickle".format(self.dataset)
        elif self.split == "val":
            split_name = "{}_category_split_val.pickle".format(self.dataset)
        else:
            split_name = "{}_category_split_train_phase_train.pickle".format(self.dataset)

        with open(os.path.join(self.root_path, self.dataset, split_name), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.data = data['data']
            self.origin_labels = data['labels']

        self.Label2idx = dict(zip(list(set(self.origin_labels)),
                                  range(len(list(set(self.origin_labels))))))
        self.label = []
        for l in self.origin_labels:
            self.label.append(self.Label2idx[l])

        img_label = []
        for id, (img, label) in enumerate(zip(self.data, self.label)):
            img_label.append((img, label))
        self.img_label = img_label


        if self.split == 'val' or self.split == 'test':

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif self.split == 'train':
            image_size = 84

            self.transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.label[i]
        image = self.transform(img)

        return image, label


if __name__ == '__main__':
    pass