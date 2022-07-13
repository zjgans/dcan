import os
import os.path as osp
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetLoader(Dataset):

    def __init__(self, split, args=None):

        self.dataset = args.dataset
        self.split = split
        self.root_path = args.data_root
        print("Current train dataset: ", self.dataset, self.split)
        if self.root_path is None:
            self.root_path = "/media/a504/D/zhouj/dataset"

        with open(os.path.join(self.root_path, self.dataset, "{}.pickle".format(self.split)), 'rb') as f:
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


        # Transformation
        if split == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])

        else:

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.label[i]
        image = self.transform(img)

        return image, label


if __name__ == '__main__':
    pass

