import torch
import torch.utils.data as tordata
import os.path as osp
import numpy as np
from functools import partial


class CTPatchDataset(tordata.Dataset):
    def __init__(self, npy_root, hu_range, transforms=None):
        self.transforms = transforms
        self.root = npy_root
        self.hu_min, self.hu_max = hu_range
        self.file = np.load(self.root, mmap_mode='r', allow_pickle=True, encoding='bytes')

    def __getitem__(self, index):
        assert index < len(self.root)
        data = self.file[index].astype('float32')
        data = data - 1024
        data = torch.from_numpy(data)
        # normalize to [0, 1]
        data = (torch.clamp(data, self.hu_min, self.hu_max) - self.hu_min) / (self.hu_max - self.hu_min)
        low_dose, full_dose = data
        if self.transforms is not None:
            low_dose = self.transforms(low_dose)
            full_dose = self.transforms(full_dose)
        return low_dose, full_dose

    def __len__(self):
        return len(self.root)


data_root = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset')
dataset_dict = {
    'cmayo_train_64': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/train_64.npy')),
    'cmayo_test_512': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/test_512.npy')),
}
