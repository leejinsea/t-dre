import torch
import json
import pickle
import os
import h5py
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms      #torchvision transform


device = torch.device('cuda')

class VGDataset(Dataset):
    def __init__(self, img_dir_root, vg_data_path, look_up_tables_path, dataset_type=None, transform=None, train_caption = None):

        assert dataset_type in {None, 'train', 'test', 'val'}

        super(VGDataset, self).__init__()

        self.train_target = train_caption
        self.img_dir_root = img_dir_root
        self.vg_data_path = vg_data_path
        self.look_up_tables_path = look_up_tables_path
        self.dataset_type = dataset_type  # if dataset_type is None, all data will be use
        self.transform = transform

        # === load data here ====
        self.look_up_tables = pickle.load(open(look_up_tables_path, 'rb'))

    def set_dataset_type(self, dataset_type, verbose=True):

        assert dataset_type in {None, 'train', 'test', 'val'}

        if verbose:
            print('[DenseCapDataset]: {} switch to {}'.format(self.dataset_type, dataset_type))

        self.dataset_type = dataset_type

    def __getitem__(self, idx):


        with h5py.File(self.vg_data_path, 'r') as vg_data:

            vg_idx = self.look_up_tables['split_des'][self.dataset_type][
                idx] if self.dataset_type else idx

            img_path = os.path.join('C:\datasets\\visual_genome\origin\\vg_images',
                                    self.look_up_tables['idx_to_filename'][
                                        vg_idx])

            raw_img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(raw_img)
            else:
                img = transforms.ToTensor()(raw_img)

            first_box_idx = vg_data['img_to_first_box'][vg_idx]
            last_box_idx = vg_data['img_to_last_box'][vg_idx]


            regions = torch.as_tensor(
                vg_data['boxes'][first_box_idx: last_box_idx + 1],
                dtype=torch.float32)

            caps = torch.as_tensor(vg_data['captions'][first_box_idx: last_box_idx + 1],
                dtype=torch.long)
            caps_len = torch.as_tensor(vg_data['lengths'][first_box_idx: last_box_idx + 1],
                dtype=torch.long)


            gt_labels = torch.ones((regions.shape[0],), dtype=torch.int64)

            reg_targets = {
                'boxes': regions,
                'labels': gt_labels,
            }


            caption_target = {
                'boxes': regions,
                'caps': caps,
                'caps_len': caps_len
            }

        return raw_img, img, reg_targets, caption_target

    def __len__(self):

        if self.dataset_type:
            return len(self.look_up_tables['split_des'][self.dataset_type])
        else:
            return len(self.look_up_tables['filename_to_idx'])

    def collate_fn(Data):

        raw_img, imgs, targets_r, target_cap = zip(*Data)

        return raw_img, imgs, targets_r, target_cap
