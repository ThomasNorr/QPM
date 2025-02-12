# Dataset should lie under /root/
# root is currently set to ~/tmp/Datasets/CUB200
# If cropped iamges, like for PIP-Net, ProtoPool, etc. are used, then the crop_root should be set to a folder containing the
# cropped images in the expected structure, obtained by following ProtoTree's instructions.
# https://github.com/M-Nauta/ProtoTree/blob/main/README.md#preprocessing-cub
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from dataset_classes.utils import txt_load


class CUB200Class(Dataset):
    root = Path.home() / "tmp/Datasets/CUB200"
    crop_root = Path.home() / "tmp/Datasets/PPCUB200"
    name = "CUB2011"
    base_folder = 'CUB_200_2011/images'
    def __init__(self,  train, transform, crop=True):
        self.train = train
        self.transform = transform
        self.crop = crop
        self._load_metadata()
        self.loader = default_loader

        if crop:
            self.adapt_to_crop()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def adapt_to_crop(self):
      #  ds_name = [x for x in self.cropped_dict.keys() if x in self.root][0]
        self.root = self.crop_root
        folder_name = "train" if self.train else "test"
        folder_name = folder_name + "_cropped"
        self.base_folder = 'CUB_200_2011/' + folder_name

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        img = self.transform(img)
        return img, target

    @classmethod
    def get_image_attribute_labels(self, train=False):
        image_attribute_labels = pd.read_csv(
          Path.home() / 'tmp/Datasets/CUB200'/'CUB_200_2011'/ "attributes"/
                         'image_attribute_labels.txt',
            sep=' ', names=['img_id', 'attribute', "is_present", "certainty", "time"], on_bad_lines="skip")
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        merged = image_attribute_labels.merge(train_test_split, on="img_id")
        filtered_data = merged[merged["is_training_img"] == train]
        return filtered_data


    def get_indices_for_target(self, index):
        return np.where(self.data["target"] == index + 1)[
            0]

    @staticmethod
    def filter_attribute_labels(labels, min_certainty=3):
        is_invisible_present = labels[labels["certainty"] == 1]["is_present"].sum()
        if is_invisible_present != 0:
            raise ValueError("Invisible present")
        labels["img_id"] -= min(labels["img_id"])
        labels["img_id"] = fillholes_in_array(labels["img_id"])
        labels[labels["certainty"] == 1]["certainty"] = 4
        labels = labels[labels["certainty"] >= min_certainty]
        labels["attribute"] -= min(labels["attribute"])
        labels = labels[["img_id", "attribute", "is_present"]]
        labels["is_present"] = labels["is_present"].astype(bool)
        return labels



    @classmethod
    def get_class_sim(self, base_folder=None):
        if base_folder is None:
            base_folder = self.root
        path = base_folder / "CUB_200_2011/class_sim_gts/class_sim_gt.npy"

        def calc_class_sim_gt():
            data = txt_load(
                Path.home() / "tmp/Datasets/CUB200/CUB_200_2011/attributes/class_attribute_labels_continuous.txt").splitlines()
            data = [x.split(" ") for x in data]
            data = [[float(entry) for entry in x] for x in data]
            data = np.array(data)
            n_classes = 200
            class_sim_gt = np.zeros((n_classes, n_classes))
            class_sim_gt_cbm = np.zeros((n_classes, n_classes))
            for i in range(n_classes):
                for j in range(n_classes):
                    class_sim_gt[i, j] = data[i, :] @ data[j, :] / (
                            np.linalg.norm(data[i, :]) * np.linalg.norm(data[j, :]))

            return class_sim_gt

        if os.path.exists(path):
            class_sim_gt = np.load(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            class_sim_gt = calc_class_sim_gt()
            np.save(path, class_sim_gt)
        class_sim_gt[np.eye(200) == 1] = 0
        return class_sim_gt


def fillholes_in_array(array):
    unique_values = np.unique(array)
    mapping = {x: i for i, x in enumerate(unique_values)}
    array = array.map(mapping)
    return array




def load_cub_class_mapping():
    mapping_path = CUB200Class.root / "labelMapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            data = json.load(f)
        return data
    else:
        answer = calculate_cub_mapping()
        with open(mapping_path, "w") as f:
            json.dump(answer, f)
        return answer



def calculate_cub_mapping():
    path = CUB200Class.root / "CUB_200_2011" / "images"
    answer_dict = {}
    for item in os.listdir(path):
        value, label = item.split(".", 1)
        value = int(value)
        answer_dict[value - 1] = label
    return answer_dict