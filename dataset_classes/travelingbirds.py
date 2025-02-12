# TravelingBirds dataset needs to be downloaded from https://worksheets.codalab.org/bundles/0x518829de2aa440c79cd9d75ef6669f27
# as it comes from https://github.com/yewsiang/ConceptBottleneck
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_classes.cub200 import CUB200Class
from dataset_classes.utils import index_list_with_sorting, mask_list


class TravelingBirds(CUB200Class):
    init_base_folder = 'CUB_fixed'
    root = Path.home() / "tmp/Datasets/TravelingBirds"
    crop_root = Path.home() / "tmp/Datasets/PPTravelingBirds"
    def get_all_samples_dir(self, dir):

        self.base_folder = os.path.join(self.init_base_folder, dir)
        main_dir = Path(self.root) / self.init_base_folder / dir
        return self.get_all_sample(main_dir)

    def adapt_to_crop(self):
        self.root = self.crop_root
        folder_name = "train" if self.train else "test"
        folder_name = folder_name + "_cropped"
        self.base_folder = 'CUB_fixed/' + folder_name

    def get_all_sample(self, dir):
        answer = []
        for i, sub_dir in enumerate(sorted(os.listdir(dir))):
            class_dir = dir / sub_dir
            for single_img in os.listdir(class_dir):
                answer.append([Path(sub_dir) / single_img, i + 1])
        return answer
    def _load_metadata(self):
        train_test_split = pd.read_csv(
            os.path.join(Path(self.root).parent / "CUB200", 'CUB_200_2011', 'train_test_split.txt'),
            sep=' ', names=['img_id', 'is_training_img'])
        data = pd.read_csv(
            os.path.join(Path(self.root).parent / "CUB200", 'CUB_200_2011', 'images.txt'),
            sep=' ', names=['img_id', "path"])
        img_dict = {x[1]: x[0] for x in data.values}
        # TravelingBirds has all train+test images in both folders, just with different backgrounds.
        # They are separated by train_test_split of CUB200.
        if self.train:
            samples = self.get_all_samples_dir("train")
            mask = train_test_split["is_training_img"] == 1
        else:
            samples = self.get_all_samples_dir("test")
            mask = train_test_split["is_training_img"] == 0
        ids = np.array([img_dict[str(x[0])] for x in samples])
        sorted = np.argsort(ids)
        samples = index_list_with_sorting(samples, sorted)
        samples = mask_list(samples, mask)
        filepaths = [x[0] for x in samples]
        labels = [x[1] for x in samples]
        samples = pd.DataFrame({"filepath": filepaths, "target": labels})
        self.data = samples
