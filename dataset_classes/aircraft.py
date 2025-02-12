from __future__ import annotations

import os
import pathlib
from typing import Any, Callable, Optional, Tuple

import PIL.Image
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive


class FGVCAircraftClass(VisionDataset):
    """`FGVC Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    The dataset contains 10,200 images of aircraft, with 100 images for each of 102
    different aircraft model variants, most of which are airplanes.
    Aircraft models are organized in a three-levels hierarchy. The three levels, from
    finer to coarser, are:

    - ``variant``, e.g. Boeing 737-700. A variant collapses all the models that are visually
        indistinguishable into one class. The dataset comprises 102 different variants.
    - ``family``, e.g. Boeing 737. The dataset comprises 70 different families.
    - ``manufacturer``, e.g. Boeing. The dataset comprises 41 different manufacturers.

    Args:
        split (string, optional): The dataset split, supports ``train``, ``val``,
            ``trainval`` and ``test``.
        annotation_level (str, optional): The annotation level, supports ``variant``,
            ``family`` and ``manufacturer``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    root = pathlib.Path.home() / "tmp" / "Datasets" / "FGVCAircraft"
    def __init__(
            self,

            train: bool = True,
            annotation_level: str = "variant",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
    ) -> None:
        super().__init__(self.root, transform=transform, target_transform=target_transform)
        self._split = "trainval" if train else "test"  # verify_str_arg(split, "split", ("train", "val", "trainval", "test"))
        self._annotation_level = verify_str_arg(
            annotation_level, "annotation_level", ("variant", "family", "manufacturer")
        )

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        annotation_file = os.path.join(
            self._data_path,
            "data",
            {
                "variant": "variants.txt",
                "family": "families.txt",
                "manufacturer": "manufacturers.txt",
            }[self._annotation_level],
        )
        with open(annotation_file, "r") as f:
            self.classes = [line.strip() for line in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        image_data_folder = os.path.join(self._data_path, "data", "images")
        labels_file = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{self._split}.txt")

        # self._image_files = []
        #  self._labels = []
        self.samples = []
        targets = []
        with open(labels_file, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                #  self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                # self._labels.append(self.class_to_idx[label_name])
                self.samples.append(
                    (os.path.join(image_data_folder, f"{image_name}.jpg"), self.class_to_idx[label_name]))
                targets.append(self.class_to_idx[label_name])
        self.targets = np.array(targets)

    # def get_indices_for_target(self, index):
    #     return np.where(self.targets == index)[
    #         0]
    #
    # def get_feature_labels(self):
    #     return pd.Series(self.targets)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self.samples[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _download(self) -> None:
        """
        Download the FGVC Aircraft dataset archive and extract it under root.
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, self.root)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)
