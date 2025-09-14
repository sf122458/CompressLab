import os
from PIL import Image
from pathlib import Path
from .base import BasicDataset

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pickle
from torchvision.datasets.utils import check_integrity

class BasicImageDataset(BasicDataset):
    """
    A simplest image dataset.
    """
    def __init__(
        self,
        return_img_info=False,
        **kwargs
    ):
        """
        Args:
            return_img_info (bool, optional): If true, return the image and the extra information
                in a dictionary format. Defaults to False.
        """
        super().__init__(**kwargs)
        if isinstance(self.root, list): # support multiple root directories for training
            self.image_list = []
            for r in self.root:
                image_paths = os.listdir(r)
                for image_path in image_paths:
                    self.image_list.append(os.path.join(r, image_path))
        else:
            image_paths = os.listdir(self.root)
            self.image_list = []
            for image_path in image_paths:
                self.image_list.append(os.path.join(self.root, image_path))
                
        self.return_img_info = return_img_info

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        image = self.transform(image)
        if self.return_img_info:
            filename = os.path.basename(self.image_list[index]).split('.')[0]
            return {"image": image, "filename": filename, "dataset": os.path.basename(os.path.dirname(self.image_list[index]))}
        return image


# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/datasets/vimeo90k.py
class Vimeo90kDataset(BasicDataset):
    """Load a Vimeo-90K structured dataset.
    This dataset is used for training.

    Vimeo-90K dataset from
    Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, William T. Freeman:
    `"Video Enhancement with Task-Oriented Flow"
    <https://arxiv.org/abs/1711.09078>`_,
    International Journal of Computer Vision (IJCV), 2019.

    Training and testing image samples are respectively stored in
    separate directories:

    .. code-block::

        - rootdir/
            - sequence/
                - 00001/001/im1.png
                - 00001/001/im2.png
                - 00001/001/im3.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'valid')
        tuplet (int): order of dataset tuplet (e.g. 3 for "triplet" dataset)
    """

    def __init__(
        self,
        split="train", 
        tuplet=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert isinstance(self.root, str), "The root should be a string."
        list_path = Path(self.root) / self._list_filename(split, tuplet)

        with open(list_path) as f:
            self.samples = [
                f"{self.root}/sequences/{line.rstrip()}/im{idx}.png"
                for line in f
                if line.strip() != ""
                for idx in range(1, tuplet + 1)
            ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

    def _list_filename(self, split: str, tuplet: int) -> str:
        tuplet_prefix = {3: "tri", 7: "sep"}[tuplet]
        list_suffix = {"train": "trainlist", "valid": "testlist"}[split]
        return f"{tuplet_prefix}_{list_suffix}.txt"
    
    
class LICDataset(BasicDataset):
    """The dataset used in DiffEIC, usually used for training and validation."""
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(self.root, str), "The root should be a string."
        self.image_list = self.list_image_files(self.root)
    
    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        image = self.transform(image)
        return image
        
    
    def list_image_files(self, root, exts=(".jpg", ".png", ".jpeg"), max_size=1):
        """List all image files in a directory and its subdirectories.
        """
        files = []
        for dir_path, _, file_names in os.walk(root):
            early_stop = False
            for file_name in file_names:
                if os.path.splitext(file_name)[1].lower() in exts:
                    if max_size >= 0 and len(files) >= max_size:
                        early_stop = True
                        break
                    files.append(os.path.join(dir_path, file_name))
            if early_stop:
                break
        return files
    
class CIFAR10(BasicDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        **kwargs: Other arguments for BasicDataset.
    """
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        train: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        assert isinstance(self.root, str), "The root should be a string."
        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True