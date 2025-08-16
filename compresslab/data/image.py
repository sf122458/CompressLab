import lightning as L
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from compresslab.utils.registry import DataRegistry
from pathlib import Path
from .base import BasicDataModule

class BasicImageDataset(Dataset):
    """
    A simplest image dataset.
    """
    def __init__(
        self,
        path: str,
        transform=None,
        **kwargs
    ):
        
        self.path = path
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.image_list = os.listdir(self.path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.image_list[index]))
        image = self.transform(image)
        return image

@DataRegistry.register("BasicImageDataModule", define_path=__file__)
class BasicImageDataModule(BasicDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.train_dataset = BasicImageDataset(
            self.train_data_dir, 
            transform=train_transform
        )

        self.val_dataset = BasicImageDataset(
            self.test_data_dir, 
            transform=test_transform
        )

        self.test_dataset = BasicImageDataset(
            self.test_data_dir, 
            transform=test_transform
        )
        


# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/datasets/vimeo90k.py
class Vimeo90kDataset(Dataset):
    """Load a Vimeo-90K structured dataset.

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

    def __init__(self, root, transform=None, split="train", tuplet=3):
        list_path = Path(root) / self._list_filename(split, tuplet)

        with open(list_path) as f:
            self.samples = [
                f"{root}/sequences/{line.rstrip()}/im{idx}.png"
                for line in f
                if line.strip() != ""
                for idx in range(1, tuplet + 1)
            ]

        self.transform = transform

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
    
@DataRegistry.register("Vimeo90kImageDataModule", define_path=__file__)
class Vimeo90kImageDataModule(BasicDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_dataset = Vimeo90kDataset(
            root=self.train_data_dir,
            transform=train_transform,
            split="train",
            tuplet=7
        )

        self.val_dataset = BasicImageDataset(
            self.test_data_dir, 
            transform=test_transform
        )

        self.test_dataset = BasicImageDataset(
            self.test_data_dir, 
            transform=test_transform
        )