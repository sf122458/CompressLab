import os
from torchvision import transforms
from PIL import Image
from pathlib import Path
from .base import BaseDataset


class BasicImageDataset(BaseDataset):
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
        self.image_list = os.listdir(self.root)
        self.return_img_info = return_img_info

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.image_list[index])).convert("RGB")
        image = self.transform(image)
        if self.return_img_info:
            filename = os.path.splitext(self.image_list[index])[0]
            return {"image": image, "filename": filename}
        return image


# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/datasets/vimeo90k.py
class Vimeo90kDataset(BaseDataset):
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
    
    
class LICDataset(BaseDataset):
    """The dataset used in DiffEIC, usually used for training and validation."""
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
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