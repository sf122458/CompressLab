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
        image = Image.open(os.path.join(self.root, self.image_list[index]))
        image = self.transform(image)
        if self.return_img_info:
            filename = os.path.splitext(self.image_list[index])[0]
            return {"image": image, "filename": filename}
        return image


# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/datasets/vimeo90k.py
class Vimeo90kDataset(BaseDataset):
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
    
# @DataRegistry.register("Vimeo90kImageDataModule", define_path=__file__)
# class Vimeo90kImageDataModule(BasicDataModule):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         train_transform = transforms.Compose([
transforms.RandomCrop((256, 256)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ])

#         test_transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])

#         self.train_dataset = Vimeo90kDataset(
#             root=self.train_data_dir,
#             transform=train_transform,
#             split="train",
#             tuplet=7
#         )

#         self.val_dataset = BasicImageDataset(
#             self.test_data_dir, 
#             transform=test_transform
#         )

#         self.test_dataset = BasicImageDataset(
#             self.test_data_dir, 
#             transform=test_transform,
#             return_img_info=True
#         )