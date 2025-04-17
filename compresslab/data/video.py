from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import random
import torch
from PIL import Image
from compresslab.utils.registry import DataRegistry
import lightning as L
from torchvision import transforms

# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/datasets/video.py
class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...

    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.

    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.

    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")

        splitfile = Path(f"{root}/{split}.list")
        splitdir = Path(f"{root}/sequences")

        if not splitfile.is_file():
            raise RuntimeError(f'Missing file "{splitfile}"')

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        with open(splitfile, "r") as f_in:
            self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]

        self.max_frames = 3  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]

        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        frames = torch.chunk(self.transform(frames), self.max_frames)

        if self.rnd_temp_order:
            if random.random() < 0.5:
                return frames[::-1]

        return frames

    def __len__(self):
        return len(self.sample_folders)
    
@DataRegistry.register("VideoDataModule")
class VideoDataModule(L.LightningDataModule):
    def __init__(self,
                 root: str,
                 batch_size: int = 32,
                 num_workers: int = 4,):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.train_dataset = VideoFolder(
            root=self.root,
            rnd_interval=True,
            rnd_temp_order=True,
            split="train",
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop((256, 256)),
            ])
        )
        self.test_dataset = VideoFolder(
            root=self.root,
            rnd_interval=False,
            rnd_temp_order=False,
            split="test",
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((256, 256)),
            ])
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)