from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import random
import torch
from PIL import Image
from compresslab.utils.registry import DataRegistry
import lightning as L
from torchvision import transforms
import os
import torchvision.transforms.functional as F
from pytorch_msssim import ms_ssim
from typing import List

class SequenceRandomCrop:
    """Perform a certain random crop on a sequence of images."""
    def __init__(self, size: int=256):
        assert isinstance(size, int) and size > 0, "size must be a positive integer"
        self.crop_size = (size, size)
        
    def __call__(self, img_list: List[torch.Tensor]) -> List[torch.Tensor]:
        i, j, h, w = transforms.RandomCrop.get_params(
            img_list[0], output_size=self.crop_size
        )

        cropped_img_list = [F.crop(img, i, j, h, w) for img in img_list]
        
        return cropped_img_list

# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/datasets/video.py
class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directory containing many sub-directorie like Vimeo90K Dataset:

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


class Vimeo90kDataset(Dataset):
    def __init__(
            self, 
            root: str = "./dataset/vimeo_setuplet",
            crop_size: int = 256,
            rnd_interval: bool = False
    ):
        splitfile = os.path.join(root, "sep_trainlist.txt")
        splitdir = os.path.join(root, "sequences")

        if not os.path.exists(splitfile):
            raise RuntimeError(f'Missing file "{splitfile}"')

        if not os.path.exists(splitdir) or not os.path.isdir(splitdir):
            raise RuntimeError(f'Missing directory "{splitdir}"')
        
        with open(splitfile, "r") as f_in:
            self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]

        self.max_frames = 3  # hard coding for now
        self.rnd_interval = rnd_interval
        self.transform = transforms.ToTensor()
        self.random_crop = SequenceRandomCrop(crop_size)
        
    def __len__(self):
        return len(self.sample_folders)
        
    def __getitem__(self, index):
        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())
    
        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]

        input_images = []
        for frame_path in frame_paths:
            input_image = Image.open(frame_path).convert("RGB")
            input_image = self.transform(input_image)
            input_images.append(input_image)

        cropped_images = self.random_crop(input_images)
        return cropped_images
        

class UVGDataset(Dataset):
    """Dataset for UVG sequences. Used for the evaluation of IP-frame codec.
    This dataset will return a sequence of images. The compression of the I-frame and P-frames is done in the model.
    """
    def __init__(self, 
                 root: str, 
                 test_full: bool = False,
                 gop_size: int = 12,
                 ):
        """
        Args:
            root(str): root directory of the UVG dataset
            test_full(bool): if True, all frames in the folders will be used for testing. Otherwise, only the first gop will be used.
            gop_size(int): size of the group of pictures (GOP). Default is 12.
        """
        
        folders = ["Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySteadyGo", "ShakeNDry", "YachtRide"]

        self.transform = transforms.ToTensor()

        self.input = []

        for idx, folder in enumerate(folders):
            img_list = os.listdir(os.path.join(root, folder))

            cnt = 0
            for img in img_list:
                if img.endswith('.png'):
                    cnt += 1

            if test_full:
                frame_range = cnt // gop_size
            else:
                frame_range = 1

            for i in range(frame_range):
                input_path = []
                for j in range(gop_size):
                    input_path.append(os.path.join(root, folder, f"im{i * gop_size + j + 1:03d}.png"))
                
                self.input.append(input_path)

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        input_images = []
        for filename in self.input[index]:
            input_image = Image.open(filename)
            input_image = self.transform(input_image)
            h, w = input_image.shape[-2], input_image.shape[-1]
            h, w = h // 64 * 64, w // 64 * 64
            input_image = F.center_crop(input_image, (h, w))
            input_images.append(input_image)
        return input_images
    
#TODO
class HEVCDataset(Dataset):
    """Dataset for HEVC test sequences. Usually used for evaluation.
    """
    def __init__(self, 
                 root: str,
                 refdir: str,
                 crf: str = 'H265QP22',
                 type: str ='B'):

        self.ref = []

        self.hevcclass = {
            'A': ['Nebuta', 'PeopleOnStreet', 'SteamLocomotive', 'Traffic'],
            'B': ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene'],
            'C': ['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'],
            'D': ['BasketballPass', 'BlowingBubbles', 'BQSquare', 'RaceHorses'],
            'E': ['FourPeople', 'Johnny', 'KristenAndSara'],
            'F': ['BasketballDrillText', 'ChinaSpeed', 'SlideEditing', 'SlideShow']
            }
        
        assert type in self.hevcclass.keys(), f"Type {type} not supported. Available types: {list(self.hevcclass.keys())}"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        intra_bpp = {}

        for idx, seq in enumerate(self.hevcclass[type]):
            seq_intra_bpp = intra_bpp[idx]

            img_list = os.listdir(os.path.join())
            
    def getbpp(self, crf):
        if crf == 'H265QP22':
            intra_bpp = []

    def __len__(self):
        return len(self.ref)
    
    def __getitem__(self, index):
        ref_image = Image.open(self.ref[index])
        ref_image = self.transform(ref_image)


@DataRegistry.register("VideoDataModule", define_path=__file__)
class VideoDataModule(L.LightningDataModule):
    """
    Lightning DataModule for loading video datasets for video compression tasks.
    """
    def __init__(self,
                 train_data_dir: str = "data/vimeo_setuplet/sequences/",
                 test_data_dir: str = "data/UVG/images/",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_crop_size: int = 256,
                 test_gop_size: int = 12):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = train_crop_size
        self.gop_size = test_gop_size

    def setup(self, stage):
        self.train_dataset = Vimeo90kDataset(
            root=self.train_data_dir,
            rnd_interval=True,
            crop_size=self.crop_size
        )
        self.val_dataset = UVGDataset(
            root=self.test_data_dir,
            test_full=False,
            gop_size=self.gop_size
        )
        self.test_dataset = UVGDataset(
            root=self.test_data_dir,
            test_full=True,
            gop_size=self.gop_size
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
