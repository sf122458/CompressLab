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

class PairRandomCrop:
    """Perform a certain random crop on a pair of images."""
    def __init__(self, size: int=256):
        assert isinstance(size, int) and size > 0, "size must be a positive integer"
        self.crop_size = (size, size)
        
    def __call__(self, img1, img2):
        assert img1.shape == img2.shape, "Input images must have the same size, got {} and {}".format(img1.shape, img2.shape)
        
        i, j, h, w = transforms.RandomCrop.get_params(
            img1, output_size=self.crop_size
        )
        
        img1 = F.crop(img1, i, j, h, w)
        img2 = F.crop(img2, i, j, h, w)
        
        return img1, img2

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
    

class Vimeo90kDataset(Dataset):
    """Dataset for Vimeo90k sequences. Usually used for training.
    """
    def __init__(self, 
                 rootdir: str = "./dataset/vimeo_setuplet/sequences/", 
                 size: int = 256):
        """
        Args:
            rootdir (string): root directory of the dataset
            size (int): size of the cropped images, default is 256
        """
        self.image_input_list, self.image_ref_list = self.get_vimeo(rootdir=rootdir)

        print(f"Found {len(self.image_input_list)} input images and {len(self.image_ref_list)} reference images.")

        self.transform = transforms.ToTensor()
        self.random_crop = PairRandomCrop(size)

    def get_vimeo(self, 
                  rootdir: str = "./dataset/vimeo_setuplet/sequences/",
                  filefolderlist: str ="./dataset/vimeo_setuplet/test.txt"):
        """
        Args:
            rootdir (string): root directory of the dataset
            filefolderlist (string): path to the file containing list of folders
        Returns:
            train_frame (list): list of input image paths
            ref_frame (list): list of reference image paths
        """
        with open(filefolderlist) as f:
            data = f.readlines()

        train_frame = []
        ref_frame = []

        for line in data:
            filename = os.path.join(rootdir, line.strip())
            train_frame += [filename]
            refnumber = int(filename[-5:-4]) - 2
            refname = filename[0:-5] + str(refnumber) + '.png'
            ref_frame += [refname]

        return train_frame, ref_frame

    def __len__(self):
        return len(self.image_input_list)
    
    def __getitem__(self, index):
        input_image = Image.open(self.image_input_list[index])
        ref_image = Image.open(self.image_ref_list[index])

        input_image = self.transform(input_image)
        ref_image = self.transform(ref_image)

        input_image, ref_image = self.random_crop(input_image, ref_image)

        return input_image, ref_image
    

class UVGDataset(Dataset):
    """Dataset for UVG sequences. Usually used for evaluation.
    """
    def __init__(self, root: str,
                 lmbda: int = 1024,
                 test_full: bool = False):

        folders = ["Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySteadyGo", "ShakeNDry", "YachtRide"]

        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []

        self.transform = transforms.ToTensor()

        assert lmbda in [256, 512, 1024, 2048], "lmbda must be one of [256, 512, 1024, 2048]"

        if lmbda == 2048:
            refdir = 'H265L20'
        elif lmbda == 1024:
            refdir = 'H265L23'
        elif lmbda == 512:
            refdir = 'H265L26'
        elif lmbda == 256:
            refdir = 'H265L29'

        intra_frame_bpp = self.getbpp(refdir)

        for idx, folder in enumerate(folders):
            seqIbpp = intra_frame_bpp[idx]
            img_list = os.listdir(os.path.join(root, folder))

            cnt = 0
            for img in img_list:
                if img[-4:] == '.png':
                    cnt += 1
            
            if test_full:
                framerange = cnt // 12
            else:
                framerange = 1
            
            for i in range(framerange):
                refpath = os.path.join(root, folder, refdir, "im" + str(i * 12 + 1).zfill(4) + ".png")
                inputpath = []
                for j in range(12):
                    inputpath.append(os.path.join(root, folder, "im" + str(i * 12 + j + 1).zfill(3) + ".png"))
                
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            
    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = [1.2929020996093752,
                    0.6758680826822915,
                    0.94005859375,
                    0.6770526529947917,
                    0.7543700358072918,
                    0.8640651041666668,
                    0.6924034016927084]
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = [0.7243849283854167,
                    0.471212158203125,
                    0.5672164713541666,
                    0.3604554036458334,
                    0.550234619140625,
                    0.5805125325520833,
                    0.5005953776041667]
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = [0.3411645507812501,
                    0.3319017740885417,
                    0.36831477864583334,
                    0.20547257486979167,
                    0.40615576171875,
                    0.40312744140625,
                    0.36251399739583334]
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = [0.14195882161458334,
                    0.23096484374999998,
                    0.259547607421875,
                    0.13520450846354168,
                    0.302080322265625,
                    0.28926432291666665,
                    0.26244807942708337]
        else:
            raise FileNotFoundError('cannot find ref : ', ref_i_folder)
        if len(Ibpp) == 0:
            raise ValueError('You need to generate I frames and fill the bpps above!')
        return Ibpp
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        ref_image = Image.open(self.ref[index])
        ref_image = self.transform(ref_image)
        h, w = ref_image.shape[1], ref_image.shape[2]
        h, w = h // 64 * 64, w // 64 * 64
        ref_image = F.center_crop(ref_image, (h, w))
        input_images = []
        ref_psnr = None
        ref_msssim = None
        for filename in self.input[index]:
            input_image = Image.open(filename)
            input_image = self.transform(input_image)
            input_image = F.center_crop(input_image, (h, w))

            if ref_psnr is None:
                ref_psnr = 10 * torch.log10(1 / ((ref_image - input_image) ** 2).mean()).item()
                ref_msssim = ms_ssim(ref_image.unsqueeze(0), input_image.unsqueeze(0), data_range=1.0).item()
            else:
                input_images.append(input_image)

        return torch.stack(input_images), ref_image, self.refbpp[index], ref_psnr, ref_msssim

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



@DataRegistry.register("CompressAIVideoDataModule")
class CompressAIVideoDataModule(L.LightningDataModule):
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
    

@DataRegistry.register("DVCDataModule")
class DVCDataModule(L.LightningDataModule):
    def __init__(self,
                 train_data_dir: str = "data/vimeo_setuplet/sequences/",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 test_data_dir: str = "data/UVG/images/",
                 lmbda: int = 1024,
                 size: int = 256):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_lmbda = lmbda
        self.size = size

    def setup(self, stage=None):
        self.train_dataset = Vimeo90kDataset(rootdir=self.train_data_dir, size=self.size)
        self.test_dataset = UVGDataset(root=self.test_data_dir, 
                                       lmbda=self.test_lmbda,
                                       test_full=False)

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