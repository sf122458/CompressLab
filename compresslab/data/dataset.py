import lightning as L
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BasicImageDataset(Dataset):
    """
    A simplest image dataset.
    """
    def __init__(self,
                 path: str,
                 transform=None,
                 **kwargs):
        
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

class BasicImageDataModule(L.LightningDataModule):
    def __init__(self, train_data_dir: str, test_data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def setup(self, stage):
        self.data_fit = BasicImageDataset(
            self.train_data_dir, 
            transform=self.train_transform
        )
        self.data_test = BasicImageDataset(
            self.test_data_dir, 
            transform=self.test_transform
        )

    def train_dataloader(self):
        return DataLoader(self.data_fit, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)