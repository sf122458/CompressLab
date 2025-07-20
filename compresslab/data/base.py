import lightning as L
from torch.utils.data import DataLoader

class BasicDataModule(L.LightningDataModule):
    def __init__(self, 
                 train_data_dir: str, 
                 test_data_dir: str,
                 num_devices: int,
                 batch_size: int = 32, 
                 num_workers: int = 4):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size // num_devices
        self.num_workers = num_workers

    # def setup(self, stage=None):
    #     self.train_dataset = None
    #     self.val_dataset = None
    #     self.test_dataset = None
    #     raise NotImplementedError("This method should be overridden by subclasses.")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)