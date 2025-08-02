import lightning as L
from torch.utils.data import DataLoader

class BasicDataModule(L.LightningDataModule):
    def __init__(self, 
                 train_data_dir: str, 
                 test_data_dir: str,
                 num_devices: int,
                 batch_size: int = 32, 
                 num_workers: int = 4):
        """
        Args:
            train_data_dir (str): Directory for training data.
            test_data_dir (str): Directory for testing data.
            num_devices (int): Number of devices for distributed training.
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size_per_device = batch_size // num_devices
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size_per_device,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1, # avoid OOM in validation
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)