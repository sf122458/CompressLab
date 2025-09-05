import logging
import lightning as L
from typing import Dict, Any, Union, List
from torch.utils.data import DataLoader, Dataset
from compresslab.utils.config import DatasetConfig
from compresslab.utils.registry import DataRegistry
from torchvision import transforms

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
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
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

class BaseDataset(Dataset):
    """A dataset base class that processes the transformations for images.
    """
    def __init__(
        self,
        root: Union[str, List[str]],
        transform: Dict[str, Any] = None,
    ):
        self.root = root
        transform_list = []
        if transform is not None:
            for key, value in transform.items():
                if value is not None:
                    transform_list.append(getattr(transforms, key)(**value))
                else:
                    transform_list.append(getattr(transforms, key)())
        transform_list.append(transforms.ToTensor())    
        self.transform = transforms.Compose(transform_list)
            
        
class DataModule(L.LightningDataModule):
    def __init__(self, 
                 train: DatasetConfig,
                 val: DatasetConfig,
                 test: Union[DatasetConfig, List[DatasetConfig]],
                 # dataloader parameters
                 num_devices: int,
                 batch_size: int = 32, 
                 num_workers: int = 4,
                 test_only: bool = False):
        super().__init__()
        
        self.test_only = test_only
        
        self.batch_size_per_device = batch_size // num_devices
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        
        if train is None or val is None:
            logging.warning("Train or Val dataset is not provided. Automatically set to test only mode.")
            self.test_only = True
        elif not self.test_only:
            self.train_dataset = DataRegistry.get(train.Key)(**train.Params)
            self.val_dataset = DataRegistry.get(val.Key)(**val.Params)    
        
            
        if isinstance(test, list):
            self.test_dataset = [DataRegistry.get(t.Key)(**t.Params) for t in test]
        else:
            self.test_dataset = DataRegistry.get(test.Key)(**test.Params)
    
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
        if isinstance(self.test_dataset, list):
            return [DataLoader(ds,
                               batch_size=1,
                               shuffle=False,
                               num_workers=self.num_workers) for ds in self.test_dataset]
        else:
            return DataLoader(self.test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=self.num_workers)