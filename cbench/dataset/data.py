from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Dict, List, Any

from cbench.utils.registry import TransformRegistry

class ImageDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 transform: Dict[str, Any]=None):
        
        t_list = []
        for key, params in transform:
            t_list.append(TransformRegistry.get(key)(**params))

        self.transform = transforms.Compose(t_list)
        


    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

if __name__ == "__main__":
    transforms.ToTensor(None)