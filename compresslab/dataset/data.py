import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Dict, List, Any

from compresslab.utils.registry import TransformRegistry

class ImageDataset(Dataset):
    def __init__(self,
                 path: str,
                 transform: Dict[str, Any]=None):
        
        self.path = path
        
        if transform is not None:
            t_list = [transforms.ToTensor()]
            print(transform.items())
            for (key, params) in transform.items():
                print(key, params)
                if params is not None:
                    t_list.append(TransformRegistry.get(key)(**params))
                else:
                    t_list.append(TransformRegistry.get(key)())

            self.transform = transforms.Compose(t_list)
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        self.image_list = os.listdir(self.path)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.image_list[index]))
        image = self.transform(image)
        return image
