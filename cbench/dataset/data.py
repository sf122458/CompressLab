from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 transform=None):
        

        t = transforms.Compose([
            transforms.ToTensor()
        ])


    def __len__(self):
        pass

    def __getitem__(self, index):
        pass