import torchvision.transforms as transforms
from compresslab.utils.registry import TransformRegistry

TransformRegistry.register("ToTensor")(transforms.ToTensor)

TransformRegistry.register("RandomCrop")(transforms.RandomCrop)