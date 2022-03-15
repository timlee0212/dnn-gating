from core.dataset import Dataset, ImageDataset

class CIFAR10(Dataset, ImageDataset):
    def __init__(self, batch_size, val_batch_size, ):
        