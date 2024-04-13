import os
from torch.utils.data import Dataset
from PIL import Image


DATASETS = [
    'moj',
    'gotowy'
]
GESTURES = {
    'one': 0,
    'fist': 1,
    'stop': 2,
    'peace': 3,
    'four': 4,
    'three2': 5,
    'rock': 6,
}

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = self.load_images()
        self.transform = transform

    def load_images(self) -> list[tuple]:
        images = []
        for dataset in DATASETS:
            path = os.path.join(self.root_dir, dataset)
            for gesture, label in GESTURES.items():
                class_dir = os.path.join(path, gesture)
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    images.append((image_path, label))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("L")   # convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
    