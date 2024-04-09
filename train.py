import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


DATASET_PATH = 'dataset/'
DATASETS = [
    'moj',
    'gotowy'
]
GESTURES = {
    'finger': 0,
    'fist': 1,
    'palm': 2,
    'peace': 3
}


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = self.load_images()
        self.transform = transform

    def load_images(self):
        images = []
        for dataset in DATASETS:
            print(dataset)
            path = os.path.join(self.root_dir, dataset)
            for gesture, label in GESTURES.items():
                print(gesture)
                class_dir = os.path.join(path, gesture)
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    images.append((image_path, label))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("L")   # convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
    

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])])

dataset = CustomDataset(root_dir=DATASET_PATH, transform=transform)
print(len(dataset))
train_size = int(0.7 * len(dataset))
test_size = int((len(dataset) - train_size) / 2)
val_size = len(dataset) - train_size - test_size
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

num_workers = 2
batch_size = 64
epochs = 20

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


class Net(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
    

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: str):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return train_loss, train_acc

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: str):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
    return test_loss, test_acc


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

model = Net(input_shape=1,
            hidden_units=10,
            output_shape=len(GESTURES))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n------')
    loss, acc = train_step(model=model,
                            data_loader=train_loader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            accuracy_fn=accuracy_fn,
                            device=device)
    train_loss.append(loss.cpu().detach().numpy())
    train_accuracy.append(acc)
    loss, acc = test_step(model=model,
                            data_loader=test_loader,
                            loss_fn=loss_fn,
                            accuracy_fn=accuracy_fn,
                            device=device)
    test_loss.append(loss.cpu())
    test_accuracy.append(acc)

plt.plot(list(range(epochs)), train_loss)
plt.title('Funkcja straty zbioru uczącego w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Funkcja straty')
plt.savefig('train_loss.jpg')
plt.close()

plt.plot(list(range(epochs)), train_accuracy)
plt.title('Accuracy modelu na zbiorze uczącym w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Accuracy [%]')
plt.savefig('train_acc.jpg')
plt.close()

plt.plot(list(range(epochs)), test_loss)
plt.title('Funkcja straty zbioru testowego w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Funkcja straty')
plt.savefig('test_loss.jpg')
plt.close()

plt.plot(list(range(epochs)), test_accuracy)
plt.title('Accuracy modelu na zbiorze testowym w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Accuracy [%]')
plt.savefig('test_acc.jpg')
plt.close()

torch.save(model.state_dict(), 'hand_recognition_model.pth.tar')
