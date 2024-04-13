import os
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from cnn_clasifier.dataset import CustomDataset, GESTURES
from cnn_clasifier.architecture import Net


class Trainer:
    def __init__(self, 
                 dataset_path: str, 
                 batch_size: int = 64, 
                 num_workers: int = 6) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.dataset = CustomDataset(root_dir=dataset_path, transform=transform)
        self.train_loader, self.val_loader, self.test_loader = self._split_dataset(batch_size, num_workers)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        
    def train(self, 
              epochs: int = 20, 
              learning_rate: float = 0.01) -> None:
        writer = SummaryWriter()
        model = Net(input_shape=1,
                    hidden_units=10,
                    output_shape=len(GESTURES))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
        
        for epoch in tqdm(range(epochs)):
            print(f'Epoch: {epoch}\n------')
            self._train_step(model=model,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             epoch=epoch,
                             writer=writer)
            
            self._test_step(model=model,
                            loss_fn=loss_fn,
                            epoch=epoch,
                            writer=writer)
            
        torch.save(model.state_dict(), 'hand_recognition_model.pth.tar')
        
    def _split_dataset(self, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_size = int(0.7 * len(self.dataset))
        test_size = int((len(self.dataset) - train_size) / 2)
        val_size = len(self.dataset) - train_size - test_size
        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def _train_step(self,
                    model: torch.nn.Module,
                    loss_fn: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    writer: SummaryWriter) -> None:
        train_loss, train_acc = 0, 0
        model.to(self.device)
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += self._accuracy_fn(y_true=y,
                                    y_pred=y_pred.argmax(dim=1))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_loss /= len(self.train_loader)
        train_acc /= len(self.train_loader)
        writer.add_scalar("Train loss", train_loss, epoch)
        writer.add_scalar("Train accuracy", train_acc, epoch)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def _test_step(self,
                   model: torch.nn.Module,
                   loss_fn: torch.nn.Module,
                   epoch: int,
                   writer: SummaryWriter) -> None:
        test_loss, test_acc = 0, 0
        model.to(self.device)
        model.eval()

        with torch.inference_mode():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                test_pred = model(X)

                test_loss += loss_fn(test_pred, y)
                test_acc += self._accuracy_fn(y_true=y,
                    y_pred=test_pred.argmax(dim=1)
                )

            test_loss /= len(self.test_loader)
            test_acc /= len(self.test_loader)
            writer.add_scalar("Test loss", test_loss, epoch)
            writer.add_scalar("Test accuracy", test_acc, epoch)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


    def _accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100
        return acc


if __name__=='__main__':
    dataset_path = 'dataset/'
    batch_size = 128
    num_workers = 6
    epochs = 10
    learning_rate = 0.005
    
    Trainer(dataset_path,
            batch_size,
            num_workers).train(
                epochs,
                learning_rate
            )
