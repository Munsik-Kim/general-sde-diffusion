# src/datasets.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import make_swiss_roll

def get_dataset(name, batch_size=128):
    if name == 'swiss_roll':
        # Swiss Roll 생성
        data, _ = make_swiss_roll(n_samples=50000, noise=0.5)
        data = data[:, [0, 2]] / 10.0 
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
    elif name == 'mnist':
        # MNIST 다운로드 및 로드
        transform = transforms.Compose([
            transforms.ToTensor(),
            # [0, 1] -> [-1, 1] (SDE 학습에 유리)
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        # root='./data'에 다운로드 됩니다.
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    else:
        raise ValueError(f"Unknown dataset: {name}")