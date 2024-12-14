import torch
from torch.utils.data import Dataset
import numpy as np
import entropy_core  # Модуль из ДЗ-1, уже установлен внутри контейнера

class ProbDistributionDataset(Dataset):
    def __init__(self, size, vector_dim):
        self.size = size
        self.vector_dim = vector_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Генерируем случайное распределение вероятностей
        probs = np.random.dirichlet(np.ones(self.vector_dim), size=1)[0]
        # Вычисляем энтропию с помощью C++ биндингов
        cpp_entropy = entropy_core.vector_entropy(probs.tolist())
        # Приводим к типу torch.Tensor
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        entropy_tensor = torch.tensor([cpp_entropy], dtype=torch.float32)
        return probs_tensor, entropy_tensor
