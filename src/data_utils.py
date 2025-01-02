from typing import Tuple

import torch
from torch.utils.data import Dataset


class MyTrainDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]
