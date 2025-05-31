from torch.utils.data import Dataset
from typing import Tuple
from .buffer import RollingBuffer

class QuestionDataset(Dataset):
    def __init__(self, buffer: RollingBuffer):
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer.get_all())

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        q, a, s = self.buffer.get_all()[idx]
        return q, a, int(s)