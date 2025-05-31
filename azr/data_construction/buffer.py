from collections import deque
from typing import Deque, List, Tuple
import random

class RollingBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buf: Deque[Tuple[str, str, bool]] = deque(maxlen=max_size)

    def add(self, question: str, answer: str, success: bool) -> None:
        self._buf.append((question, answer, success))

    def get_all(self) -> List[Tuple[str, str, bool]]:
        return list(self._buf)

    def success_rate(self) -> float:
        if not self._buf:
            return 0.0
        return sum(int(s) for _, _, s in self._buf) / len(self._buf)

    def sample(self, k: int) -> List[Tuple[str, str, bool]]:
        buf = list(self._buf)
        if not buf:
            return []
        return [random.choice(buf) for _ in range(k)]