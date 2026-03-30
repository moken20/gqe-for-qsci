# --- replay buffer ---
from collections import deque
import pickle
import sys
from torch.utils.data import Dataset


class ReplayBuffer:
    def __init__(self, size=sys.maxsize, capacity=1000000):
        self.size = size
        self.buf = deque(maxlen=capacity)

    def push(self, seq, energy, old_log_probs):
        self.buf.append((seq, energy, old_log_probs))
        if len(self.buf) > self.size:
            self.buf.popleft()
            
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buf, f)
            
    def load(self, path):
        with open(path, "rb") as f:
            self.buf = pickle.load(f)
            
    def __getitem__(self, idx):
        item = self.buf[idx]
        seq, energy, old_log_probs = item
        return {"idx": seq, "energy": energy, "old_log_probs": old_log_probs}

    def __len__(self):
        return len(self.buf)


class BufferDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer, repetition):
        self.buffer = buffer
        self.repetition = repetition

    def __getitem__(self, idx):
        idx = idx % len(self.buffer)
        sample = self.buffer[idx]
        return {
            "idx": sample["idx"],
            "energy": sample["energy"],
            "old_log_probs": sample["old_log_probs"],
        }
    
    def __len__(self):
        return len(self.buffer) * self.repetition