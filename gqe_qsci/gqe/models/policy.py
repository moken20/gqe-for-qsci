from abc import ABC, abstractmethod
import torch.nn as nn


class Policy(ABC, nn.Module):
    @abstractmethod
    def act(self, state, temperature):
        pass

    @abstractmethod
    def log_prob(self, indices, temperature):
        pass