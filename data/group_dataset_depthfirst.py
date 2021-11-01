import numpy
import random
from torch.utils.data import Dataset


class GroupDatasetDepthFirst(Dataset):
    def __init__(self, max_length, generators):
        self.max_length = max_length
        self.generators = generators
        self.last_element = [1]

    def __len__(self):
        return (2 * self.generators) ** self.max_length

    def __getitem__(self, idx):
        sequence = self.last_element
        
        if self.last_element[-1] != 2 * self.generators:
            self.last_element[-1] += 1
        else:
            i = len(sequence) - 2
            while i > 0:
                pass

        return {'sequence': (sequence, len(sequence))}
