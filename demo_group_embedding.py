#!/usr/bin/env python3

import random
import numpy as np
import torch
import itertools
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import Levenshtein as lev
from lr_scheduler import WarmupMultiStepLR


class Encoder(torch.nn.Module):
    def __init__(
        self, generators,
        letter_dimension=2, dimension=256,
        bidirectional=True, layers=2
    ):
        super().__init__()
        self.generators = generators
        self.letter_dimension = letter_dimension
        self.dimension = dimension

        if self.letter_dimension != self.generators:
            self.embedding = torch.nn.Embedding(
                1 + 2 * self.generators,
                self.letter_dimension
            )
            print("trainable embeddings mode")
        else:
            def embedding(texts):
                binary = F.one_hot(
                    (texts - 1).fmod(self.generators).long() + 1,
                    num_classes=self.generators + 1
                )[..., 1:]
                binary.mul_(torch.sign(
                    -torch.stack([texts, texts], dim=2) + self.generators + 0.5
                ).long())
                return binary.float()

            self.embedding = embedding
            print("crosshair embeddings mode")

        self.lstm = torch.nn.LSTM(
            self.letter_dimension, self.dimension // (
                layers * (2 if bidirectional else 1)),
            num_layers=layers, bidirectional=True, dropout=0.2, batch_first=True
        )

    def forward(self, text, text_lengths):
        packed_embedded = pack_padded_sequence(
            self.embedding(text), text_lengths,
            batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat([hidden[i, :, :]
                           for i in range(hidden.shape[0])], dim=1)
        return hidden


def lcp(strs):
    if len(strs) == 0:
        return ""
    current = strs[0]
    for i in range(1, len(strs)):
        temp = ""
        if len(current) == 0:
            break
        for j in range(len(strs[i])):
            if j < len(current) and current[j] == strs[i][j]:
                temp += current[j]
            else:
                break
        current = temp
    return current


class GroupDataset(Dataset):
    def __init__(self, sample_count, generators):
        self.sample_count = sample_count
        self.generators = generators

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        length = max(1, int(np.random.poisson(lam=5.0)))
        sequence = [random.choice(list(range(1, 1 + 2 * self.generators)))]
        for _ in range(length - 1):
            sequence.append(random.choice(list(
                set(range(1, 1 + 2 * self.generators)) -
                set([1 + (sequence[-1] + 1) % 4])
            )))
        sequence = "".join(map(str, sequence))

        """
        If you want to guarantee the non-reducability

        while any([str(i) + str(1 + (i + 1) % 4) in sequence for i in range(1, 1 + 2 * self.generators)]):
            for i in range(1, 1 + 2 * self.generators):
                sequence = sequence.replace(str(i) + str(1 + (i + 1) % 4), "")
        """

        return {'sequence': (list(map(int, list(sequence))), length)}


def collate_wrapper(batch):
    return pad_sequence([
        torch.IntTensor(sen['sequence'][0])
        for sen in batch
    ], batch_first=True, padding_value=0), [sen['sequence'][1] for sen in batch]


def pairwise_distances(sequences):
    batch_size = sequences.shape[0]
    result = np.zeros(shape=(batch_size, batch_size))
    for i in range(batch_size):
        for j in range(i + 1):
            s1 = "".join(map(lambda x: str(x.item()).strip("0"), sequences[i]))
            s2 = "".join(map(lambda x: str(x.item()).strip("0"), sequences[j]))
            result[i, j] = result[j, i] =\
                len(s1) + len(s2) - 2 * len(lcp([s1, s2]))
    return torch.Tensor(result)


epochs = 40
batch_size = 10
sample_count = 100
steps = 30
generators = 2
group_dataset = GroupDataset(sample_count, generators)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Encoder(generators).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = WarmupMultiStepLR(optimizer, warmup_iters=10)
criterion = torch.nn.MSELoss().to(device)

for _ in range(epochs):
    epoch_loss = 0.0
    iters = 0

    model.train()
    data_loader = itertools.islice(DataLoader(
        group_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0,
        collate_fn=collate_wrapper
    ), batch_size*steps)

    for batch in data_loader:
        optimizer.zero_grad()
        sequences, sequences_lengths = batch

        embeddings = model(sequences, sequences_lengths)
        loss = criterion(
            torch.cdist(embeddings, embeddings, p=2).unsqueeze(0),
            pairwise_distances(sequences).unsqueeze(0)
        ) / (batch_size**2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        iters += 1

    print('loss:', epoch_loss / iters)
